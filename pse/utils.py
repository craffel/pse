'''
Utilities for sequence embedding experiments.
'''
import numpy as np
import theano
import lasagne
import theano.tensor as T


def sample_sequences(X, sample_size):
    ''' Given lists of sequences, crop out or pad sequences to length
    sample_size from each sequence with a random offset

    Parameters
    ----------
    X : list of np.ndarray
        List of sequence matrices, each with shape
        (n_filters, n_time_steps, n_features)
    sample_size : int
        The size of the cropped samples from the sequences

    Returns
    -------
    X_sampled : np.ndarray
        Sampled sequences, shape
        (n_samples, n_filters, n_time_steps, n_features)
    '''
    X_sampled = []
    for sequence_X in X:
        # If a sequence is smaller than the provided sample size, append 0s
        if sequence_X.shape[1] < sample_size:
            # Append zeros to the sequence to make shape[0] = sample_size
            X_pad = np.zeros(
                (sequence_X.shape[0], sample_size - sequence_X.shape[1],
                 sequence_X.shape[2]),
                dtype=theano.config.floatX)
            X_sampled.append(np.concatenate((sequence_X, X_pad), axis=1))
        else:
            # Compute a random offset to start cropping from
            offset = np.random.random_integers(
                0, sequence_X.shape[1] - sample_size)
            # Extract a subsequence at the offset
            X_sampled.append(sequence_X[:, offset:offset + sample_size])
    # Combine into new output array
    return np.array(X_sampled)


def random_derangement(n):
    '''
    Permute the numbers up to n such that no number remains in the same place

    Parameters
    ----------
    n : int
        Upper bound of numbers to permute from

    Returns
    -------
    v : np.ndarray, dtype=int
        Derangement indices

    Notes
    -----
        From
        http://stackoverflow.com/questions/26554211/numpy-shuffle-with-constraint
    '''
    while True:
        v = np.arange(n)
        for j in np.arange(n - 1, -1, -1):
            p = np.random.randint(0, j+1)
            if v[p] == j:
                break
            else:
                v[j], v[p] = v[p], v[j]
        else:
            if v[0] != 0:
                return v


def get_next_batch(X, Y, sample_size_X, sample_size_Y, batch_size, n_iter):
    ''' Randomly generates positive and negative example minibatches

    Parameters
    ----------
    X, Y : np.ndarray
        Sequence tensors for X/Y modalities
    sample_size_X, sample_size_Y : int
        Sampled sequence length for X/Y modalities
    batch_size : int
        Size of each minibatch to grab
    sample_size : int
        Size of each sampled sequence
    n_iter : int
        Total number of iterations to run

    Yields
    ------
    X_p, Y_p, X_n, Y_n : np.ndarray
        Positive/negative example/mask minibatch in X/Y modality
    '''
    # These are dummy values which will force the sequences to be sampled
    current_batch = 1
    # We'll only know the number of batches after we sample sequences
    n_batches = 0
    for n in xrange(n_iter):
        if current_batch >= n_batches:
            # Grab sampled sequences of length sample_size_* from X and Y
            X_sampled = sample_sequences(X, sample_size_X)
            Y_sampled = sample_sequences(Y, sample_size_Y)
            N = X_sampled.shape[0]
            n_batches = int(np.floor(N/float(batch_size)))
            # Shuffle X_p and Y_p the same
            positive_shuffle = np.random.permutation(N)
            X_p = np.array(X_sampled[positive_shuffle])
            Y_p = np.array(Y_sampled[positive_shuffle])
            # Shuffle X_n and Y_n differently (derangement ensures nothing
            # stays in the same place)
            negative_shuffle_X = np.random.permutation(N)
            negative_shuffle_Y = negative_shuffle_X[random_derangement(N)]
            X_n = np.array(X_sampled[negative_shuffle_X])
            Y_n = np.array(Y_sampled[negative_shuffle_Y])
            current_batch = 0
        # Yield batch slices
        batch = slice(current_batch*batch_size, (current_batch + 1)*batch_size)
        yield (X_p[batch], Y_p[batch], X_n[batch], Y_n[batch])
        current_batch += 1


class AttentionLayer(lasagne.layers.Layer):
    '''
    A layer which computes a weighted average across the second dimension of
    its input, where the weights are computed according to the third dimension.
    This results in the second dimension being flattened.  This is an attention
    mechanism - which "steps" (in the second dimension) are attended to is
    determined by a learned transform of the features.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape

    W : Theano shared variable, numpy array or callable
        An initializer for the weights of the layer. If a shared variable or a
        numpy array is provided the shape should  be (num_inputs,).

    b : Theano shared variable, numpy array, callable or None
        An initializer for the biases of the layer. If a shared variable or a
        numpy array is provided the shape should be () (it is a scalar).
        If None is provided the layer will have no biases.

    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If None
        is provided, the layer will be linear.
    '''
    def __init__(self, incoming, W=lasagne.init.Normal(),
                 b=lasagne.init.Constant(0.),
                 nonlinearity=lasagne.nonlinearities.tanh,
                 **kwargs):
        super(AttentionLayer, self).__init__(incoming, **kwargs)
        # Use identity nonlinearity if provided nonlinearity is None
        self.nonlinearity = (lasagne.nonlinearities.identity
                             if nonlinearity is None else nonlinearity)

        # Add weight vector parameter
        self.W = self.add_param(W, (self.input_shape[2],), name="W")
        if b is None:
            self.b = None
        else:
            # Add bias scalar parameter
            self.b = self.add_param(b, (), name="b", regularizable=False)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_output_for(self, input, **kwargs):
        # Dot with W to get raw weights, shape=(n_batch, n_steps)
        activation = T.dot(input, self.W)
        # Add bias
        if self.b is not None:
            activation = activation + self.b
        # Apply nonlinearity
        activation = self.nonlinearity(activation)
        # Perform softmax
        activation = T.exp(activation)
        activation /= activation.sum(axis=1).dimshuffle(0, 'x')
        # Weight steps
        weighted_input = input*activation.dimshuffle(0, 1, 'x')
        # Compute weighted average (summing because softmax is normed)
        return weighted_input.sum(axis=1)


def bhatt_coeff(x, y, bins=20):
    '''
    Compute the Bhattacharyya distance between samples of two random variables.

    Parameters
    ----------
    x, y : np.ndarray
        Samples of two random variables.

    bins : int
        Number of bins to use when approximating densities.

    Returns
    -------
    bhatt_coeff : float
        Bhattacharyya coefficient.
    '''
    # Find histogram range - min to max
    bounds = [min(min(x), min(y)), max(max(x), max(y))]
    # Compute histograms
    x_hist = np.histogram(x, bins=bins, range=bounds)[0]
    y_hist = np.histogram(y, bins=bins, range=bounds)[0]
    # Normalize
    x_hist = x_hist.astype(float)/x_hist.sum()
    y_hist = y_hist.astype(float)/y_hist.sum()
    # Compute Bhattacharyya coefficient
    return np.sum(np.sqrt(x_hist*y_hist))
