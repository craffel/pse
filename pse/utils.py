'''
Utilities for sequence embedding experiments.
'''
import numpy as np
import lasagne
import theano.tensor as T


def sample_sequences(X, batch_size):
    ''' Create batches of sequences from a list of sequences, where each
    sequence in each batch is cropped with a random offset so that it has the
    same length as the smallest sequence in the batch.

    Parameters
    ----------
    X : list of np.ndarray
        List of sequence matrices, each with shape
        (n_filters, n_time_steps, n_features)
    batch_size : int
        The number of sequences to include in each batch.

    Returns
    -------
    X_sampled : List of np.ndarray
        List of sampled sequences, each of shape
        (batch_size, n_filters, n_time_steps, n_features)
    '''
    N = len(X)
    # We will populate this list with batches of sampled sequences
    X_sampled = []
    for i in range(0, N, batch_size):
        # Get indices of this minibatch in the list of sequences
        batch = range(i, min(i + batch_size, N))
        # The length of each sequence in this batch will be the length of the
        # shortest sequence in the batch
        size = min(X[n].shape[1] for n in batch)
        # Compute a random crop offset for each sequence in the batch
        offsets = [np.random.random_integers(0, X[n].shape[1] - size)
                   for n in batch]
        # Construct the batch and append it to the output list
        batch_sampled = [X[n][:, o:o + size] for n, o in zip(batch, offsets)]
        X_sampled.append(np.array(batch_sampled))

    return X_sampled


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


def get_next_batch(X, Y, batch_size, n_iter):
    ''' Randomly generates positive and negative example minibatches

    Parameters
    ----------
    X, Y : list of np.ndarray
        Sequence tensors for X/Y modalities
    sample_size_X, sample_size_Y : int
        Sampled sequence length for X/Y modalities
    batch_size : int
        Size of each minibatch to grab
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
            N = len(X)
            # Shuffle X_p and Y_p the same
            positive_shuffle = np.random.permutation(N)
            # Construct batches from each shuffled collection of sequences
            X_p = sample_sequences(
                [X[n] for n in positive_shuffle], batch_size)
            Y_p = sample_sequences(
                [Y[n] for n in positive_shuffle], batch_size)
            # Shuffle X_n and Y_n differently (derangement ensures nothing
            # stays in the same place)
            negative_shuffle_X = np.random.permutation(N)
            negative_shuffle_Y = negative_shuffle_X[random_derangement(N)]
            X_n = sample_sequences(
                [X[n] for n in negative_shuffle_X], batch_size)
            Y_n = sample_sequences(
                [Y[n] for n in negative_shuffle_Y], batch_size)
            # Update the number of batches and reset the current batch index
            n_batches = len(X_p)
            current_batch = 0
        # Yield batch slices
        yield (X_p[current_batch], Y_p[current_batch],
               X_n[current_batch], Y_n[current_batch])
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

    num_units : int
        Dimensionality of the hidden layer.

    W : Theano shared variable, numpy array or callable
        An initializer for the weight matrix of the hidden layer. If a shared
        variable or a numpy array is provided the shape should be (num_inputs,
        num_units).

    b : Theano shared variable, numpy array, callable or None
        An initializer for the bias vector of the hidden layer. If a shared
        variable or a numpy array is provided the shape should be (num_units,).
        If None is provided the hidden layer will have no biases.

    v : Theano shared variable, numpy array or callable
        An initializer for the attention mechanism weight vector. If a shared
        variable or a numpy array is provided the shape should be (num_units,).

    c : Theano shared variable, numpy array, callable or None
        An initializer for the attention mechanism bias scalar. If a shared
        variable or a numpy array is provided the shape should be () (it is a
        scalar).  If None is provided the attention mechanism.

    nonlinearity : callable or None
        The nonlinearity that is applied to the hidden layer activations. If
        None is provided, the layer will be linear.
    '''
    def __init__(self, incoming, num_units, W=lasagne.init.Normal(),
                 b=lasagne.init.Constant(0.), v=lasagne.init.Normal(),
                 c=lasagne.init.Constant(0.),
                 nonlinearity=lasagne.nonlinearities.tanh, **kwargs):
        super(AttentionLayer, self).__init__(incoming, **kwargs)
        # Use identity nonlinearity if provided nonlinearity is None
        self.nonlinearity = (lasagne.nonlinearities.identity
                             if nonlinearity is None else nonlinearity)
        self.num_units = num_units

        # Add hidden weight matrix parameter
        self.W = self.add_param(
            W, (self.input_shape[2], self.num_units), name="W")
        if b is None:
            self.b = None
        else:
            # Add hidden bias vector parameter
            self.b = self.add_param(
                b, (num_units,), name="b", regularizable=False)
        # Add attention weight vector
        self.v = self.add_param(v, (self.num_units,), name="v")
        if c is None:
            self.c = None
        else:
            # Add attention bias scalar parameter
            self.c = self.add_param(c, (), name="c", regularizable=False)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_output_for(self, input, **kwargs):
        # Dot with W to get raw weights, shape=(n_batch, n_steps)
        activation = T.dot(input, self.W)
        # Add bias
        if self.b is not None:
            activation = activation + self.b
        # Apply nonlinearity and aggregate with v
        activation = T.dot(self.nonlinearity(activation), self.v)
        # Add bias
        if self.c is not None:
            activation = activation + self.c
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
