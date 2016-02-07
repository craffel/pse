'''
Utilities for sequence embedding experiments.
'''
import numpy as np
import theano
import scipy.spatial
import pickle
import lasagne
import collections
import theano.tensor as T


def standardize(X):
    ''' Return column vectors to standardize X, via (X - X_mean)/X_std

    Parameters
    ----------
    X : np.ndarray, shape=(n_examples, n_features)
        Data matrix

    Returns
    -------
    X_mean : np.ndarray, shape=(n_features,)
        Mean column vector
    X_std : np.ndarray, shape=(n_features,)
        Standard deviation column vector
    '''
    return np.mean(X, axis=0), np.std(X, axis=0)


def load_data(train_list, valid_list):
    '''
    Load in dataset given lists of files in each split.
    Also standardizes (using train mean/std) the data.
    Each file should be a .npz file with a key 'X' for data in the X modality
    and 'Y' for data in the Y modality.

    Parameters
    ----------
    train_list : list of str
        List of paths to files in the training set.
    valid_list : list of str
        List of paths to files in the validation set.

    Returns
    -------
    X_train : list
        List of np.ndarrays of X modality features in training set
    Y_train : list
        List of np.ndarrays of Y modality features in training set
    X_valid : list
        List of np.ndarrays of X modality features in validation set
    Y_valid : list
        List of np.ndarrays of Y modality features in validation set
    '''
    # We'll use dicts where key is the data subset, so we can iterate
    X = collections.defaultdict(list)
    Y = collections.defaultdict(list)
    for file_list, key in zip([train_list, valid_list],
                              ['train', 'valid']):
        # Load in all files
        for filename in file_list:
            data = np.load(filename)
            # Convert to floatX with correct column order
            X[key].append(np.array(
                data['X'], dtype=theano.config.floatX, order='C'))
            Y[key].append(np.array(
                data['Y'], dtype=theano.config.floatX, order='C'))
        # Get mean/std for training set
        if key == 'train':
            X_mean, X_std = standardize(np.concatenate(X[key], axis=0))
            Y_mean, Y_std = standardize(np.concatenate(Y[key], axis=0))
        # Use training set mean/std to standardize
        X[key] = [(x - X_mean)/X_std for x in X[key]]
        Y[key] = [(y - Y_mean)/Y_std for y in Y[key]]

    return X['train'], Y['train'], X['valid'], Y['valid']


def sample_sequences(X, sample_size):
    ''' Given lists of sequences, crop out or pad sequences to length
    sample_size from each sequence with a random offset

    Parameters
    ----------
    X : list of np.ndarray
        List of sequence matrices, each with shape (n_time_steps, n_features)
    sample_size : int
        The size of the cropped samples from the sequences

    Returns
    -------
    X_sampled : np.ndarray
        Sampled sequences, shape (n_samples, n_time_steps, n_features)
    '''
    X_sampled = []
    for sequence_X in X:
        # If a sequence is smaller than the provided sample size, append 0s
        if sequence_X.shape[0] < sample_size:
            # Append zeros to the sequence to make shape[0] = sample_size
            X_pad = np.zeros(
                (sample_size - sequence_X.shape[0], sequence_X.shape[1]),
                dtype=theano.config.floatX)
            X_sampled.append(np.concatenate((sequence_X, X_pad), axis=0))
        else:
            # Compute a random offset to start cropping from
            offset = np.random.random_integers(
                0, sequence_X.shape[0] - sample_size)
            # Extract a subsequence at the offset
            X_sampled.append(sequence_X[offset:offset + sample_size])
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


def mean_reciprocal_rank(X, Y, indices):
    ''' Computes the mean reciprocal rank of the correct match
    Assumes that X[n] should be closest to Y[n]
    Uses squared euclidean distance

    Parameters
    ----------
    X : np.ndarray, shape=(n_examples, n_features)
        Data matrix in X modality
    Y : np.ndarray, shape=(n_examples, n_features)
        Data matrix in Y modality
    indices : np.ndarray
        Denotes which rows to use in MRR calculation

    Returns
    -------
    mrr_pessimist : float
        Mean reciprocal rank, where ties are resolved pessimistically
        That is, rank = # of distances <= dist(X[:, n], Y[:, n])
    mrr_optimist : float
        Mean reciprocal rank, where ties are resolved optimistically
        That is, rank = # of distances < dist(X[:, n], Y[:, n]) + 1
    '''
    # Compute distances between each codeword and each other codeword
    distance_matrix = scipy.spatial.distance.cdist(X, Y, metric='sqeuclidean')
    # Rank is the number of distances smaller than the correct distance, as
    # specified by the indices arg
    n_le = distance_matrix.T <= distance_matrix[np.arange(X.shape[0]), indices]
    n_lt = distance_matrix.T < distance_matrix[np.arange(X.shape[0]), indices]
    return (np.mean(1./n_le.sum(axis=0)),
            np.mean(1./(n_lt.sum(axis=0) + 1)))


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


def build_network(input_shape, conv_layer_specs, dense_layer_specs):
    '''
    Construct a list of layers of a network given the network's structure.

    Parameters
    ----------
    input_shape : tuple
        Dimensionality of input to be fed into the network
    conv_layer_specs, dense_layer_specs : list of dict
        List of dicts, where each dict corresponds to keyword arguments
        for each subsequent layer.  Note that
        dense_layer_specs[-1]['num_units'] should be the output dimensionality
        of the network.

    Returns
    -------
    layers : dict
        Dictionary which stores important layers in the network, with the
        following mapping: `'in'` maps to the input layer, and `'out'` maps
        to the output layer.
    '''
    # Start with input layer
    layer = lasagne.layers.InputLayer(shape=input_shape)
    # Store a dictionary which conveniently maps names to layers we will need
    # to access later
    layers = {'in': layer}
    # Optionally add convolutional layers
    if len(conv_layer_specs) > 0:
        # Add a "n_channels" dimension to the input
        layer = lasagne.layers.ReshapeLayer(layer, ([0], 1, [1], [2]))
        # Add each layer according to its specification, with some things
        # baked in to all layers
        for layer_spec in conv_layer_specs:
            layer = lasagne.layers.Conv2DLayer(
                layer, stride=(1, 1),
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.HeNormal(), pad='same', **layer_spec)
            layer = lasagne.layers.MaxPool2DLayer(layer, (2, 2))
        # Now, combine the "n_channels" dimension with the "n_features"
        # dimension
        layer = lasagne.layers.DimshuffleLayer(layer, (0, 2, 1, 3))
        layer = lasagne.layers.ReshapeLayer(layer, ([0], [1], -1))
    # Add the attention layer to aggregate over time steps
    # We must force He initialization because Lasagne doesn't like 1-dim
    # shapes in He and Glorot initializers
    layer = AttentionLayer(
        layer, W=lasagne.init.Normal(1./np.sqrt(layer.output_shape[-1])))
    # Add dense layers
    for layer_spec in dense_layer_specs:
        layer = lasagne.layers.DenseLayer(
            layer, W=lasagne.init.HeNormal(), **layer_spec)
    # Keep track of the final layer
    layers['out'] = layer

    return layers


def save_model(param_list, output_file):
    '''
    Write out a pickle file of a network

    Parameters
    ----------
    param_list : list of np.ndarray
        A list of values, per layer, of the parameters of the network
    output_file : str
        Path to write the file to
    '''
    with open(output_file, 'wb') as f:
        pickle.dump(param_list, f)


def load_model(layers, param_file):
    '''
    Load in the parameters from a pkl file into a model

    Parameters
    ----------
    layers : list
        A list of layers which define the model
    param_file : str
        Pickle file of model parameters to load
    '''
    with open(param_file) as f:
        params = pickle.load(f)
    lasagne.layers.set_all_param_values(layers[-1], params)


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
