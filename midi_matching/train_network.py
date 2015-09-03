""" Functions for mapping sequences to a common space
"""
import numpy as np
import theano.tensor as T
import theano
import lasagne
import sys
sys.path.append('..')
import utils
import collections


def train(data, conv_layer_specs, lstm_layer_specs, dense_layer_specs,
          bidirectional, dense_dropout, concat_hidden, alpha_XY, m_XY,
          optimizer=lasagne.updates.rmsprop, batch_size=20, epoch_size=100,
          initial_patience=1000, improvement_threshold=0.99,
          patience_increase=10, max_iter=100000):
    ''' Utility function for training a siamese net for cross-modality hashing
    Assumes data['X_train'][n] should be mapped close to data['Y_train'][m]
    only when n == m

    :parameters:
        - data : dict of np.ndarray
            Training/validation sequences/masks in X/Y modality
            Should contain keys X_train, X_train_mask, Y_train, Y_train_mask,
            X_validate, X_validate_mask, Y_validate, Y_validate_mask
            Sequence matrix shape=(n_sequences, n_time_steps, n_features)
            Mask matrix shape=(n_sequences, n_time_steps)
        - conv_layer_specs, lstm_layer_specs, dense_layer_specs : list of dict
            List of dicts, where each dict corresponds to keyword arguments
            for each subsequent layer.  Note that
            dense_layer_specs[-1]['num_units'] should be the output
            dimensionality of the network.
        - bidirectional : bool
            Whether the LSTM layers should be bidirectional or not
        - concat_hidden : bool
            If True, utilize the output of all LSTM layers for output
            compuation
        - dense_dropout : bool
            If True, include dropout between the dense layers
        - alpha_XY : float
            Scaling parameter for cross-modality negative example cost
        - m_XY : int
            Cross-modality negative example threshold
        - optimizer: function
            Function which takes a Theano expression and parameters and
            computes parameter updates to minimize the Theano expression (for
            example, something from lasagne.updates).
        - batch_size : int
            Mini-batch size
        - epoch_size : int
            Number of mini-batches per epoch
        - initial_patience : int
            Always train on at least this many batches
        - improvement_threshold : float
            Validation cost must decrease by this factor to increase patience
        - patience_increase : int
            How many more epochs should we wait when we increase patience
        - max_iter : int
            Maximum number of batches to train on

    :returns:
        - epoch : iterator
            Results for each epoch are yielded
    '''
    # Create networks
    layers = {
        'X': utils.build_network(
            (None, None, data['X_train'][0].shape[-1]), conv_layer_specs,
            lstm_layer_specs, dense_layer_specs, bidirectional, concat_hidden,
            dense_dropout),
        'Y': utils.build_network(
            (None, None, data['Y_train'][0].shape[-1]), conv_layer_specs,
            lstm_layer_specs, dense_layer_specs, bidirectional, concat_hidden,
            dense_dropout)}
    # Inputs to X modality neural nets
    X_p_input = T.tensor3('X_p_input')
    X_p_mask = T.matrix('X_p_mask')
    X_n_input = T.tensor3('X_n_input')
    X_n_mask = T.matrix('X_n_mask')
    # Y network
    Y_p_input = T.tensor3('Y_p_input')
    Y_p_mask = T.matrix('Y_p_mask')
    Y_n_input = T.tensor3('Y_n_input')
    Y_n_mask = T.matrix('Y_n_mask')

    # Compute \sum max(0, m - ||a - b||_2)^2
    def hinge_cost(m, a, b):
        dist = m - T.sqrt(T.sum((a - b)**2, axis=1))
        return T.mean((dist*(dist > 0))**2)

    def hasher_cost(deterministic):
        X_p_output = lasagne.layers.get_output(
            layers['X']['out'],
            {layers['X']['in']: X_p_input, layers['X']['mask']: X_p_mask},
            deterministic=deterministic)
        X_n_output = lasagne.layers.get_output(
            layers['X']['out'],
            {layers['X']['in']: X_n_input, layers['X']['mask']: X_n_mask},
            deterministic=deterministic)
        Y_p_output = lasagne.layers.get_output(
            layers['Y']['out'],
            {layers['Y']['in']: Y_p_input, layers['Y']['mask']: Y_p_mask},
            deterministic=deterministic)
        Y_n_output = lasagne.layers.get_output(
            layers['Y']['out'],
            {layers['Y']['in']: Y_n_input, layers['Y']['mask']: Y_n_mask},
            deterministic=deterministic)
        # Unthresholded, unscaled cost of positive examples across modalities
        cost_p = T.mean(T.sum((X_p_output - Y_p_output)**2, axis=1))
        # Thresholded, scaled cost of cross-modality negative examples
        cost_n = alpha_XY*hinge_cost(m_XY, X_n_output, Y_n_output)
        # Sum positive and negative costs for overall cost
        cost = cost_p + cost_n
        return cost

    # Combine all parameters from both networks
    params = (lasagne.layers.get_all_params(layers['X']['out'])
              + lasagne.layers.get_all_params(layers['Y']['out']))
    # Compute RMSProp gradient descent updates
    updates = optimizer(hasher_cost(False), params)
    # Function for training the network
    train = theano.function([X_p_input, X_p_mask, X_n_input, X_n_mask,
                             Y_p_input, Y_p_mask, Y_n_input, Y_n_mask],
                            hasher_cost(False), updates=updates)

    # Compute cost without training
    cost = theano.function([X_p_input, X_p_mask, X_n_input, X_n_mask,
                            Y_p_input, Y_p_mask, Y_n_input, Y_n_mask],
                           hasher_cost(True))

    # Start with infinite validate cost; we will always increase patience once
    current_validate_cost = np.inf
    patience = initial_patience

    # Create fixed negative example validation set
    X_validate_shuffle = np.random.permutation(data['X_validate'].shape[0])
    Y_validate_shuffle = X_validate_shuffle[
        utils.random_derangement(data['X_validate'].shape[0])]
    data_iterator = utils.get_next_batch(
        data['X_train'], data['X_train_mask'], data['Y_train'],
        data['Y_train_mask'], batch_size, max_iter)
    # We will accumulate the mean train cost over each epoch
    train_cost = 0

    for n, (X_p, X_p_m, Y_p, Y_p_m,
            X_n, X_n_m, Y_n, Y_n_m) in enumerate(data_iterator):
        # Occasionally Theano was raising a MemoryError, this fails gracefully
        try:
            train_cost += train(X_p, X_p_m, X_n, X_n_m, Y_p, Y_p_m, Y_n, Y_n_m)
        except MemoryError as e:
            print "MemoryError: {}".format(e)
            return
        # Stop training if a NaN is encountered
        if not np.isfinite(train_cost):
            print 'Bad training cost {} at iteration {}'.format(train_cost, n)
            break
        # Validate the net after each epoch
        if n and (not n % epoch_size):
            epoch_result = collections.OrderedDict()
            epoch_result['iteration'] = n
            # Compute average training cost over the epoch
            epoch_result['train_cost'] = train_cost / float(epoch_size)
            # Reset training cost mean accumulation
            train_cost = 0
            # Also compute validate cost
            epoch_result['validate_cost'] = 0
            validate_batches = 0
            # We need to accumulate the cost over batches to avoid MemoryErrors
            for n in range(0, data['X_validate'].shape[0], batch_size):
                batch_slice = slice(n, n + batch_size)
                epoch_result['validate_cost'] += cost(
                    data['X_validate'][batch_slice],
                    data['X_validate_mask'][batch_slice],
                    data['X_validate'][X_validate_shuffle][batch_slice],
                    data['X_validate_mask'][X_validate_shuffle][batch_slice],
                    data['Y_validate'][batch_slice],
                    data['Y_validate_mask'][batch_slice],
                    data['Y_validate'][Y_validate_shuffle][batch_slice],
                    data['Y_validate_mask'][Y_validate_shuffle][batch_slice])
                validate_batches += 1
            epoch_result['validate_cost'] /= float(validate_batches)

            if epoch_result['validate_cost'] < current_validate_cost:
                patience_cost = improvement_threshold*current_validate_cost
                if epoch_result['validate_cost'] < patience_cost:
                    patience += epoch_size*patience_increase
                current_validate_cost = epoch_result['validate_cost']

            # Yield scores and statistics for this epoch
            X_params = lasagne.layers.get_all_param_values(layers['X']['out'])
            Y_params = lasagne.layers.get_all_param_values(layers['Y']['out'])
            yield (epoch_result, X_params, Y_params)

            if n > patience:
                break

    return
