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


def train(data, sample_size_X, sample_size_Y, conv_layer_specs,
          dense_layer_specs, dense_dropout, alpha_XY, m_XY,
          optimizer=lasagne.updates.rmsprop, batch_size=20, epoch_size=100,
          initial_patience=1000, improvement_threshold=0.99,
          patience_increase=10, max_iter=100000):
    ''' Utility function for training a siamese net for cross-modality hashing
    Assumes data['X_train'][n] should be mapped close to data['Y_train'][m]
    only when n == m

    :parameters:
        - data : dict of list of np.ndarray
            Training/validation sequences in X/Y modality
            Should contain keys X_train, Y_train, X_validate, Y_validate
            Sequence matrix shape=(n_sequences, n_time_steps, n_features)
        - sample_size_X, sample_size_Y : int
            Sampled sequence length for X/Y modalities
        - conv_layer_specs, dense_layer_specs : list of dict
            List of dicts, where each dict corresponds to keyword arguments
            for each subsequent layer.  Note that
            dense_layer_specs[-1]['num_units'] should be the output
            dimensionality of the network.
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
            dense_layer_specs, dense_dropout),
        'Y': utils.build_network(
            (None, None, data['Y_train'][0].shape[-1]), conv_layer_specs,
            dense_layer_specs, dense_dropout)}
    # Inputs to X modality neural nets
    X_p_input = T.tensor3('X_p_input')
    X_n_input = T.tensor3('X_n_input')
    # Y network
    Y_p_input = T.tensor3('Y_p_input')
    Y_n_input = T.tensor3('Y_n_input')

    # Compute \sum max(0, m - ||a - b||_2)^2
    def hinge_cost(m, a, b):
        dist = m - T.sqrt(T.sum((a - b)**2, axis=1))
        return T.mean((dist*(dist > 0))**2)

    def hasher_cost(deterministic):
        X_p_output = lasagne.layers.get_output(
            layers['X']['out'],
            {layers['X']['in']: X_p_input},
            deterministic=deterministic)
        X_n_output = lasagne.layers.get_output(
            layers['X']['out'],
            {layers['X']['in']: X_n_input},
            deterministic=deterministic)
        Y_p_output = lasagne.layers.get_output(
            layers['Y']['out'],
            {layers['Y']['in']: Y_p_input},
            deterministic=deterministic)
        Y_n_output = lasagne.layers.get_output(
            layers['Y']['out'],
            {layers['Y']['in']: Y_n_input},
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
    train = theano.function([X_p_input, X_n_input, Y_p_input, Y_n_input],
                            hasher_cost(False), updates=updates)

    # Compute cost without training
    cost = theano.function([X_p_input, X_n_input, Y_p_input, Y_n_input],
                           hasher_cost(True))

    # Compute output without training
    X_output = theano.function(
        [layers['X']['in'].input_var],
        lasagne.layers.get_output(layers['X']['out'], deterministic=True))
    Y_output = theano.function(
        [layers['Y']['in'].input_var],
        lasagne.layers.get_output(layers['Y']['out'], deterministic=True))

    # Start with infinite validate cost; we will always increase patience once
    current_validate_cost = np.inf
    patience = initial_patience

    # Create sampled sequences for validation
    X_validate = utils.sample_sequences(
        data['X_validate'], sample_size_X)
    Y_validate = utils.sample_sequences(
        data['Y_validate'], sample_size_Y)
    # Create fixed negative example validation set
    X_validate_shuffle = np.random.permutation(X_validate.shape[0])
    Y_validate_shuffle = X_validate_shuffle[
        utils.random_derangement(X_validate.shape[0])]
    # Create iterator to sample sequences from training data
    data_iterator = utils.get_next_batch(
        data['X_train'], data['Y_train'], sample_size_X, sample_size_Y,
        batch_size, max_iter)
    # We will accumulate the mean train cost over each epoch
    train_cost = 0

    for n, (X_p, Y_p, X_n, Y_n) in enumerate(data_iterator):
        # Occasionally Theano was raising a MemoryError, this fails gracefully
        try:
            train_cost += train(X_p, X_n, Y_p, Y_n)
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

            # We need to accumulate the validation cost and network output over
            # batches to avoid MemoryErrors
            epoch_result['validate_cost'] = 0
            validate_batches = 0
            X_val_output = []
            Y_val_output = []
            for batch_idx in range(0, X_validate.shape[0], batch_size):
                # Extract slice from validation set for this batch
                batch_slice = slice(batch_idx, batch_idx + batch_size)
                # Compute and accumulate cost
                epoch_result['validate_cost'] += cost(
                    X_validate[batch_slice],
                    X_validate[X_validate_shuffle][batch_slice],
                    Y_validate[batch_slice],
                    Y_validate[Y_validate_shuffle][batch_slice])
                # Keep track of # of batches for normalization
                validate_batches += 1
                # Compute network output and accumulate result
                X_val_output.append(X_output(X_validate[batch_slice]))
                Y_val_output.append(Y_output(Y_validate[batch_slice]))
            # Normalize cost by number of batches and store
            epoch_result['validate_cost'] /= float(validate_batches)
            # Concatenate per-batch output to tensors
            X_val_output = np.concatenate(X_val_output, axis=0)
            Y_val_output = np.concatenate(Y_val_output, axis=0)
            # Compute in-class and out-of-class distances
            in_dists = np.mean((X_val_output - Y_val_output)**2, axis=1)
            out_dists = np.mean((X_val_output[X_validate_shuffle] -
                                Y_val_output[Y_validate_shuffle])**2, axis=1)
            # Objective is Bhattacharrya coefficient of in-class and
            # out-of-class distances
            epoch_result['validate_objective'] = utils.bhatt_coeff(
                in_dists, out_dists)

            # Test whether this validate cost is the new smallest
            if epoch_result['validate_cost'] < current_validate_cost:
                # To update patience, we must be smaller than
                # improvement_threshold*(previous lowest validation cost)
                patience_cost = improvement_threshold*current_validate_cost
                if epoch_result['validate_cost'] < patience_cost:
                    # Increase patience by the supplied about
                    patience += epoch_size*patience_increase
                # Even if we didn't increase patience, update lowest valid cost
                current_validate_cost = epoch_result['validate_cost']
            # Store patience after this epoch
            epoch_result['patience'] = patience

            # Yield scores and statistics for this epoch
            X_params = lasagne.layers.get_all_param_values(layers['X']['out'])
            Y_params = lasagne.layers.get_all_param_values(layers['Y']['out'])
            yield (epoch_result, X_params, Y_params)

            if n > patience:
                break

    return
