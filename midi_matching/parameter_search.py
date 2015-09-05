""" Search for good hyperparameters for embedding sequences of audio and MIDI
spectrograms into a common space. """

import functools
import lasagne
import simple_spearmint
import train_network
import lasagne
import glob
import os
import sys
sys.path.append('..')
import utils
import numpy as np

BASE_DATA_DIRECTORY = 'data/'
N_TRIALS = 100
OUTPUT_DIM = 128


def objective(params, data):
    """
    Parameters
    ----------
    params : dict
        Dictionary which maps parameter names to their values
    data : dict
        Data dictionary to pass to train_network.train

    Returns
    -------
    best_epoch : dict
        Results dictionary for the epoch with the lowest cost
    best_X_params, best_Y_params : dict
        X and Y network parameters with the lowest validation cost

    Note
    ----
    Will return None, None, None if training diverged before one epoch
    """
    # Construct layer specifications from parameters
    # First convolutional layer always has 5x12 filters, second always 3x3
    conv_layer_specs = [{'filter_size': (5, 12), 'num_filters': 16},
                        {'filter_size': (3, 3), 'num_filters': 32}]
    # Truncate the conv_layer_specs list according to how many layers
    conv_layer_specs = conv_layer_specs[:params['n_conv_layers']]
    # Construct the LSTM gate nitializer from the supplied weight matrix std
    gate = lasagne.layers.Gate(
        W_in=lasagne.init.Normal(10**params['lstm_W_std_exponent']),
        W_hid=lasagne.init.Normal(10**params['lstm_W_std_exponent']),
        W_cell=lasagne.init.Normal(10**params['lstm_W_std_exponent']))
    # Forget gate optionally has a different (higher) bias value
    forget_gate = lasagne.layers.Gate(
        W_in=lasagne.init.Normal(10**params['lstm_W_std_exponent']),
        W_hid=lasagne.init.Normal(10**params['lstm_W_std_exponent']),
        W_cell=lasagne.init.Normal(10**params['lstm_W_std_exponent']),
        b=lasagne.init.Constant(params['forget_bias']))
    # Cell is a different gate, with W_cell=None and tanh nonlinearity
    cell = lasagne.layers.Gate(
        W_in=lasagne.init.Normal(10**params['lstm_W_std_exponent']),
        W_hid=lasagne.init.Normal(10**params['lstm_W_std_exponent']),
        W_cell=None,
        nonlinearity=lasagne.nonlinearities.tanh)
    # Construct prototype LSTM layer spec; they all have the same structure
    lstm_layer_specs = [{'num_units': 100, 'ingate': gate, 'forgetgate':
                         forget_gate, 'outgate': gate, 'cell': cell,
                         'grad_clipping': params['grad_clipping'],
                         'gradient_steps': params['gradient_steps']}]
    # Copy single LSTM layer spec n_lstm_layers times
    lstm_layer_specs = lstm_layer_specs*params['n_lstm_layers']
    # ALways have the final dense output layer have OUTPUT_DIM units,
    # optionally have another with 2048 units
    dense_layer_specs = [
        {'num_units': 2048, 'nonlinearity': lasagne.nonlinearities.rectify},
        {'num_units': OUTPUT_DIM, 'nonlinearity': lasagne.nonlinearities.tanh}]
    dense_layer_specs = dense_layer_specs[-params['n_dense_layers']:]
    # Convert learning rate exponent to actual learning rate
    learning_rate = float(10**params['learning_rate_exp'])
    # Avoid upcasting from using a 0d ndarray, use python float instead.
    momentum = float(params['momentum'])
    # Create partial functions for each optimizer with the learning rate and
    # momentum filled in
    if params['optimizer'] == 'NAG':
        optimizer = functools.partial(
            lasagne.updates.nesterov_momentum, learning_rate=learning_rate,
            momentum=momentum)
    # By abuse of notation, 'momentum' is rho in RMSProp and beta2 in adam
    elif params['optimizer'] == 'rmsprop':
        optimizer = functools.partial(
            lasagne.updates.rmsprop, learning_rate=learning_rate, rho=momentum)
    elif params['optimizer'] == 'adam':
        optimizer = functools.partial(
            lasagne.updates.adam, learning_rate=learning_rate, beta2=momentum)
    # Compute max length as median of lengths
    max_length_X = int(np.median([len(X) for X in data['X_train']]))
    max_length_Y = int(np.median([len(Y) for Y in data['Y_train']]))
    # Train the network, accumulating epoch results as we go
    epochs = [(e_r, X_p, Y_p) for (e_r, X_p, Y_p) in train_network.train(
        data, max_length_X, max_length_Y, conv_layer_specs, lstm_layer_specs,
        dense_layer_specs, params['bidirectional'], params['dense_dropout'],
        params['concat_hidden'], float(params['alpha_XY']),
        float(params['m_XY']), optimizer=optimizer,
        # TODO
        max_iter=1001)]
    # If no epochs were completed due to an error or NaN cost, return Nones
    if len(epochs) == 0:
        return None, None, None
    # Find the index of the epoch with the lowest validate cost
    best_epoch = np.argmin([e[0]['validate_cost'] for e in epochs])
    return epochs[best_epoch]


if __name__ == '__main__':
    space = {
        'n_lstm_layers': {'type': 'int', 'min': 1, 'max': 3},
        'n_dense_layers': {'type': 'int', 'min': 1, 'max': 2},
        'n_conv_layers': {'type': 'int', 'min': 0, 'max': 2},
        'bidirectional': {'type': 'enum', 'options': [0, 1]},
        'dense_dropout': {'type': 'enum', 'options': [0, 1]},
        'concat_hidden': {'type': 'enum', 'options': [0, 1]},
        'forget_bias': {'type': 'enum', 'options': [0, 1, 2, 5, 10]},
        'grad_clipping': {'type': 'enum', 'options': [0, 1, 10]},
        'gradient_steps': {'type': 'enum', 'options': [-1, 100]},
        'alpha_XY': {'type': 'float', 'min': 0, 'max': 1},
        'm_XY': {'type': 'int', 'min': 1, 'max': 16},
        'lstm_W_std_exponent': {'type': 'int', 'min': -3, 'max': -1},
        'learning_rate_exp': {'type': 'int', 'min': -6, 'max': -2},
        'momentum': {'type': 'enum', 'options': [.9, .99, .999]},
        'optimizer': {'type': 'enum', 'options': ['NAG', 'rmsprop', 'adam']}}

    experiment = simple_spearmint.SimpleSpearmint(space, noiseless=False)

    train_list = list(glob.glob(os.path.join(
        BASE_DATA_DIRECTORY, 'train', '*.npz')))
    valid_list = list(glob.glob(os.path.join(
        BASE_DATA_DIRECTORY, 'valid', '*.npz')))
    # Load in the data
    (X_train, Y_train, X_validate, Y_validate) = utils.load_data(
        train_list, valid_list)
    # Convert to data dict
    # TODO: Do this in utils
    data = {}
    data['X_train'] = X_train
    data['X_validate'] = X_validate
    data['Y_train'] = Y_train
    data['Y_validate'] = Y_validate

    for _ in range(N_TRIALS):
        # Get new parameter suggestion
        params = experiment.suggest()
        print params
        # Train a network with these parameters
        best_epoch, X_params, Y_params = objective(params, data)
        print "Objective:", best_epoch
        print
        # Update hyperparameter optimizer
        experiment.update(params, best_epoch['validate_cost'])
