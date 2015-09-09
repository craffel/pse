""" Search for good hyperparameters for embedding sequences of audio and MIDI
spectrograms into a common space. """

import functools
import lasagne
import simple_spearmint
import train_network
import glob
import os
import sys
sys.path.append('..')
import utils
import numpy as np
import traceback
import cPickle as pickle

BASE_DATA_DIRECTORY = 'data/'
N_TRIALS = 100
OUTPUT_DIM = 128
RESULTS_PATH = 'parameter_trials'


def layer_specs_from_params(params):
    '''
    Convert params from simple_spearmint to arguments to pass to
    utils.build_network.

    Parameters
    ----------
    params : dict
        Dictionary including keys 'n_conv_layers' and 'n_dense_layers'

    Returns
    -------
    conv_layer_specs, dense_layer_specs : list of dict
        List of dicts, where each dict corresponds to keyword arguments
        for each subsequent layer.  Note that
        dense_layer_specs[-1]['num_units'] should be the output dimensionality
        of the network.
    '''
    # Construct layer specifications from parameters
    # First convolutional layer always has 5x12 filters, rest always 3x3
    conv_layer_specs = [{'filter_size': (5, 12), 'num_filters': 16},
                        {'filter_size': (3, 3), 'num_filters': 32},
                        {'filter_size': (3, 3), 'num_filters': 64}]
    # Truncate the conv_layer_specs list according to how many layers
    conv_layer_specs = conv_layer_specs[:params['n_conv_layers']]
    # ALways have the final dense output layer have OUTPUT_DIM units,
    # optionally have another with 2048 units
    dense_layer_specs = [
        {'num_units': 2048, 'nonlinearity': lasagne.nonlinearities.rectify},
        {'num_units': 2048, 'nonlinearity': lasagne.nonlinearities.rectify},
        {'num_units': OUTPUT_DIM, 'nonlinearity': lasagne.nonlinearities.tanh}]
    dense_layer_specs = dense_layer_specs[-params['n_dense_layers']:]
    return conv_layer_specs, dense_layer_specs


def objective(params, data, output_path):
    """
    Parameters
    ----------
    params : dict
        Dictionary which maps parameter names to their values
    data : dict
        Data dictionary to pass to train_network.train
    output_path : str
        Where to write the results file for this trial.  If the trial fails,
        no results file will be written.

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
    # Convert # layers params to layer specification lists
    conv_layer_specs, dense_layer_specs = layer_specs_from_params(params)
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
    elif params['optimizer'] == 'rmsprop':
        # By abuse of notation, 'momentum' is rho in RMSProp
        optimizer = functools.partial(
            lasagne.updates.rmsprop, learning_rate=learning_rate, rho=momentum)
    elif params['optimizer'] == 'adam':
        # Also abusing, momentum is beta2, except we squash the value so that
        # it is usually close to 1 because small beta2 values don't make
        # much sense
        beta2 = float(np.tanh(momentum*5))
        optimizer = functools.partial(
            lasagne.updates.adam, learning_rate=learning_rate, beta2=beta2)
    # Compute max length as median of lengths
    max_length_X = int(np.median([len(X) for X in data['X_train']]))
    max_length_Y = int(np.median([len(Y) for Y in data['Y_train']]))
    epochs = []
    # Pretty-print epoch status table header
    print "{:>9} | {:>9} | {:>9} | {:>9}".format(
        'iteration', 'objective', 'patience', 'valid cost')
    try:
        # Train the network, accumulating epoch results as we go
        for (e_r, X_p, Y_p) in train_network.train(
                data, max_length_X, max_length_Y, conv_layer_specs,
                dense_layer_specs, params['dense_dropout'],
                float(params['alpha_XY']), float(params['m_XY']),
                optimizer=optimizer):
            # Stop training of a nan training cost is encountered
            if not np.isfinite(e_r['train_cost']):
                break
            epochs.append((e_r, X_p, Y_p))
            # Print status after this epoch
            print "{:>9d} | {:.7f} | {:>9d} | {:.7f}".format(
                e_r['iteration'], e_r['validate_objective'],
                e_r['patience'], e_r['validate_cost'])
            sys.stdout.flush()
    # If there was an error while training, delete epochs to dump NaN epoch
    except Exception:
        print "ERROR: "
        print traceback.format_exc()
        epochs = []
    print

    # Convert params dict to a string of the form
    # param1_name_param1_value_param2_name_param2_value...
    param_string = "_".join(
        '{}_{}'.format(name, val) if type(val) != float else
        '{}_{:.3f}'.format(name, val) for name, val in params.items())
    # Construct a path where the pickle results file will be written
    output_filename = os.path.join(output_path,
                                   "{}.pkl".format(param_string))

    # If no epochs were completed due to an error or NaN cost, write out a NaN
    # epoch and return Nones
    if len(epochs) == 0:
        # Store a fake NaN best epoch
        nan_epoch = {'iteration': 0, 'validate_objective': np.nan,
                     'patience': 0, 'validate_cost': np.nan}
        with open(output_filename, 'wb') as f:
            pickle.dump({'params': params,
                         'best_epoch': nan_epoch}, f)
        return None, None, None

    # Find the index of the epoch with the lowest objective value
    best_epoch_idx = np.argmin([e[0]['validate_objective'] for e in epochs])
    best_epoch, X_params, Y_params = epochs[best_epoch_idx]

    # Store this result
    with open(output_filename, 'wb') as f:
        pickle.dump({'params': params, 'best_epoch': best_epoch}, f)

    return best_epoch, X_params, Y_params


if __name__ == '__main__':

    # Create the results dir if it doesn't exist
    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)

    space = {
        'n_dense_layers': {'type': 'int', 'min': 1, 'max': 3},
        'n_conv_layers': {'type': 'int', 'min': 1, 'max': 3},
        'dense_dropout': {'type': 'enum', 'options': [0, 1]},
        'alpha_XY': {'type': 'float', 'min': 0, 'max': 1},
        'm_XY': {'type': 'int', 'min': 1, 'max': 16},
        'learning_rate_exp': {'type': 'int', 'min': -6, 'max': -2},
        'momentum': {'type': 'float', 'min': 0, 'max': .999},
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

    # Load in all trial results pickle files
    for trial_file in glob.glob(os.path.join(RESULTS_PATH, '*.pkl')):
        with open(trial_file) as f:
            trial = pickle.load(f)
            # Seed the experiment with the result from this trial
            experiment.update(
                trial['params'], trial['best_epoch']['validate_objective'])

    for _ in range(N_TRIALS):
        # Get new parameter suggestion
        params = experiment.suggest()
        print params
        # Train a network with these parameters
        best_epoch, X_params, Y_params = objective(params, data, RESULTS_PATH)
        if best_epoch is not None:
            print "Objective:", best_epoch
            print
            # Update hyperparameter optimizer
            experiment.update(params, best_epoch['validate_objective'])
        else:
            print 'Failed.'
            experiment.update(params, np.nan)
