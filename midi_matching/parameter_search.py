""" Search for good hyperparameters for embedding sequences of audio and MIDI
spectrograms into a common space. """

import simple_spearmint
import glob
import os
import sys
sys.path.append('..')
import utils
import cPickle as pickle
import train_network

BASE_DATA_DIRECTORY = 'data/'
N_TRIALS = 100
RESULTS_PATH = 'parameter_trials'


def write_result(params, best_epoch, output_path):
    """ Write the result of a parameter search trial, i.e. a pickle file with
    the parameters used and the best epoch during training.

    Parameters
    ----------
    params : dict
        Dictionary which maps parameter names to their values
    data : dict
        Data dictionary to pass to train_network.train
    output_path : str
        Where to write the results file for this trial.  If the trial fails,
        no results file will be written.
    """
    # Convert params dict to a string of the form
    # param1_name_param1_value_param2_name_param2_value...
    param_string = "_".join(
        '{}_{}'.format(name, val) if type(val) != float else
        '{}_{:.3f}'.format(name, val) for name, val in params.items())
    # Construct a path where the pickle results file will be written
    output_filename = os.path.join(
        output_path, "{}.pkl".format(param_string))

    # Store this result
    with open(output_filename, 'wb') as f:
        pickle.dump({'params': params, 'best_epoch': best_epoch}, f)


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
        best_epoch, X_params, Y_params = train_network.objective(params, data)
        write_result(params, best_epoch, RESULTS_PATH)
        print "Objective:", best_epoch
        print
        # Update hyperparameter optimizer
        experiment.update(params, best_epoch['validate_objective'])
