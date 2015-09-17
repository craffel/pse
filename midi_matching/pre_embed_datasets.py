""" Embed all sequences from the MSD and the clean MIDIs. """
import train_best_network
import train_network
import sys
sys.path.append('..')
import utils
import cPickle as pickle
import lasagne
import os
import glob
import numpy as np
import theano
import time

RESULTS_PATH = 'parameter_trials'
MODEL_PATH = 'best_model'
BASE_DATA_PATH = 'data'

if __name__ == '__main__':
    # Load in training files
    X_train = []
    Y_train = []
    for filename in glob.glob(os.path.join(BASE_DATA_PATH, 'train', '*.npz')):
        data = np.load(filename)
        # Convert to floatX with correct column order
        X_train.append(np.array(
            data['X'], dtype=theano.config.floatX, order='C'))
        Y_train.append(np.array(
            data['Y'], dtype=theano.config.floatX, order='C'))
    # Stack to compute training mean and std
    X_mean, X_std = utils.standardize(np.concatenate(X_train, axis=0))
    Y_mean, Y_std = utils.standardize(np.concatenate(Y_train, axis=0))
    # Compute max length as median of lengths
    max_length_X = int(np.median([len(X) for X in X_train]))
    max_length_Y = int(np.median([len(Y) for Y in Y_train]))

    # Retrieve the hyperparameters which achivieved the lowest objective
    best_params, _ = train_best_network.get_best_trial(RESULTS_PATH)
    # Convert parameters to layer specifications
    (conv_layer_specs,
     dense_layer_specs) = train_network.layer_specs_from_params(best_params)
    # Build networks
    layers = {
        'X': utils.build_network(
            (None, None, X_train[0].shape[-1]), conv_layer_specs,
            dense_layer_specs),
        'Y': utils.build_network(
            (None, None, Y_train[0].shape[-1]), conv_layer_specs,
            dense_layer_specs)}
    # Load in parameters of trained best-performing networks
    with open(os.path.join(MODEL_PATH, 'X_params.pkl')) as f:
        X_params = pickle.load(f)
    # Set parameter values of build network
    lasagne.layers.set_all_param_values(layers['X']['out'], X_params)
    with open(os.path.join(MODEL_PATH, 'Y_params.pkl')) as f:
        Y_params = pickle.load(f)
    lasagne.layers.set_all_param_values(layers['Y']['out'], Y_params)
    # Compile functions for embedding with each network
    embed_X = theano.function([layers['X']['in'].input_var],
                              lasagne.layers.get_output(layers['X']['out']))
    embed_Y = theano.function([layers['Y']['in'].input_var],
                              lasagne.layers.get_output(layers['Y']['out']))

    # Construct glob of msd cqt npy files
    Y_files = glob.glob(os.path.join(
        BASE_DATA_PATH, 'msd_cqt', '*', '*', '*', '*.npy'))
    # Pre-allocate embedding matrix
    embedded_Y = np.empty((len(Y_files), train_network.OUTPUT_DIM),
                          dtype=theano.config.floatX)
    # Keep track of which row in embedded_Y corresponds to which MSD ID
    npy_mapping = []
    # We will process this many sequences at a time
    batch_size = 200
    # Keep track of progress and time
    print 'MSD:'
    print '',
    now = time.time()
    # Iterate over batch start index
    for batch_idx in range(0, len(Y_files), batch_size):
        # Store the npy files corresponding to this batch
        npy_mapping += Y_files[batch_idx:batch_idx + batch_size]
        # Load in a list of this batch
        batch = [(np.load(Y_files[n]) - Y_mean)/Y_std for n in
                 range(batch_idx, batch_idx + batch_size)
                 if n < len(Y_files)]
        # Randomly sample subsequences of this batch
        batch = utils.sample_sequences(batch, max_length_Y)
        # Embed the batch and store the embedding in the matrix
        embedded_Y[batch_idx:batch_idx + batch_size] = embed_Y(batch)
        # Report percent done and time
        print "\r{:.3f}% in {}s".format(
            (100.*(batch_idx + batch_size))/len(Y_files),
            time.time() - now),
        now = time.time()
        sys.stdout.flush()
    print 'Done.'
    # Write out embedding matrix and mapping
    np.save('data/msd_embedded.npy', embedded_Y)
    with open('data/msd_embedded_mapping.pkl', 'wb') as f:
        pickle.dump(npy_mapping, f)

    # Embed MIDI cqts from dev and test sets
    for dataset in ['dev', 'test']:
        # Where will we write out the files?
        output_path = os.path.join(BASE_DATA_PATH,
                                   '{}_embedded'.format(dataset))
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        # Get all CQT npy files
        X_files = glob.glob(os.path.join(BASE_DATA_PATH, dataset, '*.npy'))
        # Load in all CQTs from the files
        X_cqts = []
        for filename in X_files:
            # Convert to floatX with correct column order
            X_cqts.append(np.array(
                (np.load(filename) - X_mean)/X_std,
                dtype=theano.config.floatX, order='C'))
        # Randomly sample subsequences
        X_cqts = utils.sample_sequences(X_cqts, max_length_X)
        # Process fewer sequences at once because they are longer
        batch_size = 50
        print '{}:'.format(dataset)
        print '',
        # Iterate over batch start index
        for batch_idx in range(0, X_cqts.shape[0], batch_size):
            # Extract the batch
            batch = X_cqts[batch_idx:batch_idx + batch_size]
            # Embed the batch
            batch_embedded = embed_X(batch)
            # Write out each embedded sequence in the batch
            for n in range(batch_size):
                if batch_idx + n >= X_cqts.shape[0]:
                    break
                output_file = os.path.join(
                    output_path, os.path.split(X_files[batch_idx + n])[1])
                np.save(output_file, batch_embedded[n])
            # Report percent done
            print "\r{:.3f}%".format(
                (100.*(batch_idx + batch_size))/len(X_files)),
            sys.stdout.flush()
        print 'Done.'
