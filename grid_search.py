"""
Script to perform a grid search
"""


import os
import sys
import json
import time
import logging
import argparse
import itertools

import tensorflow as tf
# Get rid of the deprecation warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np

from models.models import GraphNeuralSolver 

# Make path absolute
dir = os.path.dirname(os.path.realpath(__file__))

# Build parser
parser = argparse.ArgumentParser()

# Define training parameters
parser.add_argument('--rdm_seed', type=int,
    help='Random seed. Random by default.')
parser.add_argument('--gpu', type=int, default=None,
    help='Use GPUs for data generation.')
parser.add_argument('--profile', type=bool, default=False,
    help='Computational graph profiling, for debug purpose.')
parser.add_argument('--max_iter', type=int, default=1000,
    help='Number of training steps')
parser.add_argument('--minibatch_size', type=int, default=1000,
    help='Size of each minibatch')
parser.add_argument('--track_validation', type=float, default=1000,
    help='Tracking validation metrics every XX iterations')
parser.add_argument('--data_directory', type=str, default='data/',
    help='Path to the folder containing data')

# Define hyperparameters ranges
parser.add_argument('--learning_rate', type=float, default=[3e-2], nargs='+',
    help='Learning rate')
parser.add_argument('--discount', type=float, default=[0.9], nargs='+',
    help='Discount factor for training')
parser.add_argument('--latent_dimension', type=int, default=[10], nargs='+',
    help='Dimension of the latent messages, and of the hidden layers of neural net blocks')
parser.add_argument('--hidden_layers', type=int, default=[2, 3], nargs='+',
    help='Number of hidden layers in each neural network block')
parser.add_argument('--correction_updates', type=int, default=[10], nargs='+',
    help='Number of correction update of the neural network')
parser.add_argument('--non_linearity', type=str, default=['leaky_relu'], nargs='+',
    help='Non linearity of the neural network')

if __name__ == '__main__':

    # Get arguments
    args = parser.parse_args()

    # Set tensorflow random seed for reproductibility, if defined
    if args.rdm_seed is not None:
        tf.set_random_seed(args.rdm_seed)
        #tf.random.set_seed(args.rdm_seed)
        np.random.seed(args.rdm_seed)

    # Select visible GPU
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)

    # Setup session
    config = tf.compat.v1.ConfigProto()
    config.allow_soft_placement=True
    if args.gpu is not None:
        config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)

    # Setup hyperparameter search directory
    grid_search_dir = dir + '/results/grid_search_' + str(int(time.time()))
    if not os.path.exists(grid_search_dir):
        os.makedirs(grid_search_dir)

    # Set logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logFile = os.path.join(grid_search_dir, 'search.log')
    handler = logging.FileHandler(logFile, "w", encoding=None, delay="true")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s: %(message)s", datefmt='%Y-%m-%d %H:%M')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logging.info('')
    logging.info('Starting a grid search for hyperparameters optimization :')
    logging.info('')
    logging.info('    Random seed : {}'.format(args.rdm_seed))
    logging.info('    Max iterations for training : {}'.format(args.max_iter))
    logging.info('    Minibatch size : {}'.format(args.minibatch_size))
    logging.info('    Store validation loss every XX iterations : {}'.format(args.track_validation))
    logging.info('    Data directory : {}'.format(args.data_directory))
    logging.info('')
    logging.info('    Hyperparameters :')
    logging.info('        Learning rate : {}'.format(args.learning_rate))
    logging.info('        Discount factor : {}'.format(args.discount))
    logging.info('        Latent dimension : {}'.format(args.latent_dimension))
    logging.info('        Number of hidden layers : {}'.format(args.hidden_layers))
    logging.info('        Number of correction updates : {}'.format(args.correction_updates))
    logging.info('        Type of non linearity : {}'.format(args.non_linearity))

    # Initialize best validation loss
    best_val_loss = None

    # For each combinations
    for x in itertools.product(
        args.learning_rate, 
        args.discount,
        args.beta,
        args.latent_dimension,
        args.hidden_layers,
        args.correction_updates,
        args.non_linearity
        ):

        # Get combination
        learning_rate, discount, latent_dimension, hidden_layers, correction_updates, non_linearity = x

        # Create a dir that explicitely states all the parameters
        model_dir_name = 'lr_{}_dsct_{}__hdm_{}_hly_{}_cup_{}_nli_{}'.format(learning_rate, 
            discount, latent_dimension, hidden_layers, correction_updates, non_linearity)
        model_dir = os.path.join(grid_search_dir, model_dir_name) 
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        logging.info(" ")
        logging.info("Starting a new model:")

        # Build model
        model = GraphNeuralSolver(
            sess,
            latent_dimension=latent_dimension,
            hidden_layers=hidden_layers,
            correction_updates=correction_updates,
            non_lin=non_linearity,
            input_dim=1,
            output_dim=1,
            minibatch_size=args.minibatch_size,
            name='gns',
            directory=model_dir,
            default_data_directory=args.data_directory,)

        # Train model
        model.train(
            max_iter=args.max_iter,
            learning_rate=learning_rate, 
            discount=discount, 
            data_directory=args.data_directory,
            save_step=args.track_validation,
            profile=args.profile)

        # Evaluate model
        current_val_loss = model.evaluate(mode='val',
            data_directory=os.path.join(dir, args.data_directory))

        logging.info("")
        # Keep track of the best loss
        if best_val_loss is None:
            best_val_loss = current_val_loss
            best_val_dir = model_dir
        elif current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_val_dir = model_dir
            logging.info("    New best model!")
        logging.info("    Observed loss on validation : {}".format(current_val_loss))
        logging.info("    Current best loss on validation : {}".format(best_val_loss))

        # Reset computation graph and session
        tf.reset_default_graph()
        sess = tf.Session(config=config)

    # Once all models have been trained and evaluated, give the dir of the best model
    logging.info(" ")
    logging.info("    Best model in " + best_val_dir)
    logging.info("    Best score : {}".format(best_val_loss))


