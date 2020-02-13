"""
This script generates train, val and test sets of linear systems.
Each datapoint consists in tuples (A,B,X) such that
    A.X=B
A is described as a sparse matrix, and B and X as vectors

"""

import os
import sys
import json
import shutil
import argparse

import tensorflow as tf
import numpy as np

from models.data_generation import DataGenerator

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str,
    help='Mandatory. Name of data directory that will be created.')
parser.add_argument('--rdm_seed', type=int,
    help='Random seed for data generation.')
parser.add_argument('--gpu', type=bool, default=False,
    help='Use GPUs for data generation.')

parser.add_argument('--train_size', type=int, default=1000,
    help='Number of samples in the train dataset.')
parser.add_argument('--val_size', type=int, default=100,
    help='Number of samples in the validation dataset.')
parser.add_argument('--test_size', type=int, default=100,
    help='Number of samples in the test dataset.')

parser.add_argument('--n_nodes', type=int, default=10,
    help='Number of nodes in the system. '\
    'Constant across train, test and val datasets')
parser.add_argument('--n_edges', type=int, default=15,
    help='Number of edges in the system. '\
    'Constant across train, test and val datasets')

parser.add_argument('--p_cons', type=float, default=0.1,
    help='Probability for each node to be constrained.')

parser.add_argument('--a_distrib', default=['uniform', 0.1, 1.], nargs='+',
    help='Distribution followed by the stiffness of each edge.')
parser.add_argument('--x_c_distrib', default=['uniform', -10., 10.], nargs='+',
    help='Distribution followed by the stiffness of each edge.')
parser.add_argument('--b_distrib', default=['uniform', 0.1, 1.], nargs='+',
    help='Distribution followed by the stiffness of each edge.')

parser.add_argument('--force_type', type=str, default='springs',
    help='Interaction force type. Spring-like by default')


if __name__ == '__main__':

    # Get args
    args = parser.parse_args()

    # Check if data_dir was specified
    if args.data_dir is None:
        sys.exit('Please provide a data_dir!')

    # Check if data_dir already exists
    if os.path.exists(args.data_dir):
        sys.exit('Data directory already exists! '\
            'Please delete/rename the currently existing directory, or provide a different data_dir.')

    # Build the data directory
    os.makedirs(args.data_dir)

    # Setup session
    config = tf.ConfigProto()
    config.allow_soft_placement=True
    config.log_device_placement=False
    if args.gpu:
        config.gpu_options.allow_growth = True

    # Create session
    sess = tf.Session(config=config)

    # Set tensorflow random seed for reproductibility
    if args.rdm_seed is not None:
        tf.set_random_seed(args.rdm_seed)

    # Build a data generator
    data_generator = DataGenerator(
        sess,
        name='DataGenerator',
        n_nodes=args.n_nodes,
        n_edges=args.n_edges,
        p_cons=args.p_cons,
        n_samples=1,
        a_distrib=args.a_distrib,
        x_c_distrib=args.x_c_distrib,
        b_distrib=args.b_distrib
        )

    n_samples = {
        'train': args.train_size,
        'val': args.val_size,
        'test': args.test_size
    }

    for mode in n_samples:
        sess.run(data_generator.n_samples.assign(n_samples[mode]))
        A, B, X = sess.run([data_generator.A, data_generator.B, data_generator.X])
        np.save(os.path.join(args.data_dir, 'A_'+mode+'.npy'), A)
        np.save(os.path.join(args.data_dir, 'B_'+mode+'.npy'), B)
        np.save(os.path.join(args.data_dir, 'X_'+mode+'.npy'), X)

    # Store dataset characteristics in a dict
    dataset_params = {
        'data_dir': args.data_dir,
        'rdm_seed': args.rdm_seed,
        'gpu': args.gpu,
        'train_size': args.train_size,
        'val_size': args.val_size,
        'test_size': args.test_size,
        'n_nodes': args.n_nodes,
        'n_edges': args.n_edges,
        'p_cons': args.p_cons,
        'a_distrib': args.a_distrib,
        'x_c_distrib': args.x_c_distrib,
        'b_distrib': args.b_distrib,
        'force_type': args.force_type
    }

    # Save dataset characteristics
    path_to_config = os.path.join(args.data_dir, 'config.json')
    with open(path_to_config, 'w') as f:
        json.dump(dataset_params, f)

    # Copy the desired force template
    src_force = os.path.join('force_templates', args.force_type+'.py')
    dst_force = os.path.join(args.data_dir, args.force_type+'.py')
    new_dst_force = os.path.join(args.data_dir, 'forces.py')
    shutil.copy(src_force, args.data_dir)
    os.rename(dst_force, new_dst_force)









