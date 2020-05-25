import os
import time
import argparse


import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline


# Get rid of the deprecation warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


from tensorflow.python.lib.io import tf_record
from tensorflow.core.util import event_pb2
from tensorflow.python.framework import tensor_util

from models.models import GraphNeuralSolver

# Build parser
parser = argparse.ArgumentParser()

# Define mode
parser.add_argument('--model_path', type=str,
    help='Path to model directory')
parser.add_argument('--data_path', type=str,
    help='Path to data directory')

parser.add_argument('--minibatch_size', type=int, default=[10, 100, 1000], nargs='+',
    help='Minibatch that are to be tested')


if __name__ == '__main__':

    # Get arguments
    args = parser.parse_args()

    # Setup session
    config = tf.compat.v1.ConfigProto()
    config.allow_soft_placement=True
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)

    # Build the Graph Neural Solver
    model = GraphNeuralSolver(sess, 
                              model_to_restore=args.model_path, 
                              default_data_directory=args.data_path)

    # Importing test data
    A_np = np.load(os.path.join(args.data_path, 'A_test.npy'))
    B_np = np.load(os.path.join(args.data_path, 'B_test.npy'))

    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    i = 0

    
    for size in args.minibatch_size:

        # Build feed_dict
        feed_dict = {
            model.A:A_np[:size], 
            model.B:B_np[:size]
        }

        # Use the model to predict
        X_pred = sess.run(model.X_final, feed_dict=feed_dict)
        X_pred = sess.run(model.X_final, feed_dict=feed_dict, options=options, run_metadata=run_metadata)
        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        with open('profile_{}.json'.format(i), 'w') as f:
            f.write(chrome_trace)
        i += 1

        start  = time.time()
        X_pred = sess.run(model.X_final, feed_dict=feed_dict)
        end = time.time()
        print('Took {}s for {} samples'.format(end-start, size))
