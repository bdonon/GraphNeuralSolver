import os
import json
import logging
import time
import copy

from tqdm import tqdm

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

from models.layers import FullyConnected, EquilibriumViolation, custom_gather, custom_scatter

class GraphNeuralSolver:

    def __init__(self,
        sess,
        latent_dimension=10,
        hidden_layers=3,
        correction_updates=5,
        non_lin='leaky_relu',
        input_dim=1,
        output_dim=1,
        edge_dim=1,
        minibatch_size=10,
        name='graph_neural_solver',
        directory='./',
        model_to_restore=None,
        default_data_directory='datasets/spring/default'):

        self.sess = sess
        self.latent_dimension = latent_dimension
        self.hidden_layers = hidden_layers
        self.correction_updates = correction_updates
        self.non_lin = non_lin
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.edge_dim = edge_dim
        self.minibatch_size = minibatch_size
        self.name = name
        self.directory = directory
        self.current_train_iter = 0
        self.default_data_directory = default_data_directory

        # Initialize list of trainable variables
        self.trainable_variables = []

        # Reload config if there is a model to restore
        if (model_to_restore is not None) and os.path.exists(model_to_restore):

            logging.info('    Restoring model from '+model_to_restore)

            # Reload the parameters from a json file
            path_to_config = os.path.join(model_to_restore, 'config.json')
            with open(path_to_config, 'r') as f:
                config = json.load(f)
            self.set_config(config)

        # Build weight tensors
        self.build_weights()

        # Build computational graph
        self.build_graph(self.default_data_directory)

        # Restore trained weights if there is a model to restore
        if (model_to_restore is not None) and os.path.exists(model_to_restore):

            # Reload the weights from a ckpt file
            saver = tf.compat.v1.train.Saver(self.trainable_variables)
            path_to_weights = os.path.join(model_to_restore, 'model.ckpt')
            saver.restore(self.sess, path_to_weights)

        # Else, randomly initialize weights
        else:
            self.sess.run(tf.compat.v1.variables_initializer(self.trainable_variables))

        # Log config infos
        self.log_config()


    def build_weights(self):
        """
        Builds all the trainable variables
        """

        # Build weights of each correction update block, and store them
        self.correction_block = {}

        self.phi_from = {}
        self.phi_to = {}

        self.phi_normalizer = {}
        self.correction_normalizer = {}

        for update in range(self.correction_updates):

            self.correction_block[str(update)] = FullyConnected(
                non_lin=self.non_lin,
                latent_dimension=self.output_dim+self.latent_dimension,
                hidden_layers=self.hidden_layers,
                name=self.name+'_correction_block_{}'.format(update),
                input_dim=3*(self.latent_dimension+self.output_dim)+2*self.input_dim
            )
            self.phi_from[str(update)] = FullyConnected(
                non_lin=self.non_lin,
                latent_dimension=self.output_dim+self.latent_dimension,
                hidden_layers=self.hidden_layers,
                name=self.name+'_phi_from_{}'.format(update),
                input_dim=2*(self.latent_dimension+self.output_dim)+2*self.edge_dim
            )
            self.phi_to[str(update)] = FullyConnected(
                non_lin=self.non_lin,
                latent_dimension=self.output_dim+self.latent_dimension,
                hidden_layers=self.hidden_layers,
                name=self.name+'_phi_to_{}'.format(update),
                input_dim=2*(self.latent_dimension+self.output_dim)+2*self.edge_dim
            )
            self.phi_normalizer[str(update)] = FullyConnected(
                hidden_layers=1,
                name=self.name+'_phi_normalizer_{}'.format(update),
                input_dim=2*(self.latent_dimension+self.output_dim)+2*self.edge_dim,
                output_dim=2*(self.latent_dimension+self.output_dim)+2*self.edge_dim
            )
            self.correction_normalizer[str(update)] = FullyConnected(
                hidden_layers=1,
                name=self.name+'_correction_normalizer_{}'.format(update),
                input_dim=3*(self.latent_dimension+self.output_dim)+2*self.input_dim,
                output_dim=3*(self.latent_dimension+self.output_dim)+2*self.input_dim
            )

        self.D = FullyConnected(
            non_lin=self.non_lin,
            latent_dimension=self.output_dim+self.latent_dimension,
            hidden_layers=self.hidden_layers,#1,
            name=self.name+'_D_{}'.format(update),
            input_dim=self.output_dim,
            output_dim=self.output_dim
        )

        # Build operation that computes the distance to the target equation
        self.loss_function = EquilibriumViolation(self.default_data_directory)

    def build_graph(self, default_data_directory):
        """
        Builds the computation graph.
        Assumes that all graphs have been merged into one supergraph
        """

        # Load train and val sets
        A_train_np = np.load(os.path.join(default_data_directory, 'A_train.npy')).astype(np.float32)
        B_train_np = np.load(os.path.join(default_data_directory, 'B_train.npy')).astype(np.float32)
        
        A_valid_np = np.load(os.path.join(default_data_directory, 'A_val.npy')).astype(np.float32)
        B_valid_np = np.load(os.path.join(default_data_directory, 'B_val.npy')).astype(np.float32)

        # Load into tf data
        A_train = tf.data.Dataset.from_tensor_slices(A_train_np)
        B_train = tf.data.Dataset.from_tensor_slices(B_train_np)

        A_valid = tf.data.Dataset.from_tensor_slices(A_valid_np)
        B_valid = tf.data.Dataset.from_tensor_slices(B_valid_np)

        # Build train and valid datasets
        self.train_dataset = tf.data.Dataset.zip((A_train, B_train)).shuffle(100).repeat().batch(self.minibatch_size)
        self.valid_dataset = tf.data.Dataset.zip((A_valid, B_valid)).shuffle(100).repeat().batch(20)

        # Build iterator
        self.iterator = tf.compat.v1.data.Iterator.from_structure(
            tf.compat.v1.data.get_output_types(self.train_dataset),
            None)
        self.next_element = self.iterator.get_next()

        # Build operations to initialize training and validation
        self.training_init_op = self.iterator.make_initializer(self.train_dataset)
        self.validation_init_op = self.iterator.make_initializer(self.valid_dataset)

        # Get the output of the data handler
        self.A, self.B = self.next_element

        self.minibatch_size_tf = tf.shape(self.A)[0]
        self.num_nodes = tf.shape(self.B)[1]
        self.num_edges = tf.shape(self.A)[1]
        self.A_dim = tf.shape(self.A)[2]

        # Extract indices from matrix A
        self.indices_from = tf.cast(self.A[:,:,0], tf.int32)
        self.indices_to = tf.cast(self.A[:,:,1], tf.int32)

        # Extract edge characteristics from matrix A
        self.A_ij = self.A[:,:,2:]

        # Initialize the discount factor that will later be updated
        self.discount = tf.Variable(0., trainable=False)
        self.beta = tf.Variable(0.1, trainable=False)
        self.sess.run(tf.compat.v1.variables_initializer([self.discount, self.beta]))

        # Initialize messages, predictions and losses dict
        self.H = {}
        self.X = {}
        self.loss = {}
        self.error = {}
        self.jacobian = {}
        self.proxy_diff = {}
        self.total_loss = None
        self.Phi_input_norm = {}
        self.correction_input_norm = {}
        self.loss_distrib = {}

        # Initialize latent message and prediction to 0
        self.H['0'] = tf.zeros([self.minibatch_size_tf, self.num_nodes, self.output_dim+self.latent_dimension])
        self.X['0'] = self.D(self.H['0'][:,:,:self.output_dim])

        # Iterate over every correction update
        for update in range(self.correction_updates):


            # Gather messages from both extremities of each edges
            self.H_from = custom_gather(self.H[str(update)], self.indices_from)
            self.H_to = custom_gather(self.H[str(update)], self.indices_to)

            # Concatenate all the inputs of the phi neural network
            self.Phi_input = tf.concat([self.H_from, self.H_to, tf.math.log(tf.abs(self.A_ij)+1e-10), tf.math.sign(self.A_ij)], axis=2)

            # Normalize the input using batch norm. A reshaping step is required for batch norm to work properly, but does not affect dimensions
            #self.Phi_input = tf.reshape(self.Phi_input, [self.minibatch_size_tf * self.num_edges, 2*(self.latent_dimension+self.output_dim)+2*self.edge_dim])

            self.Phi_input_norm[str(update)] = self.phi_normalizer[str(update)](self.Phi_input)

            # Compute the phi using the dedicated neural network blocks
            self.Phi_from = self.phi_from[str(update)](self.Phi_input_norm[str(update)]) #* (1.-self.mask_loop)
            self.Phi_to = self.phi_to[str(update)](self.Phi_input_norm[str(update)]) #* (1.-self.mask_loop)

            # Get the sum of each transformed messages at each node
            self.Phi_from_sum = custom_scatter(
                self.indices_from, 
                self.Phi_from, 
                [self.minibatch_size_tf, self.num_nodes, self.latent_dimension+self.output_dim])
            self.Phi_to_sum = custom_scatter(
                self.indices_to, 
                self.Phi_to, 
                [self.minibatch_size_tf, self.num_nodes, self.latent_dimension+self.output_dim])

            # Concatenate all the inputs of the correction neural network
            self.correction_input = tf.concat([
                self.H[str(update)],
                tf.math.log(tf.abs(self.B)+1e-10), 
                tf.math.sign(self.B),
                self.Phi_from_sum,
                self.Phi_to_sum], axis=2)


            self.correction_input_norm[str(update)] = self.correction_normalizer[str(update)](self.correction_input)

            # Compute the correction using the dedicated neural network block
            self.correction = self.correction_block[str(update)](self.correction_input_norm[str(update)])

            # Apply correction, and extract the predictions from the latent message
            self.H[str(update+1)] = self.H[str(update)] + self.correction * 1e-2

            # Decode the first "output_dim" components of H
            self.X[str(update+1)] = self.D(self.H[str(update+1)][:,:,:self.output_dim])

            # Compute KL divergence between the input of each neural network a gaussian
            self.mean_phi_norm =  tf.reduce_mean(self.Phi_input_norm[str(update)], axis=[1,2])
            self.mean_corr_norm =  tf.reduce_mean(self.correction_input_norm[str(update)], axis=[1,2])

            self.centered_phi_norm = self.Phi_input_norm[str(update)] - tf.reduce_mean(self.Phi_input_norm[str(update)], axis=[0,1], keepdims=True)
            self.centered_phi_norm = tf.expand_dims(self.centered_phi_norm, -1)
            self.centered_phi_norm_T = tf.transpose(self.centered_phi_norm, [0,1,3,2])

            self.centered_corr_norm = self.correction_input_norm[str(update)] - tf.reduce_mean(self.correction_input_norm[str(update)], axis=[0,1], keepdims=True)
            self.centered_corr_norm = tf.expand_dims(self.centered_corr_norm, -1)
            self.centered_corr_norm_T = tf.transpose(self.centered_corr_norm, [0,1,3,2])

            self.correlation_mat_phi = tf.reduce_mean(tf.matmul(self.centered_phi_norm, self.centered_phi_norm_T), axis=[0,1])
            self.correlation_mat_corr = tf.reduce_mean(tf.matmul(self.centered_corr_norm, self.centered_corr_norm_T), axis=[0,1])

            self.correlation_mat_phi_trace = tf.linalg.trace(self.correlation_mat_phi)
            self.correlation_mat_corr_trace = tf.linalg.trace(self.correlation_mat_corr)

            self.correlation_mat_phi_logdet = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(self.correlation_mat_phi)))
            self.correlation_mat_corr_logdet = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(self.correlation_mat_corr)))

            self.loss_distrib[str(update+1)] = tf.reduce_sum(self.mean_phi_norm**2) + tf.reduce_sum(self.mean_corr_norm**2) + \
                self.correlation_mat_phi_trace + self.correlation_mat_corr_trace \
                - self.correlation_mat_phi_logdet - self.correlation_mat_corr_logdet

            # Compute the violation of the desired equation
            self.loss[str(update+1)], self.error[str(update+1)], self.jacobian[str(update+1)], self.proxy_diff[str(update+1)] = self.loss_function(self.X[str(update+1)], self.A, self.B)
            tf.compat.v1.summary.scalar("loss_{}".format(update+1), self.loss[str(update+1)])

            # Compute the discounted loss
            if self.total_loss is None:
                self.total_loss = (self.loss[str(update+1)]+self.beta*self.loss_distrib[str(update+1)]) * self.discount**(self.correction_updates-1-update)
            else:
                self.total_loss += (self.loss[str(update+1)]+self.beta*self.loss_distrib[str(update+1)]) * self.discount**(self.correction_updates-1-update)

        # Get the final prediction and the final loss
        self.X_final = self.X[str(self.correction_updates)]
        self.loss_final = self.loss[str(self.correction_updates)]

        # Initialize the optimizer
        self.learning_rate = tf.Variable(0., trainable=False)
        self.sess.run(tf.compat.v1.variables_initializer([self.learning_rate]))
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)

        # Gradient clipping to avoid 
        self.gradients, self.variables = zip(*self.optimizer.compute_gradients(self.total_loss))
        self.gradients, _ = tf.clip_by_global_norm(self.gradients, 10)

        #with tf.control_dependencies(self.update_ops):
        self.opt_op = self.optimizer.apply_gradients(zip(self.gradients, self.variables))

        self.sess.run(tf.compat.v1.variables_initializer(self.optimizer.variables()))

        # Build summary to visualize the final loss in Tensorboard
        tf.compat.v1.summary.scalar("loss_final", self.loss_final)
        self.merged_summary_op = tf.compat.v1.summary.merge_all()

        # Gather trainable variables
        for update in range(self.correction_updates):
            self.trainable_variables.extend(self.phi_from[str(update)].trainable_variables)
            self.trainable_variables.extend(self.phi_to[str(update)].trainable_variables)
            self.trainable_variables.extend(self.correction_block[str(update)].trainable_variables)
            self.trainable_variables.extend(self.phi_normalizer[str(update)].trainable_variables)
            self.trainable_variables.extend(self.correction_normalizer[str(update)].trainable_variables)
        self.trainable_variables.extend(self.D.trainable_variables)

        

    def log_config(self):
        """
        Logs the config of the whole model
        """

        logging.info('    Configuration :')
        logging.info('        Latent dimensions : {}'.format(self.latent_dimension))
        logging.info('        Number of hidden layers per block : {}'.format(self.hidden_layers))
        logging.info('        Number of correction updates : {}'.format(self.correction_updates))
        logging.info('        Non linearity : {}'.format(self.non_lin))
        logging.info('        Input dimension : {}'.format(self.input_dim))
        logging.info('        Output dimension : {}'.format(self.output_dim))
        logging.info('        Edge dimension : {}'.format(self.edge_dim))
        logging.info('        Current training iteration : {}'.format(self.current_train_iter))
        logging.info('        Model name : ' + self.name)

    def set_config(self, config):
        """
        Sets the config according to an inputed dict
        """

        self.latent_dimension = config['latent_dimension']
        self.hidden_layers = config['hidden_layers']
        self.correction_updates = config['correction_updates']
        self.non_lin = config['non_lin']
        self.output_dim = config['output_dim']
        self.input_dim = config['input_dim']
        self.edge_dim = config['edge_dim']
        self.name = config['name']
        self.directory = config['directory']
        self.current_train_iter = config['current_train_iter']


    def get_config(self):
        """
        Gets the config dict
        """

        config = {
            'latent_dimension': self.latent_dimension,
            'hidden_layers': self.hidden_layers,
            'correction_updates': self.correction_updates,
            'non_lin': self.non_lin,
            'output_dim': self.output_dim,
            'input_dim': self.input_dim,
            'edge_dim': self.edge_dim,
            'name': self.name,
            'directory': self.directory,
            'current_train_iter': self.current_train_iter
        } 
        return config

    def save(self):
        """
        Saves the configuration of the model and the trained weights
        """

        # Save config
        config = self.get_config()
        path_to_config = os.path.join(self.directory, 'config.json')
        with open(path_to_config, 'w') as f:
            json.dump(config, f)

        # Save weights
        saver = tf.compat.v1.train.Saver(self.trainable_variables)
        path_to_weights = os.path.join(self.directory, 'model.ckpt')
        saver.save(self.sess, path_to_weights)


    def train(self, 
        max_iter=10, 
        minibatch_size=10, 
        learning_rate=3e-4, 
        discount=0.9, 
        beta=0.1,
        data_directory='data/',
        save_step=None):
        """
        Performs a training process while keeping track of the validation score
        """

        # Log infos about training process
        logging.info('    Starting a training process :')
        logging.info('        Max iteration : {}'.format(max_iter))
        logging.info('        Minibatch size : {}'.format(minibatch_size))
        logging.info('        Learning rate : {}'.format(learning_rate))
        logging.info('        Discount : {}'.format(discount))
        logging.info('        Beta : {}'.format(beta))
        logging.info('        Training data : {}'.format(data_directory))
        logging.info('        Saving model every {} iterations'.format(save_step))

        # Change dataset
        self.sess.run(self.training_init_op)

        # Build writer dedicated to training for Tensorboard
        self.training_writer = tf.compat.v1.summary.FileWriter(
            os.path.join(self.directory, 'train'))

        # Build writer dedicated to validation for Tensorboard
        self.validation_writer = tf.compat.v1.summary.FileWriter(
            os.path.join(self.directory, 'val'))

        # Set discount factor and learning rate
        self.sess.run(self.discount.assign(discount))
        self.sess.run(self.beta.assign(beta))
        self.sess.run(self.learning_rate.assign(learning_rate))

        # Copy the latest training iteration of the model
        starting_point = copy.copy(self.current_train_iter)


        # Training loop
        for i in tqdm(range(starting_point, starting_point+max_iter)):

            # Store current training step, so that it's always up to date
            self.current_train_iter = i

            # Perform SGD step
            self.sess.run(self.opt_op)

            # Store final loss in a summary
            self.summary = self.sess.run(self.merged_summary_op)
            self.training_writer.add_summary(self.summary, self.current_train_iter)

            # Periodically log metrics and save model
            if ((save_step is not None) & (i % save_step == 0)) or (i == starting_point+max_iter-1):

                # Get minibatch train loss
                loss_final_train = self.sess.run(self.loss_final)

                # Change source data to validation
                self.sess.run(self.validation_init_op)

                # Get minibatch val loss
                loss_final_val = self.sess.run(self.loss_final)

                # Store final loss in validation
                self.summary = self.sess.run(self.merged_summary_op)
                self.validation_writer.add_summary(self.summary, self.current_train_iter)

                # Change source data to validation
                self.sess.run(self.training_init_op)

                # Log metrics
                logging.info('    Learning iteration {}'.format(i))
                logging.info('        Training loss (minibatch) : {}'.format(loss_final_train))
                logging.info('        Validation loss (minibatch): {}'.format(loss_final_val))

                # Save model
                self.save()

        # Save model at the end of training
        self.save()

    def evaluate(self,
        mode='val',
        data_directory='data'):
        """
        Evaluate loss on the desired dataset and stores predictions
        """

        # Import numpy dataset
        data_plot = 'datasets/spring/large'
        data_file = '_test.npy'
        A = np.load(os.path.join(data_directory, 'A_'+mode+'.npy'))
        B = np.load(os.path.join(data_directory, 'B_'+mode+'.npy'))

        # Compute final loss
        loss = self.sess.run(self.loss_final, feed_dict={self.A:A, self.B:B})

        # Compute and save the final prediction
        X_final = self.sess.run(self.X_final, feed_dict={self.A:A, self.B:B})
        np.save(os.path.join(self.directory, 'X_final_pred_'+mode+'.npy'), X_final)

        return loss









