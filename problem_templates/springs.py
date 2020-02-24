# This file defines the interaction and counteraction forces, as well as their first order derivatives
import tensorflow as tf

class Forces:
	"""
	Defines the elementary forces for interacting springs
	Each force should follow a similar template, that is compatible with tensorflow
	"""
	def __init__(self):
		"""
		Here are the inputs dimensions:
			- X :	[n_samples, n_nodes, d_out]
			- B : 	[n_samples, n_nodes, d_in_B]
			- X_i : [n_samples, n_edges, d_out]
			- X_j : [n_samples, n_edges, d_out]
			- A_ij: [n_samples, n_edges, d_in_A]
		"""
		self.d_F = 1
		self.type = 'Spring-like interaction'

	def F_round(self, X, B):
		return -B 			# tf.int32, [n_samples, n_nodes, d_F]

	def F_bar(self, X_i, X_j, A_ij):
		return A_ij * X_j	# tf.int32, [n_samples, n_edges, d_F]

	def dF_round_dX(self, X, B):
		return 0.*B			# tf.int32, [n_samples, n_nodes, d_F]

	def dF_bar_dX_i(self, X_i, X_j, A_ij):
		return 0.*A_ij		# tf.int32, [n_samples, n_edges, d_F]

	def dF_bar_dX_j(self, X_i, X_j, A_ij):
		return A_ij			# tf.int32, [n_samples, n_edges, d_F]