import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Reshape, Flatten
import t3f

import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)


class SoftmaxEmbeddingLayer(tf.keras.layers.Layer):
	"""
	Parameter embedding layer that generates the weights used for stacking the tensor networks. It
	takes the parameter array, lambda = (ell, a1, a2), as input and outputs K numbers that sum to 1.

	Attributes:
		output_dim (int): The dimension of the output
		expansion_dim (int): The dimension used for expanding the input in intermediate layers.
	"""

	def __init__(self, output_dim, d, expansion_dim = 30, **kwargs):
		super(SoftmaxEmbeddingLayer, self).__init__(**kwargs)
		self.reduction_layer = None
		self.expansion_layers = None
		self.output_dim = output_dim
		self.expansion_dim = expansion_dim
		self.d = d  # Number of dense layers

	def build(self, input_shape):
		# Expansion layers to increase dimensionality
		self.expansion_layers = [tf.keras.layers.Dense(self.expansion_dim, activation = 'relu') for _ in range(self.d)]
		# Reduction layer to bring dimensionality back to the desired output dimension
		self.reduction_layer = tf.keras.layers.Dense(self.output_dim)

	def call(self, inputs):
		expanded = inputs
		for layer in self.expansion_layers:
			expanded = layer(expanded)
		return tf.nn.softmax(self.reduction_layer(expanded))

	def get_config(self):
		return {'output_dim': self.output_dim, 'expansion_dim': self.expansion_dim}


class EinsumTTLRegularizer(tf.keras.regularizers.Regularizer):
	"""
	Regularizer for the Einsum layer of the TTL layer class, penalizing high-frequency components of the
	weights vector.

	Attributes:
		strength (float): The regularization strength.
		midpoint (int): Index demarcating the inner and outer boundaries, i.e. x[:midpoint] contains
						data for the inner boundary, and x[midpoint:] contains data for the outer boundary.
						The regularization is designed so it does not penalize variations across this index.
	"""

	def __init__(self, strength, midpoint):
		self.strength = strength
		self.midpoint = midpoint

	def __call__(self, x):
		diff = tf.abs(x[1:self.midpoint - 1] - x[0:self.midpoint - 2]) \
			   + tf.abs(x[self.midpoint + 1:2 * self.midpoint - 1] - x[self.midpoint:2 * self.midpoint - 2])
		return self.strength * tf.reduce_sum(diff)

	def get_config(self):
		return {'strength': self.strength, 'midpoint': self.midpoint}


def cosine_initializer(kx = 1.0):
	"""
	Initializer for the Einsum layer of the TTL layer class. Sets the weights to a linear combination
	of cos(kx * x) and cos(2 * kx * x), where x is the weight vector.

	Args:
		kx (float, optional): Frequency of the cosine terms. Defaults to 1.0.

	Returns:
		_initializer: Weight initializer function
	"""

	def _initializer(shape, dtype = None):
		x_values = np.linspace(-np.pi, np.pi, shape[0])
		cos_values = np.random.uniform(-0.1, 0.3) * np.abs(np.cos(kx * x_values)) \
					 + np.random.uniform(-0.05, 0.05) * np.abs(np.cos(2.0 * kx * x_values))
		return tf.convert_to_tensor(-cos_values, dtype = dtype)

	return _initializer


class EinsumTTL(tf.keras.layers.Layer):
	"""
	Layer that contracts the input tensor over the second dimension before passing it to the TTL.
	If regularization is enabled, it applies an `EinsumTTLRegularizer` to the kernels.

	Attributes:
	    (nx2, nx3) (integers): Shape parameters characterizing input tensor dimensions. T
	                           The shape of the input tensor is (2*nx2, nx3//2).
	    W (int): Number of einsum contractions
	    kernels (list): List of weight matrices for each einsum contraction
	    regularization_strength (float): The strength of the regularization if used.
	    use_regularization (bool): Flag to indicate whether regularization is used.
	"""

	def __init__(self, nx2, nx3, W, use_regularization, regularization_strength = 0.005, **kwargs):
		super(EinsumTTL, self).__init__(**kwargs)
		self.nx2 = nx2
		self.nx3 = nx3
		self.W = W
		self.kernels = []

		self.regularization_strength = regularization_strength
		self.use_regularization = use_regularization
		if self.use_regularization:
			regularizer = EinsumTTLRegularizer(self.regularization_strength, self.nx3 // 4)
		else:
			regularizer = None

		initializer_values_ = [1.0, 0.5, 2.0, 3.0] * W
		initializer_values = initializer_values_[:W]
		for i in range(W):
			self.kernels.append(self.add_weight(
				name = f'w{i + 1}',
				shape = (nx3 // 2,),
				regularizer = regularizer,
				initializer = cosine_initializer(initializer_values[i])
			))

	def call(self, inputs):
		parts = []
		for w in self.kernels:
			part_a = tf.einsum('abc,c->ab', inputs[:, :self.nx2, :self.nx3 // 4], w[:self.nx3 // 4]) + \
					 tf.einsum('abc,c->ab', inputs[:, :self.nx2, self.nx3 // 4:self.nx3 // 2],
							   tf.reverse(w[:self.nx3 // 4], axis = [0]))
			part_b = tf.einsum('abc,c->ab', inputs[:, self.nx2:, :self.nx3 // 4], w[self.nx3 // 4:self.nx3 // 2]) + \
					 tf.einsum('abc,c->ab', inputs[:, self.nx2:, self.nx3 // 4:self.nx3 // 2],
							   tf.reverse(w[self.nx3 // 4:self.nx3 // 2], axis = [0]))
			parts.extend([part_a, part_b])

		return tf.concat(parts, axis = 1)

	def get_config(self):
		return {'use_regularization': self.use_regularization,
				'regularization_strength': self.regularization_strength}


class TTL(tf.keras.layers.Layer):
	"""
	TTL (Tensor Train Layer) is a custom TensorFlow Keras layer that builds a model
	based on the given configuration. This layer is designed to work with 
	tensor train decomposition in neural networks.

	Attributes:
		config (dict): Configuration dictionary containing parameters for the model.
		
			'nx1', 'nx2', 'nx3': Integers, dimensions of the finite-difference grid
			
			'shape1': List of integers, defines the shape of the output tensor in the tensor train format.
					  The length of shape1 must match the length of shape2.
			'shape2': List of integers, specifies the shape of the input tensor in the tensor train format.
					  The length of shape2 must match the length of shape1.
			'ranks':  List of integers, represents the ranks in the tensor train decomposition.
					  The length of this list determines the complexity and the number of parameters in the tensor train layer.
			'W' (int): 	Number of weight vectors to use in the initial EinsumTTL layer. Setting W = 0 means that no EinsumTLL
						used.

			'use_regularization' (boolean, optional, default: False):  Indicates whether regularization is used in the EinsumTTL.
			'regularization_strength' (float, optional, default: 0): Strength of the regularization
			
		model (tf.keras.Sequential): The Sequential model built based on the provided configuration.

	Methods:
		load_config(self, config): Loads configuration
		build_model(self): Builds the layer
		call(inputs): Method for the forward pass of the layer.
	"""

	def __init__(self, config, **kwargs):
		super(TTL, self).__init__(**kwargs)
		self.model = Sequential()
		self.nx1 = None
		self.nx2 = None
		self.nx3 = None
		self.shape1 = None
		self.shape2 = None
		self.ranks = None
		self.W = None
		self.use_regularization = None
		self.regularization_strength = None
		self._required_keys = ['nx1', 'nx2', 'nx3', 'shape1', 'shape2', 'ranks', 'W']

		config.setdefault('use_regularization', False)
		config.setdefault('regularization_strength', 0.0)
		self.load_config(config)
		self.config = config
		self.build_model()

	def load_config(self, config):
		missing_keys = [key for key in self._required_keys if key not in config]
		if missing_keys:
			raise KeyError(f"Missing keys in config: {', '.join(missing_keys)}")

		if not isinstance(config['use_regularization'], bool):
			raise TypeError('use_regularization must be a boolean.')
		else:
			self.use_regularization = config['use_regularization']

		self.regularization_strength = 0.0

		for key in ['nx1', 'nx2', 'nx3', 'W']:
			if not isinstance(config[key], int):
				raise TypeError(f"{key} must be an integer.")

		for key in ['nx1', 'nx2', 'nx3']:
			if config[key] <= 0:
				raise ValueError(f"{key} must be positive.")

		if config['W'] < 0:
			raise ValueError("W must be non-negative.")

		nx1, nx2, nx3 = config['nx1'], config['nx2'], config['nx3']
		self.nx1 = nx1
		self.nx2 = nx2
		self.nx3 = nx3

		W = config['W']
		self.W = W

		input_dim = 2 * nx2 * W
		if W == 0:
			input_dim = nx2 * nx3

		shape1, shape2 = config['shape1'], config['shape2']
		if len(shape1) != len(shape2):
			raise ValueError(
				f'shape1 and shape2 must have the same length. '
				f'Received: shape1 = {shape1}, shape2 = {shape2}.'
			)
		elif np.prod(np.array(shape1)) != nx1 * nx2:
			raise ValueError(
				f'prod(shape1) must be equal to the output dimension of the TTL '
				f'(nx1 * nx2,). Received: prod(shape1) = {np.prod(np.array(shape1))}, '
				f'nx1 * nx2 = {nx1 * nx2}.'
			)
		elif np.prod(np.array(shape2)) != input_dim:
			raise ValueError(
				f'prod(shape2) must be equal to the input dimension of the TTL '
				f'(2 * nx2 * W or nx2 * nx3 if W = 0). '
				f'Received: prod(shape2) = {np.prod(np.array(shape2))}, required input dimension = {input_dim}.'
			)
		else:
			self.shape1 = shape1
			self.shape2 = shape2

		self.ranks = config['ranks']

	def build_model(self):
		if self.W == 0:
			self.model.add(Flatten(input_shape = (2 * self.nx2, self.nx3 // 2)))
		else:
			self.model.add(EinsumTTL(self.nx2, self.nx3, self.W, self.use_regularization,
									 regularization_strength = self.regularization_strength,
									 input_shape = (2 * self.nx2, self.nx3 // 2)))
			self.model.add(Flatten())
		tt_layer = t3f.nn.KerasDense(input_dims = self.shape2, output_dims = self.shape1,
									 tt_rank = self.ranks, use_bias = False, activation = 'linear')
		self.model.add(tt_layer)
		self.model.add(Reshape((self.nx1, self.nx2)))

	def call(self, inputs):
		return self.model(inputs)
