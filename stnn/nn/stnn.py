import tensorflow as tf
from tensorflow.keras.layers import Input, Multiply, Add
from tensorflow.keras.models import Model

from .stnn_layers import TTL, SoftmaxEmbeddingLayer


def build_stnn(config):
	"""
	Constructs a Stacked Tensorial Neural Network (STNN) as a TensorFlow model based on
	the provided configuration dictionary.

	Args:
		config (dict): Configuration dictionary for the STNN model. Must contain the following entries:
						- 'K' (int): Number of tensor networks to be stacked
						- 'd' (int): Number of dense layers in the model's SoftmaxEmbeddingLayer.
						- 'nx1', 'nx2', 'nx3' (int): Dimensions of the finite-difference grid
						- All other required entries for the TTL class, not already listed above.

	Returns:
		tf.keras.Model: The constructed STNN model.

	Raises:
		ValueError: If the config dictionary does not contain positive integers 'K', 'd', 'nx2', 'nx3';
					also if config['nx3'] is not divisible by 2.
	"""
	required_keys = ['nx1', 'nx2', 'nx3', 'K', 'd', 'shape1','shape2','ranks','W']
	missing_keys = [key for key in required_keys if key not in config]
	if missing_keys:
		raise KeyError(f"Missing keys in config: {', '.join(missing_keys)}")

	for key in ['nx1', 'nx2', 'nx3', 'K', 'd']:
		if not isinstance(config[key], int):
			raise TypeError(f"{key} must be an integer.")

	for key in ['nx1', 'nx2', 'nx3', 'K', 'd']:
		if config[key] <= 0:
			raise ValueError(f"{key} must be positive.")

	if config['nx3'] % 2 == 1:
		raise ValueError('Config error: nx3 must be divisible by 2.')

	K = config['K'] # Number of tensor networks
	d = config['d'] # Number of dense layers in SoftmaxEmbeddingLayer
	input_shape = (2 * config['nx2'], config['nx3'] // 2, 1)
	input_tensor = Input(shape = input_shape)

	# Process parameter array (ell, a1, a2) and output weights for stacking the tensor networks
	preprocess_layer = SoftmaxEmbeddingLayer(K, d)
	params_input = Input(shape = (3,))
	stack_weights = preprocess_layer(params_input)[:, tf.newaxis, tf.newaxis, :]

	# Build the tensor networks using the custom keras layer class TLL
	models = [TTL(config) for _ in range(K)]

	# Combine the tensor networks based on the weights outputted by 'preprocess_layer'
	weighted_outputs = []
	for i, model in enumerate(models):
		processed_output = model(input_tensor)
		weighted_output = Multiply()([processed_output, stack_weights[..., i]])
		weighted_outputs.append(weighted_output)
	final_output = Add()(weighted_outputs)

	model = Model(inputs = [params_input, input_tensor], outputs = final_output)

	return model
