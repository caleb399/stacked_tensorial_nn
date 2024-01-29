import unittest
from stnn.nn.stnn_layers import TTL


class TestTTL(unittest.TestCase):

	def test_missing_config_keys(self):
		with self.assertRaises(KeyError):
			TTL(config = {})  # Empty config

	def test_invalid_use_regularization_type(self):
		config = {'use_regularization': 'not a boolean', 'nx1': 1, 'nx2': 1, 'nx3': 1, 'shape1': [1], 'shape2': [1],
				  'ranks': [1], 'W': 1}
		with self.assertRaises(TypeError):
			TTL(config = config)

	def test_invalid_nx_values(self):
		config = {'nx1': 5, 'nx2' : 5, 'nx3': 8, 'use_regularization': False, 'shape1': [1], 'shape2': [1], 'ranks': [1], 'W': 1}
		for nx in ['nx1', 'nx2', 'nx3']:
			for value in [-1, 0]:
				config.update({nx: value})
				with self.assertRaises(ValueError):
					TTL(config = config)

	def test_invalid_W_value(self):
		config = {'use_regularization': False, 'nx1': 1, 'nx2': 1, 'nx3': 1, 'shape1': [1], 'shape2': [1], 'ranks': [1],
				  'W': -1}
		with self.assertRaises(ValueError):
			TTL(config = config)

	def test_shape_length_mismatch(self):
		config = {'use_regularization': False, 'nx1': 1, 'nx2': 1, 'nx3': 1, 'shape1': [1, 2], 'shape2': [1],
				  'ranks': [1], 'W': 1}
		with self.assertRaises(ValueError):
			TTL(config = config)

	def test_incorrect_shape1_product(self):
		config = {'use_regularization': False, 'nx1': 2, 'nx2': 2, 'nx3': 1, 'shape1': [1, 3], 'shape2': [4],
				  'ranks': [1], 'W': 1}
		with self.assertRaises(ValueError):
			TTL(config = config)

	def test_incorrect_shape2_product(self):
		config = {'use_regularization': False, 'nx1': 1, 'nx2': 2, 'nx3': 1, 'shape1': [2], 'shape2': [1, 5],
				  'ranks': [1], 'W': 1}
		with self.assertRaises(ValueError):
			TTL(config = config)


if __name__ == '__main__':
	unittest.main()
