import unittest
import numpy as np
import copy
from stnn.nn.stnn import build_stnn


class TestBuildSTNN(unittest.TestCase):

	def setUp(self):
		self.config = {
			'K': 1,
			'nx1': 8,
			'nx2': 8,
			'nx3': 8,
			'd': 8,
			'W': 3,
			'shape1': [1, 2, 3],
			'shape2': [2, 2, 2],
			'ranks': [1, 2, 2, 1],
		}
		self.saved_config = copy.deepcopy(self.config)
		self._required_keys = ['nx1', 'nx2', 'nx3', 'K', 'd', 'shape1','shape2','ranks','W']
		self._optional_keys = ['use_regularization', 'regularization_strength']

	def test_missing_keys(self):
		for key in self._required_keys:
			del self.config[key]
			with self.assertRaises(KeyError):
				build_stnn(self.config)
			self.config[key] = self.saved_config[key]

	def test_invalid_values(self):
		for key in ['K', 'd', 'W', 'nx1', 'nx2', 'nx3']:
			for value in [1.5, 'a', None, np.nan]:
				with self.subTest(value = value):
					self.config[key] = value
					with self.assertRaises(TypeError):
						build_stnn(self.config)
					self.config[key] = self.saved_config[key]
			value = -1
			with self.subTest(value = value):
				self.config[key] = value
				with self.assertRaises(ValueError):
					build_stnn(self.config)
				self.config[key] = self.saved_config[key]

		self.config['nx3'] = 7  # not divisible by 2
		with self.assertRaises(ValueError):
			build_stnn(self.config)
		self.config[key] = self.saved_config[key]

	def test_positive_values(self):
		for key in ['K', 'nx1', 'nx2', 'nx3', 'd']:
			self.config[key] = 0
			with self.assertRaises(ValueError):
				build_stnn(self.config)
			self.config[key] = self.saved_config[key]


if __name__ == '__main__':
	unittest.main()
