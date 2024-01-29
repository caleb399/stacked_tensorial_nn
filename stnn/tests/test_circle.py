import unittest
import copy
import numpy as np
import scipy.sparse as sp
from stnn.pde.circle import get_system_circle


class TestGetSystemCircle(unittest.TestCase):

	def setUp(self):
		self.config = {
			'nx1': 10,
			'nx2': 20,
			'nx3': 30,
			'a2': 2.0,
			'ell': 1.5,
		}
		self.saved_config = copy.deepcopy(self.config)
		self._required_keys = ['nx1', 'nx2', 'nx3', 'ell', 'a2']
		self._optional_keys = []

	def test_valid_output(self):
		L, r_3D, theta_3D, w_3D, dr1, dr2, Dr_3D_coeff_meshgrid = get_system_circle(self.config)

		# Test types
		self.assertIsInstance(L, sp.csr_matrix)
		self.assertIsInstance(r_3D, np.ndarray)
		self.assertIsInstance(theta_3D, np.ndarray)
		self.assertIsInstance(w_3D, np.ndarray)
		self.assertIsInstance(Dr_3D_coeff_meshgrid, np.ndarray)

		# Test shapes
		self.assertEqual(r_3D.shape, (self.config['nx1'], self.config['nx2'], self.config['nx3']))
		self.assertEqual(theta_3D.shape, (self.config['nx1'], self.config['nx2'], self.config['nx3']))
		self.assertEqual(w_3D.shape, (self.config['nx1'], self.config['nx2'], self.config['nx3']))

		# Test values
		self.assertTrue(dr1 > 0)
		self.assertTrue(dr2 > 0)

	def test_missing_keys(self):
		for key in self._required_keys:
			del self.config[key]
			with self.assertRaises(KeyError):
				get_system_circle(self.config)
			self.config[key] = self.saved_config[key]

	def test_invalid_parameters(self):
		for key in self._required_keys:
			self.config[key] = 0  # None of the required keys should be zero.
			with self.assertRaises(ValueError):
				get_system_circle(self.config)
			self.config[key] = self.saved_config[key]

	def test_unused_params_warning(self):
		copied_params = copy.deepcopy(self.config)
		copied_params['unusedkey'] = 0
		with self.assertWarns(UserWarning) as _:
			get_system_circle(copied_params)


if __name__ == '__main__':
	unittest.main()
