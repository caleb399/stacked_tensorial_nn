import unittest
import copy
import numpy as np
import scipy.sparse as sp
from stnn.pde.ellipse import get_system_ellipse


class TestGetSystemCircle(unittest.TestCase):

	def setUp(self):
		self.config = {
			'nx1': 10,
			'nx2': 20,
			'nx3': 30,
			'a2': 2.0,
			'ell': 1.5,
			'eccentricity': 0.5
		}
		self.saved_config = copy.deepcopy(self.config)
		self._required_keys = ['nx1', 'nx2', 'nx3', 'ell', 'a2', 'eccentricity']
		self._optional_keys = []

	def test_valid_output(self):
		L, mu_3D, eta_3D, w_3D, dmu1, dmu2, Dmu_3D_coeff_meshgrid, major_axis_outer = get_system_ellipse(self.config)

		# Test types
		self.assertIsInstance(L, sp.csr_matrix)
		self.assertIsInstance(mu_3D, np.ndarray)
		self.assertIsInstance(eta_3D, np.ndarray)
		self.assertIsInstance(w_3D, np.ndarray)
		self.assertIsInstance(Dmu_3D_coeff_meshgrid, np.ndarray)

		# Test shapes
		self.assertEqual(mu_3D.shape, (self.config['nx1'], self.config['nx2'], self.config['nx3']))
		self.assertEqual(eta_3D.shape, (self.config['nx1'], self.config['nx2'], self.config['nx3']))
		self.assertEqual(w_3D.shape, (self.config['nx1'], self.config['nx2'], self.config['nx3']))
		N = self.config['nx1'] * self.config['nx2'] * self.config['nx3']
		self.assertEqual(L.shape, (N, N))

		# Test values
		self.assertTrue(dmu1 > 0)
		self.assertTrue(dmu2 > 0)
		self.assertTrue(major_axis_outer > self.config['a2'])

	def test_missing_keys(self):
		for key in self._required_keys:
			del self.config[key]
			with self.assertRaises(KeyError):
				get_system_ellipse(self.config)
			self.config[key] = self.saved_config[key]

	def test_invalid_parameters(self):
		for key in self._required_keys:
			self.config[key] = -1  # None of the required keys should be negative.
			with self.assertRaises(ValueError):
				get_system_ellipse(self.config)
			self.config[key] = self.saved_config[key]

	def test_unused_params_warning(self):
		copied_params = copy.deepcopy(self.config)
		copied_params['unusedkey'] = 0
		with self.assertWarns(UserWarning) as _:
			get_system_ellipse(copied_params)


if __name__ == '__main__':
	unittest.main()
