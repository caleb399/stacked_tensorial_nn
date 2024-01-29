import copy
import unittest

import numpy as np

from stnn.pde.pde_system import PDESystem


class TestPDESystem(unittest.TestCase):

	def setUp(self):
		self.params = {
			'nx1': 50,
			'nx2': 100,
			'nx3': 75,
			'a1': 1.0,
			'a2': 2.0,
			'ell': 0.1,
			'eccentricity': 0.5
		}
		self.system = PDESystem(self.params)

	def test_initialization(self):
		self.assertEqual(self.system.params, self.params)

	def test_attribute_types(self):
		self.assertIsInstance(self.system.ib_slice, np.ndarray)
		self.assertIsInstance(self.system.ob_slice, np.ndarray)

	def test_attribute_values(self):
		self.assertEqual(self.system.a1, 1.0 - self.params['eccentricity'])

	def test_coordinate_system(self):
		expected_coords = 'ellipse' if self.params['eccentricity'] != 0 else 'circle'
		self.assertEqual(self.system._coords, expected_coords)

	def test_unused_params_warning(self):
		copied_params = copy.deepcopy(self.params)
		copied_params['unusedkey'] = 0
		with self.assertWarns(UserWarning) as _:
			_ = PDESystem(copied_params)

	def test_L(self):
		# If f(x1, x2, x3) is constant, then L * f should be 0 except adjacent to the boundary.
		L = self.system.L
		nx1, nx2, nx3 = self.params['nx1'], self.params['nx2'], self.params['nx3']
		f0 = 2.3 * np.ones((nx1, nx2, nx3))
		result = L @ f0.ravel()
		result = result.reshape((nx1, nx2, nx3))
		np.testing.assert_allclose(result[1:-1, ...], 0, atol = 1e-7)


if __name__ == '__main__':
	unittest.main()
