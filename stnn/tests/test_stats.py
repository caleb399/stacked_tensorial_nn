import numpy as np
import unittest
import os
from stnn.utils.stats import get_stats


class TestGetStats(unittest.TestCase):

	def setUp(self):
		self.rho = np.array([[1, 2], [3, 4]])
		self.rho_pred = np.array([[1, 2], [3, 4]])
		self.filename = 'test_stats.npz'

	def test_correctness(self):
		get_stats(self.rho, self.rho_pred, self.filename)
		with np.load(self.filename) as data:
			self.assertAlmostEqual(data['max_loss'], 0.0, places=5)
			self.assertEqual(data['avg_loss'], 0.0)
			self.assertEqual(data['N'], self.rho.shape[0])

	def test_file_creation(self):
		get_stats(self.rho, self.rho_pred, self.filename)
		self.assertTrue(os.path.exists(self.filename))

	def test_file_content(self):
		get_stats(self.rho, self.rho_pred, self.filename)
		with np.load(self.filename) as data:
			self.assertIn('max_loss', data)
			self.assertIn('avg_loss', data)
			self.assertIn('N', data)

	def test_invalid_input(self):
		with self.assertRaises(ValueError):
			get_stats(np.array([1, 2]), np.array([[1, 2], [3, 4]]))

	def tearDown(self):
		if os.path.exists(self.filename):
			os.remove(self.filename)

if __name__ == '__main__':
	unittest.main()
