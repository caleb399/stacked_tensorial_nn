import unittest
import numpy as np
from stnn.data.preprocessing import train_test_split


class TestTrainTestSplit(unittest.TestCase):

	def setUp(self):
		self.X_array = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
		self.Y_array = np.array([1, 2, 3, 4])
		self.X_list = [self.X_array, self.X_array]
		self.Y_list = [self.Y_array, self.Y_array]
		self.X_list_bad = [self.Y_array, self.X_array]
		self.Y_list_bad = [self.Y_array, self.X_array]

	def test_basic_functionality_array(self):
		X_train, X_test, Y_train, Y_test = train_test_split(self.X_array, self.Y_array, test_size = 0.25)
		self.assertEqual(len(X_train), 3)
		self.assertEqual(len(X_test), 1)
		self.assertEqual(len(Y_train), 3)
		self.assertEqual(len(Y_test), 1)

	def test_basic_functionality_list(self):
		X_train, X_test, Y_train, Y_test = train_test_split(self.X_list, self.Y_list, test_size = 0.25)
		self.assertEqual(len(X_train[0]), 3)
		self.assertEqual(len(X_test[0]), 1)
		self.assertEqual(len(Y_train[0]), 3)
		self.assertEqual(len(Y_test[0]), 1)

	def test_return_type_consistency_array(self):
		X_train, X_test, Y_train, Y_test = train_test_split(self.X_array, self.Y_array, test_size = 0.25)
		self.assertIsInstance(X_train, np.ndarray)
		self.assertIsInstance(X_test, np.ndarray)
		self.assertIsInstance(Y_train, np.ndarray)
		self.assertIsInstance(Y_test, np.ndarray)

		X_train, X_test, Y_train, Y_test = train_test_split([self.X_array], [self.Y_array], test_size = 0.25)
		self.assertIsInstance(X_train, list)
		self.assertIsInstance(X_test, list)
		self.assertIsInstance(Y_train, list)
		self.assertIsInstance(Y_test, list)

	def test_return_type_consistency_list(self):
		X_train, X_test, Y_train, Y_test = train_test_split(self.X_list, self.Y_list, test_size = 0.25)
		self.assertIsInstance(X_train, list)
		self.assertIsInstance(X_test, list)
		self.assertIsInstance(Y_train, list)
		self.assertIsInstance(Y_test, list)

		# noinspection PyTypeChecker
		X_train, X_test, Y_train, Y_test = train_test_split(tuple(self.X_list), tuple(self.Y_list), test_size = 0.25)
		self.assertIsInstance(X_train, list)
		self.assertIsInstance(X_test, list)
		self.assertIsInstance(Y_train, list)
		self.assertIsInstance(Y_test, list)

	def test_random_state(self):
		X_train1, X_test1, Y_train1, Y_test1 = train_test_split(self.X_array, self.Y_array, test_size = 0.25,
																random_state = 42)
		X_train2, X_test2, Y_train2, Y_test2 = train_test_split(self.X_array, self.Y_array, test_size = 0.25,
																random_state = 42)
		np.testing.assert_array_equal(X_train1, X_train2)
		np.testing.assert_array_equal(X_test1, X_test2)
		np.testing.assert_array_equal(Y_train1, Y_train2)
		np.testing.assert_array_equal(Y_test1, Y_test2)

	def test_invalid_test_size(self):
		with self.assertRaises(ValueError):
			train_test_split(self.X_array, self.Y_array, test_size = -0.1)
		with self.assertRaises(ValueError):
			train_test_split(self.X_array, self.Y_array, test_size = 1.5)

	def test_inconsistent_length(self):
		X = np.array([[1, 2], [3, 4]])
		Y = np.array([1, 2, 3])
		with self.assertRaises(ValueError):
			train_test_split(X, Y)
		with self.assertRaises(ValueError):
			train_test_split(self.X_list_bad, self.Y_list_bad)
		with self.assertRaises(ValueError):
			train_test_split(self.X_list_bad, self.Y_list)
		with self.assertRaises(ValueError):
			train_test_split(self.X_list, self.Y_list_bad)

	def test_empty(self):
		X_empty = np.zeros(0)
		Y_empty = np.zeros(0)
		with self.assertRaises(ValueError):
			train_test_split(X_empty, Y_empty)
		with self.assertRaises(ValueError):
			train_test_split([X_empty], [])
		with self.assertRaises(ValueError):
			train_test_split([], [Y_empty])
		with self.assertRaises(ValueError):
			train_test_split([X_empty], [Y_empty])


if __name__ == '__main__':
	unittest.main()
