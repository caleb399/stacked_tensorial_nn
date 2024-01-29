import unittest
import numpy as np
import h5py
import tempfile
from stnn.data.preprocessing import get_data_from_file, load_data, load_training_data


class TestGetDataFromFile(unittest.TestCase):

	def setUp(self):
		self.temp_file = tempfile.NamedTemporaryFile(delete = False)
		self.nx1, self.nx2, self.nx3 = 30, 20, 16
		self.Nsamples = 10
		with h5py.File(self.temp_file.name, 'w') as f:
			f.create_dataset('ell', data = np.random.rand(self.Nsamples))
			f.create_dataset('a1', data = np.random.rand(self.Nsamples))
			f.create_dataset('a2', data = np.random.rand(self.Nsamples))
			f.create_dataset('rho', data = np.random.rand(self.Nsamples, self.nx1, self.nx2))
			f.create_dataset('ibf', data = np.random.rand(self.Nsamples, self.nx2, self.nx3 // 2))
			f.create_dataset('obf', data = np.random.rand(self.Nsamples, self.nx2, self.nx3 // 2))

		self.temp_file1 = tempfile.NamedTemporaryFile(delete = False)
		with h5py.File(self.temp_file1.name, 'w') as f:
			f.create_dataset('ell', data = np.random.rand(self.Nsamples))
			f.create_dataset('a1', data = np.random.rand(self.Nsamples))
			f.create_dataset('a2', data = np.random.rand(self.Nsamples))
			f.create_dataset('rho', data = np.random.rand(self.Nsamples, self.nx1, self.nx2))
			f.create_dataset('ibf', data = np.random.rand(self.Nsamples, self.nx2, self.nx3 // 2))
			f.create_dataset('obf', data = np.random.rand(self.Nsamples, self.nx2, self.nx3 // 2))

		self.bad_file = tempfile.NamedTemporaryFile(delete = False)
		with h5py.File(self.bad_file.name, 'w') as f:
			f.create_dataset('ell', data = np.random.rand(self.Nsamples))
			f.create_dataset('a1', data = np.random.rand(self.Nsamples))
			f.create_dataset('a2', data = np.random.rand(self.Nsamples))
			f.create_dataset('rho', data = np.random.rand(self.Nsamples, self.nx1, self.nx2))

	def tearDown(self):
		self.temp_file.close()
		self.temp_file1.close()
		self.bad_file.close()

	def test_missing_datasets(self):
		with self.assertRaises(ValueError):
			get_data_from_file(self.bad_file.name, self.nx2, self.nx2)

	def test_data_extraction_shapes(self):
		result = get_data_from_file(self.temp_file.name, self.nx2, self.nx3)
		self.assertEqual(result[0].shape, (self.Nsamples,))
		self.assertEqual(result[1].shape, (self.Nsamples,))
		self.assertEqual(result[2].shape, (self.Nsamples,))
		self.assertEqual(result[3].shape, (self.Nsamples, 2 * self.nx2, self.nx3 // 2))
		self.assertEqual(result[4].shape, (self.Nsamples, self.nx1, self.nx2))

	def test_nrange_parameter(self):
		Nrange = (2, 5)
		result = get_data_from_file(self.temp_file.name, self.nx2, self.nx3, Nrange = Nrange)
		expected_size = Nrange[1] - Nrange[0]
		self.assertEqual(result[0].shape, (expected_size,))
		self.assertEqual(result[0].shape, (expected_size,))

	def test_list_input(self):
		file_list = [self.temp_file.name, self.temp_file1.name]
		Nrange_list = [(0, -1), (0, -1)]
		with self.assertRaises(TypeError):
			# noinspection PyTypeChecker
			_ = get_data_from_file(file_list, self.nx2, self.nx3, Nrange = Nrange_list)

	def test_invalid_Nrange(self):
		Nrange_list = [(0, -1), (0, -1)]
		with self.assertRaises(TypeError):
			_ = get_data_from_file(self.temp_file.name, self.nx2, self.nx3, Nrange = Nrange_list)

		for Nrange in [(0, 1, 1), 1, (1), (1.5, 3), (3, 1.5), (1.5, 1.5), 'x']:
			with self.assertRaises(TypeError):
				_ = get_data_from_file(self.temp_file.name, self.nx2, self.nx3, Nrange = Nrange)
			with self.assertRaises(TypeError):
				_ = get_data_from_file(self.temp_file.name, self.nx2, self.nx3, Nrange = list(Nrange))

	def test_good_data_load(self):
		files = [self.temp_file.name, self.temp_file1.name]
		Nrange_list = [(0, None), (0, self.Nsamples)]
		ell1, ell2, a1, a2 = 0.1, 2.0, 1.0, 5.0

		params, bf, rho = load_data(files, self.nx2, self.nx3, ell1, ell2, a1, a2, Nrange_list = Nrange_list)
		self.assertEqual(params.shape, (2 * self.Nsamples, 3))
		self.assertEqual(bf.shape, (2 * self.Nsamples, 2 * self.nx2, self.nx3 // 2))
		self.assertEqual(rho.shape, (2 * self.Nsamples, self.nx1, self.nx2))

		test_size = 0.3
		(params_train, bf_train, rho_train,
		 params_test, bf_test, rho_test) = load_training_data(files, self.nx2, self.nx3,
															  ell1, ell2, a1, a2, test_size = test_size,
															  Nrange_list = Nrange_list)
		Ntest = int(test_size * 2 * self.Nsamples)
		Ntrain = 2 * self.Nsamples - Ntest
		self.assertEqual(params_train.shape, (Ntrain, 3))
		self.assertEqual(bf_train.shape, (Ntrain, 2 * self.nx2, self.nx3 // 2))
		self.assertEqual(rho_train.shape, (Ntrain, self.nx1, self.nx2))
		self.assertEqual(params_test.shape, (Ntest, 3))
		self.assertEqual(bf_test.shape, (Ntest, 2 * self.nx2, self.nx3 // 2))
		self.assertEqual(rho_test.shape, (Ntest, self.nx1, self.nx2))

	def test_bad_data_load(self):
		files = [self.temp_file.name, self.temp_file1.name]
		Nrange_list = (0, -1)
		ell1, ell2, a1, a2 = 0.1, 2.0, 1.0, 5.0
		with self.assertRaises(TypeError):
			_ = load_data(files, self.nx2, self.nx3, ell1, ell2, a1, a2, Nrange_list = Nrange_list)
		with self.assertRaises(TypeError):
			_ = load_data(files, self.nx2, self.nx3, ell1, ell2, a1, a2, Nrange_list = list(Nrange_list))

		Nrange_list = [(0, -1), (0, -1)]
		for test_size in [-1, 0.0, 1.5]:
			with self.assertRaises(ValueError):
				_ = load_training_data(files, self.nx2, self.nx3,
									   ell1, ell2, a1, a2, test_size = test_size, Nrange_list = Nrange_list)


if __name__ == '__main__':
	unittest.main()
