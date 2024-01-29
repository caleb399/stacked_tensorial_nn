import h5py
from scipy.sparse import csr_matrix
import numpy as np
import json
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2


def save_to_hdf5(filename, datasets, start_idx, end_idx):
	"""
	Saves subsets of datasets to an HDF5 file, either by creating new datasets or appending to existing ones.

	Args:
		filename (str): The name of the HDF5 file where the data will be saved.
		datasets (dict): A dictionary where keys are dataset names and values are the corresponding data arrays.
		start_idx (int): The starting index of the data slice to be saved.
		end_idx (int): The ending index (exclusive) of the data slice to be saved.

	This function will create new datasets if they do not already exist. If a dataset already exists, it will be resized
	to accommodate the new data, and the data slice will be appended.
	"""
	print(f'Saving data from n = {start_idx} to n = {end_idx}...')
	with h5py.File(filename, 'a') as f:
		for name, data in datasets.items():
			if name not in f:
				f.create_dataset(name, data = data[start_idx:end_idx], maxshape = (None,) + data.shape[1:],
								 chunks = (1,) + data.shape[1:])
			else:
				f[name].resize((f[name].shape[0] + end_idx - start_idx,) + f[name].shape[1:])
				f[name][-(end_idx - start_idx):] = data[start_idx:end_idx]
	print('Done.')


def write_sparse_matrix_hdf5(filename, sparse_matrix, dataset_name = 'sparse_matrix', format_ = 'csr'):
	"""
	Writes a sparse matrix to an HDF5 file in a specified format.

	Args:
		filename (str): Name of the output file
		sparse_matrix (scipy.sparse matrix): Sparse matrix to be written to the file.
		dataset_name (str, optional): Name of the HDF5 dataset. Defaults to 'sparse_matrix'.
		format_ (str, optional): The format of the sparse matrix. Currently only supports 'csr' (Compressed Sparse Row). 
	"""
	if format_ == 'csr':
		with h5py.File(filename, 'w') as f:
			g = f.create_group(dataset_name)
			g.create_dataset('data', data = sparse_matrix.data)
			g.create_dataset('indices', data = sparse_matrix.indices)
			g.create_dataset('indptr', data = sparse_matrix.indptr)
			g.create_dataset('shape', data = np.array(sparse_matrix.shape))
			g.attrs['format'] = 'csr'
	else:
		raise ValueError(f'Unsupported sparse matrix format: {format_}')


def read_sparse_matrix_hdf5(filename, dataset_name = 'sparse_matrix'):
	"""
	Reads a sparse matrix from an HDF5 file.

	Args:
		filename (str): Name of the output file
		dataset_name (str, optional): Name of the dataset containing the matrix. Defaults to 'sparse_matrix'.

	Returns:
		scipy.sparse matrix: The sparse matrix read from the file.
	"""
	with h5py.File(filename, 'r') as f:
		g = f[dataset_name]
		data = g['data'][:]
		indices = g['indices'][:]
		indptr = g['indptr'][:]
		shape = tuple(g['shape'][:])
		format_ = g.attrs['format']
		if format_ == 'csr':
			return csr_matrix((data, indices, indptr), shape = shape)
		else:
			raise ValueError(f'Unsupported sparse matrix format: {format_}')


def data_dump(bf, rho, rho_pred, params_dict):
	"""
	Writes data and config to file for later use.
	"""
	np.savez('sample_data.npz', rho = rho, rho_pred = rho_pred, bf = bf)
	json.dump('sample_config.json', params_dict, encoding = 'utf-8')


def save_as_frozen_graph(model, saved_model_dir):
	"""
	Converts a TensorFlow model into a 'frozen' SavedModel format.

	Args:
		model: TensorFlow model to be frozen.
		saved_model_dir (str): The directory path where the frozen model will be saved.

	Returns:
		TFModelServer object: An instance of the TFModelServer class, which can be used for serving the model.

	The function converts model variables to constants and is required for converting the model to the intermediate
	 representation (IR) used by openvino.
	"""
	# Create input specifications for the model
	input_specs = [tf.TensorSpec([1] + model.inputs[i].shape[1:].as_list(), model.inputs[i].dtype) for i in
				   range(len(model.inputs))]

	# Create a concrete function from the model
	full_model = tf.function(lambda x: model(x))
	full_model = full_model.get_concrete_function(input_specs)

	# Convert the model to a frozen function
	frozen_func = convert_variables_to_constants_v2(full_model)

	# Define a new module with a method that has a `@tf.function` decorator with input signatures
	class TFModelServer(tf.Module):
		def __init__(self, frozen_func):
			super().__init__()
			self.frozen_func = frozen_func

		@tf.function(input_signature = input_specs)
		def serve(self, *args):
			return self.frozen_func(*args)

	# Create an instance of TFModelServer with the frozen function
	model_server = TFModelServer(frozen_func)

	# Save the module as a SavedModel
	tf.saved_model.save(model_server, saved_model_dir, signatures = {"serving_default": model_server.serve})

	return model_server


def log_and_print(message):
	print(message)
	with open('log.txt', 'a') as log_file:
		log_file.write(message + '\n')
