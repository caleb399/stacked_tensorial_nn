import timeit
import numpy as np
import argparse
import json
import tensorflow as tf

from stnn.nn.stnn import build_stnn
from stnn.data.preprocessing import load_data

"""
Performs quantization of a STNN model using TensorFlow Lite. To run the script, make sure you have
a configuration file for the STNN and a weights file. The directory 'model_trial_5' contains examples
corresponding to Trial 5 from (Wagner, 2023).

The script also requires 'reference_dataset.h5' and 'test_dataset.h5' to be placed within the same
directory as the script. 'reference_dataset.h5' is used during quantization, while 'test_dataset.h5'
is used after inference to test the accuracy.

Note that the built-in quantization that is performed does not seem to be effective for the STNN 
model (e.g. ~2x speedup but 5x loss in accuracy). Due to the fact that the STNN seeks to approximate 
the numerical solution of a partial differential equation, full data precision (i.e., float32) may 
be required for inference.
"""


def main():
	# Get weights file and config file from command line arguments
	parser = argparse.ArgumentParser(description = 'Specify the weights file and config file')
	parser.add_argument('--weights_file', '-wf', type = str, required = True,
						help = 'The file containing the STNN weights in HFD5 format.')
	parser.add_argument('--config_file', '-cf', type = str, required = True,
						help = 'The file containing the STNN configuration in JSON format.')
	args = parser.parse_args()
	weights_file = args.weights_file
	stnn_config_file = args.config_file

	# Datasets; modify as needed
	hdf5_reference_dataset = 'reference_dataset.h5'
	hdf5_test_data = 'test_dataset.h5'

	# STNN model config is loaded from the provided json file.
	with open(stnn_config_file, 'r', encoding = 'utf-8') as json_file:
		stnn_config = json.load(json_file)

	# Build model from weights and config file
	model = build_stnn(stnn_config)
	model.load_weights(weights_file)
	model.compile(optimizer = 'adam', loss = 'mean_squared_error')

	# Convert to tflite model and save to file
	tflite_model = do_quantization(model, stnn_config, hdf5_reference_dataset)
	tflite_fname = 'model_quantized.tflite'
	with open(tflite_fname, 'wb') as f:
		f.write(tflite_model)
	do_inference(tflite_fname, weights_file, stnn_config, hdf5_test_data)


def do_quantization(model, stnn_config, hdf5_reference_dataset):
	"""
	Perform quantization on the given STNN tensorflow model using the TensorFlow Lite converter.

	Args:
		model (tf.keras.Model): A TensorFlow Keras model that is to be quantized.
		stnn_config (dict): A dictionary containing configuration parameters for the STNN,
							e.g. loaded from the model's config.json file.
		hdf5_reference_dataset (str): HDF5 file containing reference data to use for quantization.


	The function loads a representative dataset from a file named 'quantization_reference_dataset.h5'.

	Returns:
		tf.lite.Model: A TensorFlow Lite model that has been quantized.
	"""
	input_specs = [tf.TensorSpec([1] + model.inputs[i].shape[1:].as_list(), model.inputs[i].dtype) for i in
				   range(len(model.inputs))]
	func = tf.function(model).get_concrete_function(input_specs)
	converter = tf.lite.TFLiteConverter.from_concrete_functions([func])
	converter.optimizations = [tf.lite.Optimize.DEFAULT]

	# Representative dataset for performing quantization.
	ell_min, ell_max = stnn_config['ell_min'], stnn_config['ell_max']
	a2_min, a2_max = stnn_config['a2_min'], stnn_config['a2_max']
	nx2, nx3 = stnn_config['nx2'], stnn_config['nx3']
	params, bf, _ = load_data(hdf5_reference_dataset, nx2, nx3, ell_min, ell_max, a2_min, a2_max)
	input1_dataset = params.astype(np.float32)
	input2_dataset = bf.astype(np.float32)

	nsamples = params.shape[0]

	def representative_data_gen():
		for i in range(nsamples):
			input1_data = input1_dataset[i:i + 1, ...]
			input2_data = input2_dataset[i:i + 1, ...]
			yield [input1_data, input2_data]

	converter.representative_dataset = representative_data_gen
	converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
	converter.inference_input_type = tf.uint8
	converter.inference_output_type = tf.uint8
	tflite_model = converter.convert()

	return tflite_model


def do_inference(tflite_fname, weights_file, stnn_config, hdf5_test_data):
	"""
	Test the quantized model on some sample labeled data. Also, compare speed and accuracy with the
	full (float32) model on the same data.
	Args:
		tflite_fname (str): filename of the quantized model
		weights_file (str): filename of the weights file of the full model
		stnn_config (dict): A dictionary containing configuration parameters for the STNN,
							e.g. loaded from the model's config.json file.
		hdf5_test_data (str): filename of an HDF5 file containing test inputs for the inference
	"""
	# Load the TFLite model in TFLite Interpreter
	interpreter = tf.lite.Interpreter(model_path = tflite_fname)
	interpreter.allocate_tensors()

	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()

	ell_min, ell_max = stnn_config['ell_min'], stnn_config['ell_max']
	a2_min, a2_max = stnn_config['a2_min'], stnn_config['a2_max']
	nx2, nx3 = stnn_config['nx2'], stnn_config['nx3']
	params, bf, rho = load_data(hdf5_test_data, nx2, nx3, ell_min, ell_max, a2_min, a2_max)

	start = timeit.default_timer()
	nsamples = params.shape[0]
	qerrors = np.zeros(nsamples)
	for i in range(nsamples):
		# Quantize the input data
		input_scale, input_zero_point = input_details[0]['quantization']
		input_data_quantized = (params[i:i + 1, ...] / input_scale + input_zero_point).astype(np.uint8)
		input_scale, input_zero_point = input_details[1]['quantization']
		bf_quantized = (bf[i:i + 1, ..., np.newaxis] / input_scale + input_zero_point).astype(np.uint8)

		# Set the model input to the quantized data
		interpreter.set_tensor(input_details[0]['index'], input_data_quantized)
		interpreter.set_tensor(input_details[1]['index'], bf_quantized)
		interpreter.invoke()
		output_scale, output_zero_point = output_details[0]['quantization']
		rho_stnn_q = interpreter.get_tensor(output_details[0]['index'])
		rho_stnn_q = (rho_stnn_q - output_zero_point) * output_scale
		qerrors[i] = np.linalg.norm(rho_stnn_q - rho[i:i + 1, ...]) / np.linalg.norm(rho[i:i + 1, ...])

	end = timeit.default_timer()
	print(f'\nAverage time of quantized (uint8) model: {(end - start) / nsamples} seconds')
	print(f'Average relative error of quantized model: {np.mean(qerrors)}')

	full_model = build_stnn(stnn_config)
	full_model.load_weights(weights_file)
	full_model.compile(optimizer = 'adam', loss = 'mean_squared_error')
	start = timeit.default_timer()
	nsamples = 10
	ferrors = np.zeros(nsamples)
	for i in range(nsamples):
		rho_stnn = full_model.predict([params[i:i + 1, ...], bf[i:i + 1, ...]])[0, ...]
		ferrors[i] = np.linalg.norm(rho_stnn - rho[i:i + 1, ...]) / np.linalg.norm(rho[i:i + 1, ...])

	end = timeit.default_timer()
	print(f'\nAverage time of full (float32) model: {(end - start) / nsamples} seconds')
	print(f'Average relative error of full model: {np.mean(ferrors)}')


if __name__ == '__main__':
	main()
