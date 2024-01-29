import sys
import timeit
import warnings
import json

from stnn.linalg_backend import xp, spx, asarray, asnumpy, csr_matrix
from stnn.data.test_functions import *
from stnn.pde.pde_system import PDESystem
from stnn.nn.stnn import build_stnn
from stnn.utils.plotting import plot_comparison

"""
Performs inference using either TensorFlow OpenVINO intermediate representation (IR) formats.
This script is mainly for benchmarking purposes. While there is an option to directly compute
the solution rho(x,y) for the generated inputs (parameters and boundary conditions), this is
disabled by default due to the long runtime on systems without a GPU. To turn it on, set
compare_direct to True at the beginning of main().

On an Intel Core Ultra 155H, inference using TensorFlow takes about 4.7 seconds. Using the
openVINO intermediate representation format (which is represented as a static graph to
reduce overhead) takes about 0.7 seconds.

Example usages (on Windows):

	python inference.py tf model_trial_5\T5_config.json model_trial_5\T5_weights.h5

	python inference.py ov model_trial_5\T5_config.json model_trial_5\T5.bin model_trial_5\T5.xml
"""


def main():
	
	# Option to compute the solution directly using GMRES, and compare with the network prediction
	compare_direct = False

	# PDE parameters
	ell = 1.0
	eccentricity = 0.0
	a2 = 10.0

	# "Function generator" id. Can be any integer from -1 to 4. See data/function_generators.py for details.
	func_gen_id = 2
	
	if len(sys.argv) < 4:
		print('Usage: python inference.py arg1 arg2 arg3 [arg4] where:\n'
						'        arg1: model representation (ov|tf)\n'
						'        arg2: stnn config file (*.json)\n'
						'        arg3: weights file (*.h5 for tensorflow (tf) | *.bin for openvino)\n'
						'        arg4: [openvino only] IR config file (*.xml)\n')
		sys.exit(1)
	if sys.argv[1] == 'ov':
		try:
			from openvino.inference_engine import IECore
		except ImportError:
			print('Please install OpenVINO to use the OV inference engine.')
			sys.exit(2)
	if len(sys.argv) == 4:
		model_format = sys.argv[1]
		if model_format == 'ov':
			raise ValueError("Command line argument 'ir_config_file' must be provided "
							 "when the model format is 'ov'.")
		config_file = sys.argv[2]
		weights_file = sys.argv[3]
		ir_config_file = None
	else:
		model_format = sys.argv[1]
		config_file = sys.argv[2]
		weights_file = sys.argv[3]
		ir_config_file = sys.argv[4]

	if not (model_format == 'ov' or model_format == 'tf'):
		raise ValueError(f"Command line argument 'model_format' must be either 'ov' or 'tf'."
						 f"Instead, received: {model_format}")

	# STNN model config is loaded from the provided json file.
	try:
		with open(config_file, 'r', encoding = 'utf-8') as json_file:
			stnn_config = json.load(json_file)
	except Exception as e:
		print(f'\nError loading network configuration from {config_file}: {e}\n'
				'Check that the file exists and is properly formatted JSON.\n')
		sys.exit(1)

	# Initialize model and PDE system; generate boundary conditions to use for inference.
	system, params, bf, b = setup_inference(stnn_config, ell, eccentricity, a2, func_gen_id = func_gen_id)

	# Do the inference
	rho_stnn, stnn_time = do_inference(model_format, stnn_config, weights_file, params, bf,
									   ir_config_file = ir_config_file)
	print(f'Done with inference. Time: {stnn_time} seconds.')

	# Run direct solver for comparison
	if compare_direct:
		rho_direct, direct_time = direct_solution(system, b)

		# Compare the results
		print(f'Speedup: {direct_time / stnn_time}')
		print(f'Relative error: {np.linalg.norm(rho_stnn - rho_direct) / np.linalg.norm(rho_direct)}')
		plot_comparison(system, bf[0, ...], rho_direct, rho_stnn[0, ...],
						fontscale = 1, output_filename = 'comparison.png')


def do_inference(model_format, stnn_config, weights_file, params, bf, ir_config_file = None):
	if model_format == 'ov' and ir_config_file is None:
		raise ValueError("'ir_config_file' must be provided when the model format is 'ov'.")

	if model_format == 'tf':
		start = timeit.default_timer()
		model = build_stnn(stnn_config)
		try:
			model.load_weights(weights_file)
		except OSError as e:
			print(f'\nError loading tensorflow model weights from {weights_file}: {e}\n'
					'Check that the file is a valid HDF5 file.\n')
			sys.exit(1)
		except Exception as e:
			print(f'\nError loading tensorflow model weights from {weights_file}: {e}\n')
			sys.exit(1)
		model.compile(optimizer = 'adam', loss = 'mean_squared_error')

		rho_stnn = model.predict([params, bf])
		stnn_time = timeit.default_timer() - start
	elif model_format == 'ov':
		from openvino.inference_engine import IECore

		start = timeit.default_timer()
		model_xml = ir_config_file
		model_bin = weights_file

		# Initialize the Inference Engine
		ie = IECore()
		try:
			net = ie.read_network(model = model_xml, weights = model_bin)
		except Exception as e:
			print(f'\nError reading network from the provided IR model files {model_xml} and {model_bin}.\nOriginal error: {e}.')
			sys.exit(1)
		exec_net = ie.load_network(network = net, device_name = "CPU")

		# Get the names of the input layers
		input_layers = list(net.input_info.keys())
		input_dict = {input_layers[0]: params, input_layers[1]: bf[..., np.newaxis]}

		output_dict = exec_net.infer(inputs = input_dict)
		output_blob = next(iter(net.outputs))
		rho_stnn = output_dict[output_blob]

		stnn_time = timeit.default_timer() - start
	else:
		raise ValueError(f'Unrecognized model_format {model_format}. Should be either "tf" or "ov".')

	return rho_stnn, stnn_time


def setup_inference(stnn_config, ell, eccentricity, a2, func_gen_id = 2,
					use_test_function = False, fun_idx = 5):
	"""
	Sets up the inference configuration by 1) setting up the corresponding PDE system and 2) generating
	boundary conditions to use as input to the STNN.

	Args:
		stnn_config (dict): Configuration dictionary for the spatial-temporal neural network (STNN)
							containing keys like 'nx1', 'nx2', 'nx3', 'ell_min', 'ell_max', 'a2_min', 'a2_max'.
		ell (float): PDE parameter ell.
		eccentricity (float): Eccentricity of the inner ellipse
		a2 (float): Minor axis of the outer ellipse
		func_gen_id (int, optional): Identifier for the function generator used for BCs if a test function
									 is not used. Default is 2.
		use_test_function (bool, optional): Flag to indicate whether to use a predefined test
											function from "data/test_functions.py".  Default is False.
		fun_idx (int, optional): Index of the test function to use if 'use_test_function' is True. Default is 2.

	Returns:
		tuple: A tuple containing:
			   - PDESystem: The initialized PDE system.
			   - numpy.ndarray: Normalized parameters array.
			   - numpy.ndarray: Combined boundary data.
			   - numpy.ndarray or tuple: Boundary conditions data
	"""
	# Create PDE config and build PDE system
	pde_config = {}
	for key in ['nx1', 'nx2', 'nx3']:
		pde_config[key] = stnn_config[key]
	pde_config['ell'] = ell
	pde_config['eccentricity'] = eccentricity
	pde_config['a2'] = a2
	system = PDESystem(pde_config)

	# Test data
	if use_test_function:
		ibf_data = get_test_function(system.x2_ib, system.x2_ib - system.x3_ib, fun_idx)[system.ib_slice]
		obf_data = get_test_function(system.x2_ob, system.x2_ob - system.x3_ob, fun_idx)[system.ob_slice]
		ibf_data, obf_data, b = system.convert_boundary_data(ibf_data, obf_data)
	else:
		ibf_data, obf_data, b, _ = system.generate_random_bc(func_gen_id)

	# Load some relevant quantities from the config dictionaries
	ell_min, ell_max = stnn_config['ell_min'], stnn_config['ell_max']
	a2_min, a2_max = stnn_config['a2_min'], stnn_config['a2_max']
	nx1, nx2, nx3 = pde_config['nx1'], pde_config['nx2'], pde_config['nx3']

	# Combine boundary data in single vector
	bf = np.zeros((1, 2 * nx2, nx3 // 2))
	bf[:, :nx2, :] = ibf_data[np.newaxis, ...]
	bf[:, nx2:, :] = obf_data[np.newaxis, ...]

	params = np.zeros((1, 3))
	params[0, 0] = (a2 - a2_min) / (a2_max - a2_min)
	params[0, 1] = (ell - ell_min) / (ell_max - ell_min)
	params[0, 2] = eccentricity

	return system, params, bf, b


def direct_solution(system, b):
	# Direct solution
	start = timeit.default_timer()
	L_xp = csr_matrix(system.L) # Sparse matrix representation of the PDE operator
	nx1, nx2, nx3 = system.params['nx1'], system.params['nx2'], system.params['nx3']
	b_xp = asarray(b.reshape((nx1 * nx2 * nx3,))) # r.h.s. vector

	def callback(res):
		print(f'GMRES residual: {res}')

	f_xp, info = spx.linalg.gmres(L_xp, b_xp, maxiter = 100000, tol = 1.0e-7, restart = 400, callback = callback)
	if info > 0:
		f_xp, info = spx.linalg.gmres(L_xp, b_xp, maxiter = 50000, tol = 1.0e-7, restart = 400)

	residual = (xp.linalg.norm(b_xp - L_xp @ f_xp) / xp.linalg.norm(b_xp))

	if info > 0:
		warnings.simplefilter('always')
		warnings.warn(f'GMRES solver did not converge. Number of iterations: {info}; residual: {residual}', RuntimeWarning)

	f = asnumpy(f_xp)
	rho_direct = np.sum(f.reshape((nx1, nx2, nx3)), axis = -1)
	direct_time = timeit.default_timer() - start
	print(f'Done with direct solution. Time: {direct_time} seconds.')

	return rho_direct, direct_time


if __name__ == '__main__':
	main()
