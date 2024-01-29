import os
import timeit

import h5py
import numpy as np

from stnn.linalg_backend import xp, spx, asarray, asnumpy, csr_matrix
from stnn.pde.pde_system import PDESystem
from stnn.utils.input_output import log_and_print, save_to_hdf5

"""
This script generates training data following the recipe from (Wagner, 2023).
With the default values, it will generate 100 solutions rho(x,y) for randomly sampled
inputs ell, a1, a2, and bf (boundary data). The data will be written to the file
'generated_data.h5'. 

Note that if the file already exists, data will be appended up to 'nsamples'.
So, for example, if the file already has 10 samples, the script will simply exit.

The runtime will vary depending on the sample functions generated. For example, it is
possible that a "difficult" boundary condition will be generated that requires significantly
more iterations of GMRES to converge. On average, the script takes a few minutes on an 
RTX 3090 Ti.
"""


def main():
	num_samples = 10
	dims = (256, 64, 32)
	geom = 'ellipse'
	func_gen_id = 2
	start = timeit.default_timer()
	generate_data(num_samples, dims, geom, func_gen_id, output_filename = 'generated_data.h5')
	print(f'Total time: {timeit.default_timer() - start} seconds.')


def generate_data(num_samples, dims, geom, func_gen_id,
				  save_freq = 10, use_x0 = False, Linv_path = None,
				  output_filename = None):
	"""
	Generates labeled data with inputs (i.e., the boundary conditions) generated from the
	function generator specified by "func_gen_id".

	Args:
		num_samples (int): Number of samples to generate.
		dims (tuple of 3 integers): Grid dimensions (nx1, nx2, nx3)
		geom (str): Type of geometry, either 'ellipse' or 'circle'.
		func_gen_id (int): Class of the function to use.
		save_freq (int, optional): Frequency for saving data. Default is 2.
		use_x0 (bool, optional): Flag to determine whether to use a specific initial condition. Default is False.
		Linv_path (str, optional): Path to the linear inversion file. Default is None.
		output_filename (str, optional): Filename for the output file. Default is 'output.h5'.

	Returns:
		This function does not return a value. Instead, it generates and saves data to a file.

	Raises:
		ValueError: If an invalid geometry type is provided.
	"""

	# Default output output_filename
	if output_filename is None:
		output_filename = f'output{func_gen_id}.h5'

	nx1, nx2, nx3 = dims

	# Setting the range for ellipse and circle parameters
	ell_min, ell_max = 0.01, 1.0
	a2_min, a2_max = 2.0, 20.0

	# a1 (minor axis) is fixed at 1 for the circular geometry
	if geom == 'ellipse':
		a1_min, a1_max = 0.2, 0.999
	elif geom == 'circle':
		a1_min, a1_max = None, None
	else:
		raise ValueError(f'"geom" should be either "ellipse" or "circle"; instead received {geom}')

	# Whether to use initial guess in GMRES. This requires the full inverse solution operator
	# to be stored in 'Linv_path'.
	if use_x0:
		Linv = np.load(Linv_path)['Linv']
		Linv_xp = asarray(Linv)

	params = {
		'nx1': nx1,
		'nx2': nx2,
		'nx3': nx3,
	}

	rho_data = np.zeros((num_samples, nx1, nx2), dtype = np.float64)
	ibf_data = np.zeros((num_samples, nx2, nx3 // 2), dtype = np.float64)
	obf_data = np.zeros((num_samples, nx2, nx3 // 2), dtype = np.float64)
	ell = np.zeros(num_samples, dtype = np.float64)
	a1 = np.zeros(num_samples, dtype = np.float64)
	a2 = np.zeros(num_samples, dtype = np.float64)
	datasets = {
		'rho_data': rho_data,
		'ibf_data': ibf_data,
		'obf_data': obf_data,
		'ell': ell,
		'a1': a1,
		'a2': a2
	}

	n0 = 0
	if os.path.exists(output_filename):
		with h5py.File(output_filename, 'r') as input_file:
			n0 = input_file['ell'].shape[0]
			for name, data in datasets.items():
				if name in input_file:
					data[:n0, ...] = input_file[name][:n0, ...]

	n = n0
	while n < num_samples:
		ell[n] = np.random.uniform(ell_min, ell_max)
		a2[n] = np.random.uniform(a2_min, a2_max)
		if geom == 'ellipse':
			a1[n] = np.random.uniform(a1_min, a1_max)
		elif geom == 'circle':
			a1[n] = 1.0

		params['ell'] = ell[n]
		params['a1'] = a1[n]
		params['a2'] = a2[n]
		params['eccentricity'] = 1.0 - a1[n]

		system = PDESystem(params)
		ibf_data[n, ...], obf_data[n, ...], b, bf = system.generate_random_bc(func_gen_id)

		# Send b to linear algebra backend and flatten to a vector (1d array)
		b_xp = asarray(b.reshape((nx1 * nx2 * nx3,)))

		# Send system matrix to linear algebra backend
		L_xp = csr_matrix(system.L)

		# Construct initial iterate for GMRES, if requested.
		if use_x0:
			x0_xp = Linv_xp @ asarray(bf)
		else:
			x0_xp = None

		f_xp, info = spx.linalg.gmres(L_xp, b_xp, maxiter = 100000, tol = 1.0e-7, restart = 80, x0 = x0_xp)
		if info > 0:
			f_xp, info = spx.linalg.gmres(L_xp, b_xp, maxiter = 50000, tol = 1.0e-7, restart = 400, x0 = x0_xp)

		residual = xp.linalg.norm(b_xp - L_xp @ f_xp) / xp.linalg.norm(b_xp)
		log_string = (
			f'n = {n};'
			f'\tell = {ell[n]:.4f};'
			f'\ta1 = {a1[n]:.4f};'
			f'\ta2 = {a2[n]:.4f};'
			f'\tb2 = {system.b2:.4f};'
			f'\tresidual: {residual:.5e};'
			f'\tbnorm: {xp.linalg.norm(b_xp):.4f}'
		)
		log_and_print(log_string)

		# If no convergence, start over at beginning of loop
		if info > 0 or residual > 1e-6:
			log_string = (
				f'Failed at ell = {ell[n]:.4f};'
				f'\ta1 = {a1[n]:.4f};'
				f'\ta2 = {a2[n]:.4f};'
				f'\tb2 = {system.b2:.4f}'
			)
			log_and_print(log_string)
			continue

		# transfer out of linear algebra backend
		f = asnumpy(f_xp)
		rho_data[n, ...] = np.sum(f.reshape((nx1, nx2, nx3)), axis = -1)

		# write to output file
		if (n + 1) % save_freq == 0:
			start_idx = n - save_freq + 1
			end_idx = n + 1
			save_to_hdf5(output_filename, datasets, start_idx, end_idx)
		n += 1


if __name__ == "__main__":
	main()
