import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
import json

from stnn.utils.plotting import plot_comparison
from stnn.pde.pde_system import PDESystem
from stnn.nn.stnn import build_stnn
from stnn.data.preprocessing import load_data

"""
This script performs inference on the test datasets from (Wagner, 2023) and outputs
some statistics (relative L^2 norm difference and a few percentiles). It also plots 
a comparison of rho_predicted and rho_actual for the maximum L^2 norm difference and 
the 50, 90, 95, and 99 percentiles.

To run, specify the trial model ([1-7]; see Table 1 in (Wagner, 2023)) and the test dataset
as command-line arguments. The test dataset ID can be one of 
["piecewise", "oob", "zero_flux", "random", "boundary_layer"]

Running the script requires the test data, which can be downloaded from TBD (to be determined).

Example usage:

	python inference_batched.py -t 5 -i piecewise
"""

if True:
	raise NotImplementedError('This script is designed to operate on the test datasets from the paper, '
							  'which have not yet been included with the repository.')


# noinspection PyUnreachableCode
def main():
	# Get trial number and test id from command line arguments
	parser = argparse.ArgumentParser(description = 'Specify the trial number and test id')
	parser.add_argument('--trial_number', '-t', type = int, required = True,
						help = 'The trial number [1-7]')
	parser.add_argument('--test_id', '-i', type = str, required = True,
						help = 'The test ID [one of "piecewise", "oob", "zero_flux", "random", "boundary_layer"]')
	args = parser.parse_args()
	trial_number = args.trial_number
	test_id = args.test_id

	# Input files for trial
	weights_file = os.path.join('models', f'trial_{trial_number}', f'T{trial_number}.h5')
	stnn_config_file = os.path.join('models', f'trial_{trial_number}', f'T{trial_number}_config.json')

	# Output directories
	trialdir = f'trial_{trial_number}'
	if not os.path.exists(trialdir):
		os.mkdir(trialdir)
	testdir = os.path.join(trialdir, test_id)
	if not os.path.exists(testdir):
		os.mkdir(testdir)
	plotdir = os.path.join(testdir, 'plots')
	if not os.path.exists(plotdir):
		os.mkdir(plotdir)

	# STNN model config is loaded from the provided json file.
	with open(stnn_config_file, 'r', encoding = 'utf-8') as json_file:
		stnn_config = json.load(json_file)

	# Load test data from 'datasets' folder
	params, bf, rho = load_test_data(test_id, stnn_config)

	# Build STNN
	model = build_stnn(stnn_config)
	model.load_weights(weights_file)
	model.compile(optimizer = 'adam', loss = 'mean_absolute_error')

	# Inference
	rho_pred = model.predict([params, bf])

	# Relative mean-squared-errors
	Nsamples = rho.shape[0]
	diff_flattened = (rho - rho_pred).reshape((Nsamples, -1))
	rho_flattened = rho.reshape((Nsamples, -1))
	errs = np.linalg.norm(diff_flattened, axis = 1) / np.linalg.norm(rho_flattened, axis = 1)
	print(f'Average relative error: {np.mean(errs)}')
	max_err_idx = np.argmax(errs)
	print(f'Max relative error: {errs[max_err_idx]}; index: {max_err_idx}')

	# Error percentiles
	percentiles = {}
	closest_indices = {}
	for p in [50, 90, 95, 99]:
		percentiles[p] = np.percentile(errs, p)
		closest_indices[p] = np.argmin(np.abs(errs - percentiles[p]))
		print(f'{p}th percentile error: {percentiles[p]:.4f};\tclosest index: {closest_indices[p]}')

	# Plot histogram with 95th percentile labeled
	plot_histogram(plotdir, errs, percentiles[95])

	# Get parameter bounds from STNN config
	ell_min, ell_max = stnn_config['ell_min'], stnn_config['ell_max']
	a2_min, a2_max = stnn_config['a2_min'], stnn_config['a2_max']

	# Configuration dictionary for creating PDESystem object
	params_dict = {}
	for key in ['nx1', 'nx2', 'nx3']:
		params_dict[key] = stnn_config[key]

	# Plot comparisons of rho and rho_pred for the instances closest to the percentiles,
	# as well as the instance with the max error.
	for idx in list(closest_indices.keys()) + [max_err_idx]:
		params_dict['a2'] = a2_min + (a2_max - a2_min) * params[idx, 0]
		params_dict['ell'] = ell_min + (ell_max - ell_min) * params[idx, 1]
		params_dict['eccentricity'] = params[idx, 2]
		system = PDESystem(params_dict)
		plot_comparison(system, bf[idx, ...], rho[idx, ...], rho_pred[idx, ...],
						output_filename = os.path.join(plotdir, f'comparison_{idx}.png'))


def load_test_data(test_id, stnn_config):
	nx2, nx3 = stnn_config['nx2'], stnn_config['nx3']
	ell_min, ell_max = stnn_config['ell_min'], stnn_config['ell_max']
	a2_min, a2_max = stnn_config['a2_min'], stnn_config['a2_max']
	input_files = []

	# test data gets loaded here

	params, bf, rho = load_data(input_files, nx2, nx3, ell_min, ell_max, a2_min, a2_max)

	return params, bf, rho


def plot_histogram(plotdir, errs, percentile_95):
	# plot histogram of relative errors
	plt.hist(errs, bins = 30, color = 'blue', alpha = 0.7)
	plt.axvline(percentile_95, color = 'red', linestyle = 'dashed', linewidth = 2)
	plt.title('Histogram with 95th Percentile')
	plt.xlabel('Value')
	plt.ylabel('Frequency')
	plt.text(percentile_95, plt.ylim()[1] * 0.9,
			 f'95th Percentile: {percentile_95:.2f}', color = 'red', horizontalalignment = 'right')
	plt.savefig(os.path.join(plotdir, 'histogram.png'))


if __name__ == '__main__':
	main()
