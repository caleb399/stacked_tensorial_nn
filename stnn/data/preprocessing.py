import h5py
import numpy as np

# If STRICT_WARNING = True, the program exits when negative values are detected in ibf, obf, or rho
# This is important to check because negative values are unphysical.
STRICT_WARNING = True


def verify_nonnegative(fname, ibf, obf, rho):
	"""
	Check ibf, obf, and rho for negative values.
	"""
	found_warning = False
	if np.any(ibf < 0):
		print(f'Warning: negative values detected in array "ibf" in {fname}; min val: {ibf.min()}')
		found_warning = True
	elif np.any(obf < 0):
		print(f'Warning: negative values detected in array "obf" in {fname}')
		found_warning = True
	elif np.any(rho < 0):
		print(f'Warning: negative values detected in array "rho" in {fname}')
		found_warning = True

	if found_warning and STRICT_WARNING:
		print(f'Exiting program. To avoid exiting on this warning, set STRICT_WARNING to False in {__file__.name}')
		exit()


def get_data_from_file(fname, nx2, nx3, Nrange = None):
	"""
	Retrieves training X from the given HDF5 file. Assumes that the PDE parameters
	are stored in datasets with their respective names, i.e., 'ell', 'a1', 'a2'. Likewise,
	the density rho(x1,x2) and boundary X ibf(x2,x3) / obf(x2,x3) are stored in datasets
	'rho', 'ibf', and 'obf'.

	Args:
		 nx2 (int): Second grid dimension
		 nx3 (int): Third grid dimension
		 fname (str): Path to the HDF5 file containing the X.
		 Nrange (tuple, optional): A tuple of two integers specifying the range of X to extract (start, end).
		 						   Defaults to None.

	Returns:
		tuple: Tuple of extracted X

	Raises:
		ValueError: If the file does not contain the required datasets.
	"""
	if not isinstance(fname, str):
		raise TypeError('Filename must be a string.')
	type_check1 = not (Nrange is None or isinstance(Nrange, (tuple, list)))
	type_check2 = False
	if isinstance(Nrange, (tuple, list)):
		type_check2 = len(Nrange) != 2
		if not type_check2:
			type_check2 = not all((isinstance(i, int) or i is None) for i in Nrange)
	if type_check1 or type_check2:
		raise TypeError('Nrange must be a length-2 tuple or list of integers.')

	if Nrange is None:
		N1, N2 = None, None
	else:
		N1, N2 = Nrange

	# Check that all datasets are present
	dset_names = ['ell', 'a1', 'a2', 'rho', 'ibf', 'obf']
	with h5py.File(fname, 'r') as input_file:
		missing_keys = [key for key in dset_names if key not in input_file.keys()]
		if missing_keys:
			raise ValueError(f"Missing / incorrectly labeled datasets in file {fname}.'"
							 f"Could not find datasets: {', '.join(missing_keys)}")

		ell = input_file['ell'][N1:N2]
		a2 = input_file['a2'][N1:N2]  # minor axis of outer boundary
		a1 = input_file['a1'][N1:N2]  # minor axis of inner boundary
		eccentricity = np.ones_like(a1) - a1  # eccentricity of inner boundary
		rho = input_file['rho'][N1:N2]
		ibf = input_file['ibf'][N1:N2]  # boundary X on inner boundary
		obf = input_file['obf'][N1:N2]  # boundary X on outer boundary

		verify_nonnegative(fname, ibf, obf, rho)

		# Combine 'ibf' and 'obf' into single array
		N = rho.shape[0]
		bf = np.zeros((N, 2 * nx2, nx3 // 2), dtype = np.float32)
		bf[:, :nx2, :] = ibf
		bf[:, nx2:, :] = obf

	return a2, ell, eccentricity, bf, rho


def reshape_and_stack(a2, ell, ecc):
	a2 = a2.reshape((-1, 1))
	ell = ell.reshape((-1, 1))
	ecc = ecc.reshape((-1, 1))
	return np.hstack([a2, ell, ecc])


def apply_normalization(bf, rho):
	fac = np.average(np.abs(rho), axis = (1, 2))
	fac = fac.reshape((-1, 1, 1))
	bf /= fac
	rho /= fac
	return bf, rho


def load_data(files, nx2, nx3, ell_min, ell_max, a2_min, a2_max,
			  Nrange_list = None, params_slice = None, normalize_data = False):
	"""
	Loads X from the specified files and processes it for use with the STNN.

	Args:
		 nx2 (int): Second grid dimension
		 nx3 (int): Third grid dimension
		 ell_min / ell_max (float): Minimum / maximum value of 'ell' over parameter space
		 a2_min / a2_max (float): Minimum / maximum value of 'a2' over parameter space
		 files (str or list of str): List of file paths containing the X
		 Nrange_list (list of tuples, optional): Slice indices for the extracting X from the corresponding file. If
		 										 given, must have the same number of elements as 'file_list'. Defaults
		 										 to None.
		 params_slice (slice, optional): Boolean array for selecting X over a subset of parameter space (ell, a1, a2).
										 Defaults to None.
		 normalize_data (bool, optional): Flag to normalize 'bf' and 'rho'. Defaults to False.

	Returns:
		tuple: A tuple containing the values of ell, a1, a2, bf, and rho. The parameters
			   ell, a1, a2 are combined into a single array 'params'.
	"""
	if isinstance(files, (list, tuple)) and len(files) == 0:
		raise ValueError(f'List of files provided to "load_data" is empty.')
	if not isinstance(files, (list, tuple)):
		files = [files]
	if Nrange_list is None or len(Nrange_list) == 0:
		# Default
		Nrange_list = [None for _ in range(len(files))]
	else:
		# User-specified; check shapes
		if not isinstance(Nrange_list, (list, tuple)):
			Nrange_list = [Nrange_list]
		if len(files) != len(Nrange_list):
			raise ValueError('List of input files must have same length as list of Nrange tuples')
	a2_list = []
	ell_list = []
	ecc_list = []
	bf_list = []
	rho_list = []

	# Get X from each file and add to the lists
	for file, Nrange in zip(files, Nrange_list):
		a2, ell, ecc, bf, rho = get_data_from_file(file, nx2, nx3, Nrange = Nrange)
		a2_list.append(a2)
		ell_list.append(ell)
		ecc_list.append(ecc)
		bf_list.append(bf)
		rho_list.append(rho)

	a2 = np.concatenate(a2_list)
	ell = np.concatenate(ell_list)
	ecc = np.concatenate(ecc_list)
	bf = np.vstack(bf_list)
	rho = np.vstack(rho_list)

	# Map ell and a2 values onto [0, 1]
	ell = (ell - ell_min) / (ell_max - ell_min)
	a2 = (a2 - a2_min) / (a2_max - a2_min)

	params = reshape_and_stack(a2, ell, ecc)

	if not params_slice is None:
		# Extract subset of X, if params_slice is given
		params = params[params_slice, ...]
		bf = bf[params_slice, ...]
		rho = rho[params_slice, ...]

	if normalize_data:
		bf, rho = apply_normalization(bf, rho)

	return params, bf, rho


def load_training_data(file_list, nx2, nx3, ell_min, ell_max, a2_min, a2_max, Nrange_list = None,
					   params_slice = None, test_size = 0.1, random_state = 23, normalize_data = True):
	"""
	Loads training X from specified files and preprocesses it for use with training the STNN.

	This function wraps the 'load_data' function, adding additional steps specific to preparing training X.

	Args:
		 nx2 (int): Second grid dimension
		 nx3 (int): Third grid dimension
		 ell_min / ell_max (float): Minimum / maximum value of 'ell' over parameter space
		 a2_min / a2_max (float): Minimum / maximum value of 'a2' over parameter space
		 file_list (list of str): List of file paths containing the X
		 Nrange_list (list of tuples, optional): Slice indices for the extracting X from the corresponding file. If
		 										 given, must have the same number of elements as 'file_list'. Defaults
		 										 to None.
		 params_slice (slice, optional): Boolean array for selecting X over a subset of parameter space (ell, a1, a2).
										 Defaults to None.
		 test_size (float, optional): Size of the test/validation dataset as a fraction of the total dataset size.
		 							  Defaults to 0.1.
		 random_state (int, optional): Random seed used to select the train-test split. Defaults to 23.
		 normalize_data (bool, optional): Flag to normalize 'bf' and 'rho'. Defaults to False.

	Returns:
		tuple: A tuple containing the values of ell, a1, a2, bf, and rho. The parameters
			   ell, a1, a2 are combined into a single array 'params'.
"""
	params, bf, rho = load_data(file_list, nx2, nx3, ell_min, ell_max, a2_min, a2_max,
								Nrange_list = Nrange_list, params_slice = params_slice, normalize_data = normalize_data)

	(rho_train, rho_test,
	 Y_train, Y_test) = train_test_split(rho, [params, bf], test_size = test_size, random_state = random_state)

	params_train = Y_train[0]
	params_test = Y_test[0]
	bf_train = Y_train[1]
	bf_test = Y_test[1]

	print('Finished loading training X:')
	print(f'  params_train.shape:\t{params_train.shape}')
	print(f'  bf_train.shape:\t{bf_train.shape}')
	print(f'  rho_train.shape:\t{rho_train.shape}')
	print(f'  params_test.shape:\t{params_test.shape}')
	print(f'  bf_test.shape:\t{bf_test.shape}')
	print(f'  rho_test.shape:\t{rho_test.shape}')

	# Compute min/max extent of training X in parameter space.
	# Note that 'params' is denormalized before computing the max/min.
	min_a2 = np.min(a2_min + (a2_max - a2_min) * params[:, 0])
	min_ell = np.min(ell_min + (ell_max - ell_min) * params[:, 1])
	min_ecc = np.min(params[:, 2])

	max_a2 = np.max(a2_min + (a2_max - a2_min) * params[:, 0])
	max_ell = np.max(ell_min + (ell_max - ell_min) * params[:, 1])
	max_ecc = np.max(params[:, 2])

	print('')
	print(f'  Number of circle samples  (train):\t{np.sum(params[:, 2] < 1e-7)}')
	print(f'  Number of ellipse samples (train):\t{np.sum(params[:, 2] > 0)}')
	print(f'  Min .. Max in training X:')
	print(f'     ell:\t{min_ell:.2f} .. {max_ell:.2f}')
	print(f'     a2:\t{min_a2:.2f} .. {max_a2:.2f}')
	print(f'     ecc:\t{min_ecc:.2f} .. {max_ecc:.2f}')
	print('-------------------------------------------')

	return params_train, bf_train, rho_train, params_test, bf_test, rho_test


def train_test_split(X, Y, test_size = 0.1, random_state = None):
	"""
	Split (X, Y) pairs into random train and test subsets.

	Args:
		X (np.ndarray or list of arrays): Input dataset
		Y (np.ndarray or list of arrays): Labels for the dataset
		test_size (float): Proportion of the dataset to include in the test split
		random_state (int): Controls the shuffling applied to the X and Y before applying the split

	Returns:
		X_train, X_test, Y_train, Y_test: Lists containing train-test split of the dataset. The format is
		the same as the input X. For example, if 'X' is an array and 'Y' is a list of arrays, then X_train
		and X_test will be arrays, and Y_train and Y_test will be lists of arrays.

	Note: This function is included primarilyto reduce module dependency requirements, and it may not be memory-efficient
		  for large datasets. sklearn.model_selection.train_test_split has similar functionality and may be preferred
		  for performance-critical applications.
	"""
	if len(X) == 0 or len(Y) == 0:
		raise ValueError("Input arrays/lists X and Y cannot be empty.")

	input_X_is_array = isinstance(X, np.ndarray)
	input_Y_is_array = isinstance(Y, np.ndarray)

	if input_X_is_array:
		X = [X]
	if input_Y_is_array:
		Y = [Y]

	total_samples = X[0].shape[0]

	# Check for consistent number of samples across all datasets
	if any(x.shape[0] != total_samples for x in X) or any(y.shape[0] != total_samples for y in Y):
		raise ValueError('Inconsistent number of samples.')

	Ntest = int(test_size * total_samples)
	if Ntest < 1 or Ntest > total_samples:
		raise ValueError('Size of test dataset cannot be less than 1 or greater than the total number of samples.')

	if random_state is not None:
		np.random.seed(random_state)

	# Shuffle indices
	indices = np.arange(total_samples)
	np.random.shuffle(indices)

	# Apply shuffled indices to all datasets
	shuffled_X = [x[indices] for x in X]
	shuffled_Y = [y[indices] for y in Y]

	# Split X and Y
	X_train = [x[:-Ntest] for x in shuffled_X]
	X_test = [x[-Ntest:] for x in shuffled_X]
	Y_train = [y[:-Ntest] for y in shuffled_Y]
	Y_test = [y[-Ntest:] for y in shuffled_Y]

	# Convert back to arrays if original input was array
	if input_X_is_array:
		X_train, X_test = X_train[0], X_test[0]
	if input_Y_is_array:
		Y_train, Y_test = Y_train[0], Y_test[0]

	return X_train, X_test, Y_train, Y_test
