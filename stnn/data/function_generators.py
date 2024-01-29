import numpy as np


def generate_piecewise_linear_function(num_pieces, lower, upper, delta = 0.3):
	"""
	Generates a piece-wise linear function on the interval (lower, upper) that is 2*pi-periodic.

	Args:
		num_pieces (int): Number of linear pieces in the function.
		lower (float): The lower range of the interval
		upper (float): The upper range of the interval
		delta (float): Parameter determining how rapidly the function varies between the grid points (i.e., modulates
		the slopes of the piecewise functions). Larger values mean more variability. Default is 0.3.

	Returns:
		function: A piece-wise linear function.
	"""
	# Generate equally spaced points in the interval (-pi, pi)
	x_points = np.linspace(lower, upper, num_pieces + 1)

	# Generate random y-values for each point
	y_points = np.zeros(num_pieces + 1)
	y_points[0] = np.random.uniform(-1, 1)
	for n in range(1, y_points.shape[0]):
		y_points[n] = y_points[n - 1] + 0.3 * np.random.uniform(-1, 1)
	y_points[0] = y_points[-1]
	min_y = y_points.min()
	# ensure y values are nonegative
	if min_y < 0:
		y_points -= min_y
		y_points += np.random.uniform(0, 0.5)  # random (constant) offset

	def piecewise_linear(x):
		"""
		Evaluates the piece-wise linear function at a given x.

		Args:
			x (float): The x-coordinate at which to evaluate the function.

		Returns:
			float: The y-coordinate of the function at x.
		"""
		for i in range(num_pieces):
			if x_points[i] <= x < x_points[i + 1]:
				# Linear interpolation between the two points
				slope = (y_points[i + 1] - y_points[i]) / (x_points[i + 1] - x_points[i])
				return slope * (x - x_points[i]) + y_points[i]
		return y_points[0]  # For x = pi

	return piecewise_linear


def generate_piecewise_constant_function(num_pieces, lower, upper):
	"""
	Generates a piece-wise constant function on the interval (lower, upper) that is 2*pi periodic.

	Args:
		num_pieces (int): Number of constant pieces in the function.
		lower (float): The lower range of the interval
		upper (float): The upper range of the interval

	Returns:
		function: A piece-wise constant function.
	"""
	# Generate equally spaced points in the interval (-pi, pi)
	x_points = np.linspace(lower, upper, num_pieces + 1)

	# Generate random y-values for each constant piece
	y_values = np.random.rand(num_pieces) * 2 - 0  # Random values between 0 and 1

	# Ensure the function is 2*pi periodic
	y_values = np.append(y_values, y_values[0])

	def piecewise_constant(x):
		"""
		Evaluates the piece-wise constant function at a given x.

		Args:
			x (float): The x-coordinate at which to evaluate the function.

		Returns:
			float: The y-coordinate of the function at x.
		"""
		for i in range(num_pieces):
			if x_points[i] <= x < x_points[i + 1]:
				return y_values[i]
		return y_values[0]  # For x = pi

	return piecewise_constant


def generate_piecewise_bc(x2_grid, x3_grid, num_pieces):
	"""
	Generates a piecewise linear function on the domain defined by x2_grid and x3_grid.

	Args:
		x2_grid (numpy.ndarray): A 2D array, x2 grid values
		x3_grid (numpy.ndarray): A 2D array, x3 grid values
		num_pieces (int): The number of pieces in the piecewise linear function.

	Returns:
		numpy.ndarray: A 2D array representing the piecewise linear function
	"""
	x2_fun = generate_piecewise_linear_function(num_pieces, lower = x2_grid.min(), upper = x2_grid.max())
	x3_fun = generate_piecewise_linear_function(num_pieces, lower = x3_grid.min(), upper = x3_grid.max())

	x2vals_1d = np.zeros_like(x2_grid[:, 0])
	x3vals_1d = np.zeros_like(x3_grid[0, :])
	for i in range(x2vals_1d.shape[0]):
		x2vals_1d[i] = x2_fun(x2_grid[i, 0])
	for i in range(x3vals_1d.shape[0]):
		x3vals_1d[i] = x3_fun(x3_grid[0, i])
	x2vals_2d, x3vals_2d = np.meshgrid(x2vals_1d, x3vals_1d, indexing = 'ij')
	return x2vals_2d * x3vals_2d


def random_2d_gaussian(theta, phi):
	"""
	Generates a 2D Gaussian G(x,y), where
		x = np.cos(0.5 * freq_x * theta - phase_x)
		y = np.cos(0.5 * freq_y * phi - phase_y)
	Here, the frequencies and phases are randomly sampled, and (theta, phi) define a 2D meshgrid.

	Args:
		theta (numpy.ndarray): 2D array, meshgrid of the first coordinate
		phi (numpy.ndarray): 2D array, meshgrid of the second coordinate

	Returns:
		numpy.ndarray: A 2D array representing the values of the Gaussian on the grid.
	"""
	phase_x = np.random.uniform(0, 2 * np.pi)
	phase_y = np.random.uniform(0, 2 * np.pi)
	freq_x = np.random.randint(1, 2)
	freq_y = np.random.randint(1, 2)

	x = np.cos(0.5 * freq_x * theta - phase_x)
	y = np.cos(0.5 * freq_y * phi - phase_y)

	sigma_x = np.random.uniform(0.1, 3.0)
	sigma_y = np.random.uniform(0.1, 1.0)
	rho = 0

	covariance_matrix = np.array([[sigma_x**2, rho * sigma_x * sigma_y],
								  [rho * sigma_x * sigma_y, sigma_y**2]])
	inv_sigma_xx = 1.0 / sigma_x**2
	inv_sigma_yy = 1.0 / sigma_y**2
	inv_sigma_xy = -rho / (sigma_x * sigma_y)

	if np.any(np.linalg.eigvals(covariance_matrix) < 0):
		raise ValueError('Covariance matrix is not positive semi-definite.')

	def gaussian_2d(x, y):
		return np.exp(-0.5 * (inv_sigma_xx * x**2 + inv_sigma_yy * y**2 + 2 * inv_sigma_xy * x * y))

	gaussian_values = gaussian_2d(x, y)
	return gaussian_values


def generate_random_functions(N, X, Y, num_terms = 16, min_freq = 1, max_freq = 16, func_gen_id = 0):
	"""
	Generates N random 2pi-periodic functions on a 2D grid as a Fourier series, with different types of
	modulation applied to the amplitudes.

	Args:
		N (int): Number of functions to generate.
		X (numpy.ndarray): 2D array representing the values of the first coordinate on the grid
		Y (numpy.ndarray): 2D array representing the values of the second coordinate on the grid
		num_terms (int, optional): Number of terms in the Fourier series expansion. Default is 16.
		min_freq (int, optional): Minimum frequency for the Fourier series terms. Default is 1.
		max_freq (int, optional): Maximum frequency for the Fourier series terms. Default is 16.
		func_gen_id (int, optional): Type of function to generate based on the decay of the expansion coefficients
									 as frequency is increased. Values can range from -1 to 4. Default is 0.

	Returns:
		numpy.ndarray: A 3D numpy array of shape (N, nx, ny) containing the function values.

	Raises:
		ValueError: If max_freq is less than min_freq or if an invalid func_gen_id is provided.
	"""

	# Check if the maximum frequency is less than the minimum frequency
	if max_freq < min_freq:
		raise ValueError('max_freq cannot be less than min_freq')

	# Generate uniformly distributed functions if func_gen_id is -1
	if func_gen_id == -1:
		F_batch = np.random.uniform(0, 1, size = (N,) + X.shape)
		return F_batch

	# Initialize the batch of functions with zeros
	F_batch = np.zeros((N,) + X.shape)

	# Loop through each function to be generated
	for n in range(N):
		# Add a cosine term with a half frequency with 20% chance
		if np.random.uniform(0, 1) < 0.2:
			amp_cos_half = np.random.uniform(0, 1)  # Amplitude for cosine term
			phase_cos_half = np.random.uniform(0, 2 * np.pi)  # Phase shift for cosine term
			F_batch[n] += amp_cos_half * np.cos(0.5 * X - phase_cos_half)

		# Fourier series
		for _ in range(num_terms):
			amplitude = np.random.uniform(-1, 1)  # Random amplitude for y-component
			kx, ky = np.random.randint(min_freq, max_freq + 1, 2)  # Frequencies for x and y components
			phase_x = np.random.uniform(0, 2 * np.pi)  # Phase shift for x-component
			phase_y = np.random.uniform(0, 2 * np.pi)  # Phase shift for y-component

			# Determine the coefficient amplitude based on the func_gen_id
			if func_gen_id == 0:
				# No decay applied to amplitude
				pass
			elif func_gen_id == 1:
				if np.random.uniform(0, 1) < 0.5:
					amplitude = amplitude / kx
				else:
					amplitude = amplitude / ky
			elif func_gen_id == 2:
				amplitude = amplitude / (kx * ky)
			elif func_gen_id == 3:
				amplitude = amplitude / (kx * kx * ky * ky)
			elif func_gen_id == 4:
				# Gaussian decay with random covariance matrix
				sxx = np.random.uniform(0.1, 1.0)
				syy = np.random.uniform(0.1, 1.0)
				sxy = np.random.uniform(0.1, 1.0)
				amplitude = amplitude * np.exp(-(sxx * kx**2 + syy * ky**2 + sxy * kx * ky))
			else:
				raise ValueError(
					f'Invalid func_gen_id. Should be an integer in the range [-1, 4], but received {func_gen_id}')

			# Add the term to the nth function in the batch
			F_batch[n] += amplitude * np.cos(kx * X - phase_x) * np.cos(ky * Y - phase_y)

		# Adjust the function to ensure it's positive
		minF = np.min(F_batch[n])
		if minF < 0:
			F_batch[n] -= minF

	return F_batch
