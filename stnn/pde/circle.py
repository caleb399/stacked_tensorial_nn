from .common import *


def u_dot_thetavec(r, theta, w):
	"""
	Dot product of u = (cos(w), sin(w)) with the coordinate vector for theta.

	Args:
	 	r (float or array-like): the radial coordinate(s)
		theta (float or array-like): The angular coordinate(s).
		w (float or array-like): The w coordinate(s).

	Returns:
		numpy.ndarray: The calculated dot product for each point.
	"""
	return r * np.sin(w - theta)


def u_dot_thetahat(theta, w):
	"""
	Dot product of u = (cos(w), sin(w)) with the unit vector for theta.

	Args:
		theta (float or array-like): The angular (theta) coordinate(s).
		w (float or array-like): The w coordinate(s).

	Returns:
		numpy.ndarray: The calculated dot product for each point.
	"""
	return np.sin(w - theta)


def u_dot_rvec(theta, w):
	"""
	Dot product of u = (cos(w), sin(w)) with the coordinate vector for r.

	Args:
		theta (float or array-like): The radial (r) coordinate(s).
		w (float or array-like): The w coordinate(s).

	Returns:
		numpy.ndarray: The calculated dot product for each point.
	"""
	return np.cos(w - theta)


def get_system_circle(config):
	"""
	For a circular geometry, constructs the matrices, grids, and other quantities corresponding to the PDE system
	specified by "config"".

	Args:
		config (dict): Configuration dictionary containing the system parameters.

	Returns:
		tuple: A tuple containing matrices, grids, etc. for the PDE system
	"""
	required_keys = ['nx1', 'nx2', 'nx3', 'ell', 'a2']
	optional_keys = []

	missing_keys = [key for key in required_keys if key not in config]
	if missing_keys:
		raise KeyError(f"Missing keys in config: {', '.join(missing_keys)}")

	unused_keys = [key for key in config if key not in required_keys + optional_keys]
	if unused_keys:
		warnings.warn(f"Unused keys in config: {', '.join(unused_keys)}")

	for key in ['nx1', 'nx2', 'nx3']:
		if not isinstance(config[key], int):
			raise TypeError(f"{key} must be an integer.")

	for key in ['nx1', 'nx2', 'nx3', 'ell', 'a2']:
		if config[key] <= 0:
			raise ValueError(f"{key} must be positive.")

	if config['a2'] < 1.0:
		raise ValueError('a2 must be greater than 1.')

	nr, ntheta, nw = config['nx1'], config['nx2'], config['nx3']
	R1 = 1.0
	R2 = config['a2']
	ell = config['ell']

	# 1D grids
	theta, w = get_angular_grids(ntheta, nw)
	# r grid: non-uniform spacing and Dirichlet boundary conditions
	y = np.linspace(-np.pi / 2, np.pi / 2, nr + 2)
	r_ = (R2 - R1) * (np.sin(y) / 2 + 0.5) + R1
	dr1 = r_[1] - r_[0]
	dr2 = r_[-1] - r_[-2]
	r = r_[1:-1]

	# 1D finite-difference operators
	Dtheta_minus, Dtheta_plus = d_dx_upwind(theta, ntheta)
	D2w = d2_dx2_fourth_order(w, nw)
	Dr_minus, Dr_plus = d_dx_upwind_nonuniform(r_, nr)

	# 3D quantities. Kronecker products are used to build the 3D difference operators
	r_3D, theta_3D, w_3D = np.meshgrid(r, theta, w, indexing = 'ij')
	I_r = sp.eye(nr)
	I_theta = sp.eye(ntheta)
	I_w = sp.eye(nw)
	Dtheta_3D_minus = sp.kron(sp.kron(I_r, Dtheta_minus), I_w)
	Dtheta_3D_plus = sp.kron(sp.kron(I_r, Dtheta_plus), I_w)
	D2w_3D = sp.kron(sp.kron(I_r, I_theta), D2w)
	Dr_3D_minus = sp.kron(sp.kron(Dr_minus, I_theta), I_w)
	Dr_3D_plus = sp.kron(sp.kron(Dr_plus, I_theta), I_w)

	# Metric tensor. Note that g_12 = g_21 = 0.
	g_11 = np.ones_like(r_3D)
	g_22_over_r = r_3D  # divide out factor of r

	# Dot products
	dp_r = u_dot_rvec(theta_3D, w_3D)
	dp_thetahat = u_dot_thetahat(theta_3D, w_3D)

	# Coefficient of d / dr
	Dr_3D_coeff_meshgrid = dp_r / g_11
	test_ill_conditioned(Dr_3D_coeff_meshgrid)
	Dr_3D_coeff = sp.diags(Dr_3D_coeff_meshgrid.ravel())

	# Coefficient of d / dtheta
	Dtheta_3D_coeff_meshgrid = dp_thetahat / g_22_over_r
	Dtheta_3D_coeff = sp.diags(Dtheta_3D_coeff_meshgrid.ravel())

	# Upwind differencing
	Dr_3D_upwind = upwind_operator(Dr_3D_minus, Dr_3D_plus, Dr_3D_coeff_meshgrid)
	Dtheta_3D_upwind = upwind_operator(Dtheta_3D_minus, Dtheta_3D_plus, Dtheta_3D_coeff_meshgrid)

	# Full operator
	L = Dr_3D_coeff @ Dr_3D_upwind + Dtheta_3D_coeff @ Dtheta_3D_upwind - (1 / ell) * D2w_3D

	return L, r_3D, theta_3D, w_3D, dr1, dr2, Dr_3D_coeff_meshgrid


def get_boundary_quantities_circle(theta_3D, w_3D):
	"""
	Gets grid coordinates on the boundaries, as well as slice arrays
	for positive/negative angles with respect to the boundary angle.

	Args:
		theta_3D (numpy.ndarray): 3D array of theta values on the grid.
		w_3D (numpy.ndarray): 3D array of w values on the grid.

	Returns:
		tuple: Tuple of the grid coordinates and slice arrays
	"""
	th1 = theta_3D[0, :, :]
	wb1 = w_3D[0, :, :]
	th2 = theta_3D[-1, :, :]
	wb2 = w_3D[-1, :, :]
	ib_slice = np.cos(th1 - wb1) > 0
	ob_slice = np.cos(th2 - wb2) < 0

	return th1, th2, wb1, wb2, ib_slice, ob_slice
