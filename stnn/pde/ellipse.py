from .common import *


def u_dot_muvec(mu, eta, w):
	"""
	Dot product of u = (cos(w), sin(w)) with the coordinate vector for mu.

	Args:
		mu (float or array-like): The mu coordinate(s).
		eta (float or array-like): The eta coordinate(s).
		w (float or array-like): The w coordinate(s).

	Returns:
		numpy.ndarray: The calculated dot product for each point.
	"""
	return (0.5 * np.cosh(mu) * np.cos(eta - w) + 0.5 * np.cosh(mu) * np.cos(eta + w)
			+ 0.5 * np.sinh(mu) * np.cos(eta - w) - 0.5 * np.sinh(mu) * np.cos(eta + w))


def u_dot_etavec(mu, eta, w):
	"""
	Dot product of u = (cos(w), sin(w)) with the coordinate vector for eta.

	Args:
		mu (float or array-like): The mu coordinate(s).
		eta (float or array-like): The eta coordinate(s).
		w (float or array-like): The w coordinate(s).

	Returns:
		numpy.ndarray: The calculated dot product for each point.
	"""
	return (-0.5 * np.sinh(mu) * np.sin(eta + w) - 0.5 * np.sinh(mu) * np.sin(eta - w)
			+ 0.5 * np.cosh(mu) * np.sin(eta + w) - 0.5 * np.cosh(mu) * np.sin(eta - w))


def get_system_ellipse(config):
	"""
	For an elliptical geometry, constructs the matrices, grids, and other quantities corresponding to the PDE system
	specified by "config"".

	Args:
		config (dict): Configuration dictionary containing the system parameters.

	Returns:
		tuple: A tuple containing matrices, grids, etc. for the PDE system
	"""
	required_keys = ['nx1', 'nx2', 'nx3', 'ell', 'a2', 'eccentricity']
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

	for key in ['nx1', 'nx2', 'nx3', 'ell']:
		if config[key] <= 0:
			raise ValueError(f"{key} must be positive.")

	if not (0 <= config['eccentricity'] < 1.0):
		raise ValueError('eccentricity must be >= 0 and < 1.')

	if config['a2'] <= 1.0:
		raise ValueError('a2 must be greater than 1.')

	nmu, neta, nw = config['nx1'], config['nx2'], config['nx3']
	minor_axis_outer = config['a2']
	ell = config['ell']
	minor_axis = 1.0 - config['eccentricity']
	major_axis = 1.0
	focal_distance = np.sqrt(major_axis**2 - minor_axis**2)
	mu1 = np.arccosh(major_axis / focal_distance)
	major_axis_outer = np.sqrt(focal_distance**2 + minor_axis_outer**2)
	mu2 = np.arccosh(major_axis_outer / focal_distance)

	# 1D grids
	eta, w = get_angular_grids(neta, nw)
	# mu grid: non-uniform spacing and Dirichlet boundary conditions
	y = np.linspace(-np.pi / 2, np.pi / 2, nmu + 2, dtype = np.float64)
	mu_ = np.log((np.exp(mu2) - np.exp(mu1)) * (np.sin(y) / 2 + 0.5) + np.exp(mu1))
	dmu1 = mu_[1] - mu_[0]
	dmu2 = mu_[-1] - mu_[-2]
	mu = mu_[1:-1]

	# 1D finite-difference operators
	Deta_minus, Deta_plus = d_dx_upwind(eta, neta)
	D2w = d2_dx2_fourth_order(w, nw)
	Dmu_minus, Dmu_plus = d_dx_upwind_nonuniform(mu_, nmu)

	# 3D quantities. Kronecker products are used to build the 3D difference operators
	mu_3D, eta_3D, w_3D = np.meshgrid(mu, eta, w, indexing = 'ij')
	I_mu = sp.eye(nmu)
	I_eta = sp.eye(neta)
	I_w = sp.eye(nw)
	Deta_3D_minus = sp.kron(sp.kron(I_mu, Deta_minus), I_w)
	Deta_3D_plus = sp.kron(sp.kron(I_mu, Deta_plus), I_w)
	D2w_3D = sp.kron(sp.kron(I_mu, I_eta), D2w)
	Dmu_3D_minus = sp.kron(sp.kron(Dmu_minus, I_eta), I_w)
	Dmu_3D_plus = sp.kron(sp.kron(Dmu_plus, I_eta), I_w)

	# Metric tensor. Note that g_12 = g_21 = 0 and g_11 = g_22.
	g_11 = focal_distance * (np.cosh(mu_3D) * np.cosh(mu_3D) * np.cos(eta_3D) * np.cos(eta_3D)
			+ np.sinh(mu_3D) * np.sinh(mu_3D) * np.sin(eta_3D) * np.sin(eta_3D))

	# Dot products
	dp_mu = u_dot_muvec(mu_3D, eta_3D, w_3D)
	dp_eta = u_dot_etavec(mu_3D, eta_3D, w_3D)

	# Coefficient of d / dmu
	Dmu_3D_coeff_meshgrid = dp_mu / g_11
	test_ill_conditioned(Dmu_3D_coeff_meshgrid)

	Dmu_3D_coeff = sp.diags(Dmu_3D_coeff_meshgrid.ravel())

	# Coefficient of d / deta
	Deta_3D_coeff_meshgrid = dp_eta / g_11
	Deta_3D_coeff = sp.diags(Deta_3D_coeff_meshgrid.ravel())

	# Upwind differencing
	Dmu_3D_upwind = upwind_operator(Dmu_3D_minus, Dmu_3D_plus, Dmu_3D_coeff_meshgrid)
	Deta_3D_upwind = upwind_operator(Deta_3D_minus, Deta_3D_plus, Deta_3D_coeff_meshgrid)

	# Full operator
	L = Dmu_3D_coeff @ Dmu_3D_upwind + Deta_3D_coeff @ Deta_3D_upwind - (1 / ell) * D2w_3D

	return L, mu_3D, eta_3D, w_3D, dmu1, dmu2, Dmu_3D_coeff_meshgrid, major_axis_outer


def get_boundary_quantities_ellipse(mu_3D, eta_3D, w_3D):
	"""
	Gets grid coordinates on the boundaries, as well as slice arrays
	for positive/negative angles with respect to the boundary angle.

	Args:
		mu_3D: 3D array of mu values on the grid.
		eta_3D: 3D array of eta values on the grid.
		w_3D (numpy.ndarray): 3D array of w values on the grid.

	Returns:
		tuple: Tuple of the grid coordinates and slice arrays
	"""
	eta_2D_ib = eta_3D[0, ...]
	eta_2D_ob = eta_3D[-1, ...]
	w_2D_ib = w_3D[0, ...]
	w_2D_ob = w_3D[-1, ...]
	ib_slice = u_dot_muvec(mu_3D, eta_3D, w_3D)[0, ...] > 0
	ob_slice = u_dot_muvec(mu_3D, eta_3D, w_3D)[-1, ...] < 0
	return eta_2D_ib, eta_2D_ob, w_2D_ib, w_2D_ob, ib_slice, ob_slice
