import warnings
import numpy as np
import scipy.sparse as sp


def d_dx_upwind(x, nx):
	"""
	Sparse matrix representation of d/dx using first-order left/right differences with Dirichlet boundary conditions
	"""
	dx = x[1] - x[0]
	Dx_minus = sp.diags([-1, 1], [0, 1], shape = (nx, nx)).tolil() / dx
	Dx_plus = sp.diags([-1, 1], [-1, 0], shape = (nx, nx)).tolil() / dx
	Dx_minus[-1, 0] = 1 / dx
	Dx_plus[0, -1] = -1 / dx
	Dx_minus = Dx_minus.tocsr()
	Dx_plus = Dx_plus.tocsr()
	return Dx_minus, Dx_plus


def d2_dx2_fourth_order(x, nx):
	"""
	Sparse matrix representation of d^2/dx^2 using fourth order central differences and periodic boundary conditions
	"""
	dx = x[1] - x[0]
	D2x = sp.diags([-1, 16, -30, 16, -1], [-2, -1, 0, 1, 2],
				   shape = (nx, nx)).tolil() / (12 * dx**2)
	D2x[0, -1] = 16 / (12 * dx**2)
	D2x[0, -2] = -1 / (12 * dx**2)
	D2x[1, -1] = -1 / (12 * dx**2)
	D2x[-1, 0] = 16 / (12 * dx**2)
	D2x[-1, 1] = -1 / (12 * dx**2)
	D2x[-2, 0] = -1 / (12 * dx**2)
	D2x = D2x.tocsr()
	return D2x


def d_dx_upwind_nonuniform(x, nx):
	"""
	Sparse matrix representation of d/dx on a nonuniform grid, using first-order left/right differences 
	with Dirichlet boundary conditions.
	"""
	Dx_ = np.diff(x)
	Dx_minus = np.diff(x[1:])
	Dx_minus_inv = 1 / Dx_minus
	Dx_plus_inv = 1 / Dx_
	Dx_minus = sp.diags([-Dx_minus_inv, Dx_minus_inv], [0, 1], shape = (nx, nx)).tolil()
	Dx_plus = sp.diags([-Dx_plus_inv[1:], Dx_plus_inv[:-1]], [-1, 0], shape = (nx, nx)).tolil()
	Dx_minus = Dx_minus.tocsr()
	Dx_plus = Dx_plus.tocsr()
	return Dx_minus, Dx_plus


def get_angular_grids(nx2, nx3):
	"""
	x2 / x3 grids: uniform spacing and periodic boundary conditions
	The x3 grid has an offset to ensure cos(x2 - x3) != 0.
	"""
	x2 = np.linspace(-np.pi, np.pi, nx2, endpoint = False)
	x3_min, x3_max = 0 + 0.125 * (2 * np.pi / nx3), 2 * np.pi + 0.125 * (2 * np.pi / nx3)
	x3 = np.linspace(x3_min, x3_max, nx3, endpoint = False)
	return x2, x3


def upwind_operator(Dx_minus, Dx_plus, Dx_coeff):
	"""
	Upwind finite difference operator.

	Args:
		Dx_minus (scipy.sparse matrix): backward (minus) finite difference operator.
		Dx_plus (scipy.sparse matrix): forward  (plus) finite difference operator.
		Dx_coeff (numpy.ndarray): coefficient array

	Returns:
		scipy.sparse matrix: The upwind operator
	"""
	mask_x = Dx_coeff <= 0
	Dx_masked_minus = sp.diags(mask_x.ravel().astype(int)) @ Dx_minus
	Dx_masked_plus = sp.diags((~mask_x).ravel().astype(int)) @ Dx_plus
	Dx_upwind = Dx_masked_minus + Dx_masked_plus
	return Dx_upwind


def test_ill_conditioned(Dx_coeff):
	"""
	Test for ill-conditioning. The thresholds are heuristic only.
	"""
	ill_conditioning_test = np.min(np.abs(Dx_coeff.ravel()))
	if ill_conditioning_test < 1e-10:
		raise ValueError(f'System is ill-conditioned; min |Dx1_coeff| = {ill_conditioning_test}')
	elif ill_conditioning_test < 1e-6:
		warnings.warn(f'System may be ill-conditioned; min |Dx1_coeff| = {ill_conditioning_test}')
