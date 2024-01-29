import unittest
import numpy as np
from stnn.pde.common import d_dx_upwind, d2_dx2_fourth_order, d_dx_upwind_nonuniform


def assert_derivative(operator, function, expected_derivative, boundary = None, rtol = None, atol = None):
	"""
	Assert the derivative of a function using a given finite-difference operator

	Args:
		operator (np.ndarray or sparse matrix): Differential operator matrix.
		function (np.ndarray): Values of the function to differentiate
		expected_derivative (np.ndarray):: Expected result of the derivative.
		boundary (int): Specifies if boundary elements should be excluded, and how many.
		rtol: Relative tolerance.
		atol: Absolute tolerance.
	"""
	observed_derivative = operator @ function
	if boundary is not None:
		observed_derivative = observed_derivative[boundary:-boundary]
		expected_derivative = expected_derivative[boundary:-boundary]

	np.testing.assert_allclose(observed_derivative, expected_derivative, rtol = rtol, atol = atol)


class TestDifferentialOperators(unittest.TestCase):

	def setUp(self):
		# ----- Set up grids

		# grid for radial coordinate (non-periodic, non-uniform spacing)
		self.nx = 100000
		zs = np.linspace(-np.pi / 2, np.pi / 2, self.nx + 2)
		R1, R2 = 0.5, 1.4
		self.z_ = (R2 - R1) * (np.sin(zs) / 2 + 0.5) + R1
		self.z = self.z_[1:-1]

		# grid for angular coordinates (periodic, uniform spacing)
		self.y = np.linspace(0, 2 * np.pi, self.nx, endpoint = False)
		dy = (2 * np.pi) / self.nx

		# ----- Set tolerances

		# Tolerances for "exact" tests, i.e., where the finite differences do not have truncation error
		self.atol = 1e-6
		self.rtol = 1e-6

		# Tolerances for some inexact tests
		self.atol_inexact = 1e-3
		self.rtol_z = 3 * np.max(np.diff(self.z_))  # relative tolerance of 3*dx for first-order one-sided differences
		self.rtol_y = 3 * dy  # relative tolerance of 3*dx for first-order one-sided differences
		self.rtol_y2 = 3 * dy**4  # relative tolerance of 3*dy**4 for fourth-order central differences

		# ----- Test inputs

		# "Exact" test inputs
		self.f1 = -2.3 * np.ones(self.nx)
		self.f2 = 0.8 * self.z
		self.f3 = 0.8 * self.y
		self.f4 = -0.1 * self.y * self.y

		# Inexact test inputs
		# noinspection PyRedundantParentheses
		self.g1 = (self.z)**2 - (self.z)**3
		self.g2 = (np.cos(self.y))**2 - (np.sin(self.y))**3
		self.g3 = (np.sin(2 * self.y))**2 - (np.cos(self.y))**3

	def test_d_dx_upwind(self):
		dx_m, dx_p = d_dx_upwind(self.y, self.nx)

		assert_derivative(dx_m, self.f1, np.zeros(self.nx), rtol = self.rtol, atol = self.atol)
		assert_derivative(dx_p, self.f1, np.zeros(self.nx), rtol = self.rtol, atol = self.atol)

		assert_derivative(dx_m, self.f3, 0.8 * np.ones(self.nx), boundary = 1, rtol = self.rtol, atol = self.atol)
		assert_derivative(dx_p, self.f3, 0.8 * np.ones(self.nx), boundary = 1, rtol = self.rtol, atol = self.atol)

		expected_dg2 = -2 * np.cos(self.y) * np.sin(self.y) - 3 * np.sin(self.y)**2 * np.cos(self.y)
		assert_derivative(dx_m, self.g2, expected_dg2, rtol = self.rtol_y, atol = self.atol_inexact)
		assert_derivative(dx_p, self.g2, expected_dg2, rtol = self.rtol_y, atol = self.atol_inexact)

		expected_dg3 = 4 * np.sin(2 * self.y) * np.cos(2 * self.y) + 3 * np.cos(self.y)**2 * np.sin(self.y)
		assert_derivative(dx_m, self.g3, expected_dg3, rtol = self.rtol_y, atol = self.atol_inexact)
		assert_derivative(dx_p, self.g3, expected_dg3, rtol = self.rtol_y, atol = self.atol_inexact)

	def test_d2_dx2_fourth_order(self):
		d2x = d2_dx2_fourth_order(self.y, self.nx)

		assert_derivative(d2x, self.f1, np.zeros(self.nx), rtol = self.rtol, atol = self.atol)
		assert_derivative(d2x, self.f3, np.zeros(self.nx), boundary = 4, rtol = self.rtol, atol = self.atol)
		assert_derivative(d2x, self.f4, -0.2 * np.ones(self.nx), boundary = 4, rtol = self.rtol, atol = self.atol)

		expected_dg2 = 2 * np.sin(self.y)**2 - 2 * np.cos(self.y)**2 - \
					   6 * np.sin(self.y) * np.cos(self.y)**2 + 3 * np.sin(self.y)**3
		assert_derivative(d2x, self.g2, expected_dg2, rtol = self.rtol_y2, atol = self.atol_inexact)

		expected_dg3 = 8 * np.cos(2 * self.y)**2 - 8 * np.sin(2 * self.y)**2 - \
					   6 * np.cos(self.y) * np.sin(self.y)**2 + 3 * np.cos(self.y)**3
		assert_derivative(d2x, self.g3, expected_dg3, rtol = self.rtol_y2, atol = self.atol_inexact)

	def test_d_dx_upwind_nonuniform(self):
		dx_minus, dx_plus = d_dx_upwind_nonuniform(self.z_, self.nx)

		assert_derivative(dx_minus, self.f1, np.zeros(self.nx), boundary = 1, rtol = self.rtol, atol = self.atol)
		assert_derivative(dx_plus, self.f1, np.zeros(self.nx), boundary = 1, rtol = self.rtol, atol = self.atol)

		assert_derivative(dx_minus, self.f2, 0.8 * np.ones(self.nx), boundary = 1, rtol = self.rtol, atol = self.atol)
		assert_derivative(dx_plus, self.f2, 0.8 * np.ones(self.nx), boundary = 1, rtol = self.rtol, atol = self.atol)

		expected_dg1 = (2 * self.z - 3 * self.z**2)
		assert_derivative(dx_minus, self.g1, expected_dg1, boundary = 1, rtol = self.rtol_z, atol = self.atol_inexact)
		assert_derivative(dx_plus, self.g1, expected_dg1, boundary = 1, rtol = self.rtol_z, atol = self.atol_inexact)


if __name__ == '__main__':
	unittest.main()
