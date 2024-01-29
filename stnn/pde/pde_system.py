import numpy as np

from stnn.data.function_generators import generate_random_functions
from .circle import get_system_circle, get_boundary_quantities_circle, u_dot_thetahat
from .ellipse import (get_system_ellipse, get_boundary_quantities_ellipse, u_dot_etavec)


class PDESystem:
    """
    Constructs the PDE system given input parameters. The finite-difference matrices, grids, and other relevant
    quantities are available as attributes.

    Constructor Args:
        params (dict): Configuration dictionary containing the parameters that define the PDE system.

    Attributes:

        ib_slice (numpy.ndarray): Boolean array defining nodes adjacent to the inner boundary
        ob_slice (numpy.ndarray): Boolean array defining nodes adjacent to the outer boundary

        x2_ib (numpy.ndarray): The x2 coordinate values at the inner boundary.
        x2_ob (numpy.ndarray): The x2 coordinate values at the outer boundary.
        x3_ib (numpy.ndarray): The x3 coordinate values at the inner boundary.
        x3_ob (numpy.ndarray): The x3 coordinate values at the outer boundary.

        Dx1_coeff (numpy.ndarray): Coefficients for the advection operator in the radial direction. Used for converting
                                   boundary conditions to the r.h.s. of the linear system defining a
                                   boundary-value problem.

        dx1a (numpy.ndarray): Grid spacing adjacent to the inner boundary
        dx1b (numpy.ndarray): Grid spacing adjacent to the outer boundary

        L (numpy.ndarray): Finite-difference representation of the linear operator defining the PDE.

        x1 (numpy.ndarray): The grid values in radial coordinate (r or mu)
        x2 (numpy.ndarray): The grid values in the angular coordinate (theta or eta).
        x3 (numpy.ndarray): The grid values in the w coordinate

        a1 (float): Minor axis of the inner boundary
        a2 (float): Minor axis of the outer boundary
        b1 (float): Major axis of the inner boundary
        b2 (float): Major axis of the outer boundary

        _coords (str): The type of coordinate system used ('ellipse' or 'circle'). This affects how the grids and
                      other geometric properties are calculated.

        params (dict): The configuration dictionary containing the PDE parameters.
    """
    def __init__(self, params):
        self._required_keys = ['nx1', 'nx2', 'nx3', 'ell', 'a2', 'eccentricity']
        self._optional_keys = []
        self.ib_slice = None
        self.ob_slice = None
        self.x2_ib = None
        self.x2_ob = None
        self.x3_ib = None
        self.x3_ob = None
        self.Dx1_coeff = None
        self.dx1b = None
        self.dx1a = None
        self.L = None
        self.x1 = None
        self.x2 = None
        self.x3 = None
        self.a1 = None
        self.a2 = None
        self.b1 = None
        self.b2 = None
        self._coords = None
        self.params = params
        self.initialize()

    def initialize(self):
        """
        Constructs the system matrices and vectors based on the stored configuration.

        Depending on the 'eccentricity' parameter in `self.params`, the coordinate system is set
        to either 'circle' or 'ellipse'; the domain parametrization and finite-difference grid are defined
        accordingly.
        """
        params = self.params
        missing_keys = [key for key in self._required_keys if key not in params]
        if missing_keys:
            raise KeyError(f"Missing keys in config: {', '.join(missing_keys)}")

        # The functions 'get_system_circle' and 'get_system_ellipse' have a fair amount
        # of overlap and probably should be combined, but for now they are kept separate
        # for simplicity and readability.
        if params['eccentricity'] < 1e-7:
            self._coords = 'circle'
            L, x1, x2, x3, dx1a, dx1b, Dx1_coeff = get_system_circle(params)
            x2_ib, x2_ob, x3_ib, x3_ob, ib_slice, ob_slice = get_boundary_quantities_circle(x2, x3)
            self.b2 = params['a2']
        else:
            self._coords = 'ellipse'
            (L, x1, x2, x3, dx1a, dx1b, Dx1_coeff, major_axis_outer) = get_system_ellipse(params)
            x2_ib, x2_ob, x3_ib, x3_ob, ib_slice, ob_slice = get_boundary_quantities_ellipse(x1, x2, x3)
            self.b2 = major_axis_outer

        self.a1 = 1.0 - params['eccentricity']
        self.a2 = params['a2']
        self.b1 = 1.0

        self.x1 = x1
        self.x2 = x2
        self.x3 = x3

        self.L = L

        self.dx1a = dx1a
        self.dx1b = dx1b
        self.Dx1_coeff = Dx1_coeff

        self.x2_ib = x2_ib
        self.x2_ob = x2_ob
        self.x3_ib = x3_ib
        self.x3_ob = x3_ob
        self.ib_slice = ib_slice
        self.ob_slice = ob_slice

    def generate_random_bc(self, func_gen_id):
        """
        Generates random boundary conditions for the PDE system.

        Args:
            func_gen_id (int): Integer representing the type of 'function generator' used to construct the
            boundary conditions.

        Returns:
            tuple: A tuple containing:
                - ibf_data: Inner boundary data
                - obf_data: Outer boundary data
                - b: 3D array for passing to the GMRES solver. 'b' contains the boundary data but is defined
                     on the full 3D grid.
                - bf: Flattened boundary data before it is permuted

        The boundary conditions are defined on the inner and outer boundaries of the domain and are denoted
        by 'ibf_data' and 'obf_data'. The function passes 'ibf_data' and 'obf_data' through 'convert_boundary_data',
        which converts them into formats suitable for passing into the GMRES solver and STNN model (e.g., by
        reshaping and permutation operations).
        """

        # Note the change of variable (x2, x3) -> (x2, x2 - x3).
        ibf_data = generate_random_functions(1, self.x2_ib, self.x2_ib - self.x3_ib,
											 max_freq=self.params['nx3'], func_gen_id = func_gen_id)[0, self.ib_slice]
        obf_data = generate_random_functions(1, self.x2_ob, self.x2_ob - self.x3_ob,
											 max_freq=self.params['nx3'], func_gen_id = func_gen_id)[0, self.ob_slice]

        # Combine boundary data in single vector
        bf = np.concatenate([ibf_data, obf_data], axis = -1).flatten()

        # Permutes 'ibf_data' and 'obf_data' and construct 'b'
        ibf_data, obf_data, b = self.convert_boundary_data(ibf_data, obf_data)

        return ibf_data, obf_data, b, bf

    def convert_boundary_data(self, ibf_data, obf_data):
        """
        Converts boundary data into formats suitable for passing into the GMRES solver and STNN model.

        Args:
            ibf_data: Inner boundary data
            obf_data: Outer boundary data

        Returns:
            tuple: A tuple containing:
                - ibf_data: Inner boundary data, permuted to match the input structure of the EinsumTTL layer.
                - obf_data: Outer boundary data, permuted to match the input structure of the EinsumTTL layer
                - b: 3D array for passing to the GMRES solver. 'b' contains the boundary data but is defined
                     on the full 3D grid.
        """
        nx1, nx2, nx3 = self.params['nx1'], self.params['nx2'], self.params['nx3']
        b = np.zeros((nx1, nx2, nx3), dtype=np.float64)
        b[0, self.ib_slice] = self.Dx1_coeff[0, self.ib_slice] * (ibf_data / self.dx1a)
        b[-1, self.ob_slice] = -self.Dx1_coeff[-1, self.ob_slice] * (obf_data / self.dx1b)

        if self._coords == 'ellipse':
            sin_angle = u_dot_etavec(self.x1, self.x2, self.x3)
        elif self._coords == 'circle':
            sin_angle = u_dot_thetahat(self.x2, self.x3)
        else:
            raise ValueError(f'"_coords" attribute should be either "ellipse" or "circle"; instead received {self._coords}')

        # reshape and permute 'ibf_data' and 'obf_data'
        sin_angle_i = sin_angle[0, self.ib_slice].reshape(nx2, nx3 // 2)
        sin_angle_o = sin_angle[-1, self.ob_slice].reshape(nx2, nx3 // 2)
        W_I = np.argsort(sin_angle_i, axis=1)
        W_O = np.argsort(sin_angle_o, axis=1)
        ibf_data = ibf_data.reshape((nx2, nx3 // 2))
        obf_data = obf_data.reshape((nx2, nx3 // 2))
        for n in range(nx2):
            ibf_data[n, :] = ibf_data[n, W_I[n, :]]
            obf_data[n, :] = obf_data[n, W_O[n, :]]

        return ibf_data, obf_data, b

    def get_xy_grids(self):
        """
        Converts the native grids of the PDE system to xy coordinates (no interpolation).

        The function also applies a wrap-around in the x2 domain for plotting purposes, ensuring
        continuity across the (periodic) domain.

        Returns:
            tuple of numpy.ndarray: A tuple containing two 2D numpy arrays:
                - x_grid: The x-coordinates grid.
                - y_grid: The y-coordinates grid.
        """
        x1_1D = self.x1[:, 0, 0]
        x2_1D = self.x2[0, :, 0]
        x2_1D = np.append(x2_1D, np.array([np.pi - 1e-3]))  # wrap around for plotting
        x1_2D, x2_2D = np.meshgrid(x1_1D, x2_1D, indexing='ij')
        if self._coords == 'ellipse':
            focal_distance = np.sqrt(self.b1**2 - self.a1**2)
            x_grid = focal_distance * np.sinh(x1_2D) * np.cos(x2_2D)
            y_grid = focal_distance * np.cosh(x1_2D) * np.sin(x2_2D)
        elif self._coords == 'circle':
            x_grid = x1_2D * np.cos(x2_2D)
            y_grid = x1_2D * np.sin(x2_2D)
        else:
            raise ValueError(f'"_coords" should be either "ellipse" or "circle"; instead received {self._coords}')

        return x_grid, y_grid
