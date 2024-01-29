import numpy as np

"""
Interface to a subset of the test functions listed at
https://en.wikipedia.org/wiki/Test_functions_for_optimization
"""


def rastrigin(x, y):
	args = (x, y)
	A = 10
	return A * len(x) + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in args])


def ackley(x, y):
	return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2))) - \
		np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + np.e + 20


def sphere(x, y):
	return x**2 + y**2


def rosenbrock(x, y):
	return 100 * (y - x**2)**2 + (1 - x)**2


def beale(x, y):
	return (1.5 - x + x * y)**2 + (2.25 - x + x * y**2)**2 + (2.625 - x + x * y**3)**2


def goldstein_price(x, y):
	return (1 + (x + y + 1)**2 * (19 - 14 * x + 3 * x**2 - 14 * y + 6 * x * y + 3 * y**2)) * \
		(30 + (2 * x - 3 * y)**2 * (18 - 32 * x + 12 * x**2 + 48 * y - 36 * x * y + 27 * y**2))


def booth(x, y):
	return (x + 2 * y - 7)**2 + (2 * x + y - 5)**2


def bukin(x, y):
	return 100 * np.sqrt(abs(y - 0.01 * x**2)) + 0.01 * abs(x + 10)


def matyas(x, y):
	return 0.26 * (x**2 + y**2) - 0.48 * x * y


def levi(x, y):
	return np.sin(3 * np.pi * x)**2 + (x - 1)**2 * (1 + np.sin(3 * np.pi * y)**2) + \
		(y - 1)**2 * (1 + np.sin(2 * np.pi * y)**2)


def himmelblau(x, y):
	return (x**2 + y - 11)**2 + (x + y**2 - 7)**2


def three_hump_camel(x, y):
	return 2 * x**2 - 1.05 * x**4 + x**6 / 6 + x * y + y**2


def easom(x, y):
	return -np.cos(x) * np.cos(y) * np.exp(-((x - np.pi)**2 + (y - np.pi)**2))


def cross_in_tray(x, y):
	return -0.0001 * (abs(np.sin(x) * np.sin(y) * np.exp(abs(100 - np.sqrt(x**2 + y**2) / np.pi))) + 1)**0.1


def eggholder(x, y):
	return -(y + 47) * np.sin(np.sqrt(abs(x / 2 + (y + 47)))) - x * np.sin(np.sqrt(abs(x - (y + 47))))


def holder_table(x, y):
	return -abs(np.sin(x) * np.cos(y) * np.exp(abs(1 - np.sqrt(x**2 + y**2) / np.pi)))


def mccormick(x, y):
	return np.sin(x + y) + (x - y)**2 - 1.5 * x + 2.5 * y + 1


def schaffer2(x, y):
	return 0.5 + (np.sin(x**2 - y**2)**2 - 0.5) / (1 + 0.001 * (x**2 + y**2))**2


def schaffer4(x, y):
	return 0.5 + (np.cos(np.sin(abs(x**2 - y**2)))**2 - 0.5) / (1 + 0.001 * (x**2 + y**2))**2


def styblinski_tang(x, y):
	args = (x, y)
	return sum([xi**4 - 16 * xi**2 + 5 * xi for xi in args]) / 2


functions = [
	rastrigin,
	ackley,
	sphere,
	rosenbrock,
	beale,
	goldstein_price,
	booth,
	bukin,
	matyas,
	levi,
	himmelblau,
	three_hump_camel,
	easom,
	cross_in_tray,
	eggholder,
	holder_table,
	mccormick,
	schaffer2,
	schaffer4,
	styblinski_tang
]

function_names = [
	'rastrigin',
	'ackley',
	'sphere',
	'rosenbrock',
	'beale',
	'goldstein_price',
	'booth',
	'bukin',
	'matyas',
	'levi',
	'himmelblau',
	'three_hump_camel',
	'easom',
	'cross_in_tray',
	'eggholder',
	'holder_table',
	'mccormick',
	'schaffer2',
	'schaffer4',
	'styblinski_tang'
]

domains = {
	'rastrigin': (-5.12, 5.12),
	'ackley': (-5, 5),
	'sphere': (-1, 1),
	'rosenbrock': {'x': (-2, 2), 'y': (-10, 10)},
	'beale': (-4.5, 4.5),
	'goldstein_price': (-2, 2),
	'booth': (-10, 10),
	'bukin': {'x': (-15, -5), 'y': (-3, 3)},
	'matyas': (-10, 10),
	'levi': (-10, 10),
	'himmelblau': (-5, 5),
	'three_hump_camel': (-5, 5),
	'easom': (-100, 100),
	'cross_in_tray': (-10, 10),
	'eggholder': (-512, 512),
	'holder_table': (-10, 10),
	'mccormick': {'x': (-1.5, 4), 'y': (-3, 4)},
	'schaffer2': (-100, 100),
	'schaffer4': (-100, 100),
	'styblinski_tang': (-5, 5)
}


def scale_input(x, domain):
	min_d, max_d = domain
	return min_d + (max_d - min_d) * x


def get_test_function(X, Y, fun_idx):
	"""
	Evaluates a function on inputs (X, Y).

	Args:
		X (float or array-like): The X input values to be scaled and used in the function.
		Y (float or array-like): The Y input values to be scaled and used in the function.
								Ignored if the function takes only one argument.
		fun_idx (int): The index of the function to be retrieved from a predefined list 'functions'.

	Returns: 
		(float or array-like), values of the function on the grid
	"""
	func = functions[fun_idx]

	domain = domains[func.__name__]
	if isinstance(domain, dict):
		x_scaled = scale_input(X, domain['x'])
		y_scaled = scale_input(Y, domain['y'])
	else:
		x_scaled = scale_input(X, domain)
		y_scaled = scale_input(Y, domain)

	try:
		output = func(X, Y)
	except TypeError:
		output = func(X)
	return output
