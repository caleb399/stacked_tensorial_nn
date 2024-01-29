import unittest
import importlib


class TestDependencies(unittest.TestCase):
	def test_required_modules_installed(self):
		required_modules = [
			'numpy',
			'tensorflow',
			'numpy',
			't3f',
			'scipy',
			'h5py',
			'matplotlib',
			'pydot',
			'openvino',
		]

		for module in required_modules:
			with self.subTest(module = module):
				importlib.import_module(module)


if __name__ == '__main__':
	unittest.main()
