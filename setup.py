from setuptools import setup, find_packages

setup(
	name = 'stacked_tensorial_nn',
	version = '1.1.0',
	packages = find_packages(),
	url = 'https://github.com/caleb399/stacked_tensorial_nn',
	license = 'MIT',
	author = 'caleb399',
	author_email = 'c.g.wagner23@gmail.com',
	description = 'TensorFlow Implementation of the Stacked Tensorial Neural Network (STNN) Architecture',
	install_requires = [
		'tensorflow>=2.15.0',
		'numpy~=1.26.0',
		't3f~=1.2.0',
		'scipy~=1.12.0',
		'h5py~=3.10.0',
		'matplotlib~=3.8.2',
		'pydot~=1.4.2',
		'openvino~=2023.3.0',
		'pyyaml>=6.0.1'
	]
)
