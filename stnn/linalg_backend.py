"""
Module imports and wrapper functions of the linear algebra backend.

If  __usecupy__ is "True" and cupy is successfully imported, then

	xp 		--> cupy
	spx		--> cupyx.scipy.sparse.linalg
	
Otherwise,

	xp 		--> numpy
	spx		--> scipy.sparse

For example, if __usecupy__ is False, then

				import numpy as np
				import scipy.sparse.linalg as sp
				
is equivalent to

				from stnn.linalg_backend import xp, spx
"""
__usecupy__ = True

try:
	# If CuPy is not preferred or available, fall back to NumPy
	if not __usecupy__:
		raise ImportError
	import cupy as cp
	import cupyx.scipy.sparse.linalg
	import cupyx.scipy.sparse as cupy_sparse

	xp = cp
	spx = cupy_sparse
	using_cupy = True
except ImportError:
	import numpy as np
	import scipy.sparse.linalg
	import scipy.sparse as scipy_sparse

	xp = np
	spx = scipy_sparse
	using_cupy = False


def csr_matrix(L):
	"""
	Create a CSR (Compressed Sparse Row) matrix.

	If CuPy is available and enabled, this function will create a CuPy CSR matrix.
	Otherwise, it converts the given data to a SciPy CSR matrix.

	Parameters:
	L (array_like or sparse matrix): 2-D array or sparse matrix to convert.

	Returns:
	CSR matrix: The converted CSR matrix, using either CuPy or SciPy.
	"""
	if using_cupy:
		return spx.csr_matrix(L, dtype=xp.float64)
	return L.tocsr()


def asnumpy(arr):
	"""
	Convert an array from the backend library (CuPy or NumPy) to NumPy.
	If NumPy is enabled, the input array is returned unchanged.
	"""
	if using_cupy:
		return cp.asnumpy(arr)
	return arr


def asarray(arr):
	"""
	Convert the input to an array of the backend library (CuPy or NumPy).
	If NumPy is enabled, the input array is returned unchanged.
	"""
	if using_cupy:
		return cp.asarray(arr, dtype=cp.float64)
	return arr
