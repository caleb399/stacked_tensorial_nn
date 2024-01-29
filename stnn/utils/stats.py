import numpy as np


def get_stats(rho, rho_pred, output_filename = 'stats.npz'):
	"""
	Calculates statistical metrics for model predictions and saves them to a file.

	This function computes the normalized loss for each instance in the dataset, identifies the maximum loss and its 
	index, and calculates the average loss. These statistics are then saved to an NPZ file.

	Args:
		rho (numpy.ndarray): True values.
		rho_pred (numpy.ndarray): Predicted values.
		output_filename (str, optional): The name of the file where the statistics will be saved. Defaults to 'stats.npz'.
	"""
	if rho.shape != rho_pred.shape:
		raise ValueError('rho and rho_pred must have the same shape.')

	y_true_flattened = rho.reshape(rho.shape[0], -1)
	y_pred_flattened = rho_pred.reshape(rho_pred.shape[0], -1)
	loss = np.linalg.norm(y_true_flattened - y_pred_flattened, axis = 1) / np.linalg.norm(y_true_flattened, axis = 1)
	max_loss = np.max(loss)
	max_loss_index = np.argmax(loss)
	print(f'Maximum loss: {max_loss}')
	print(f'Index of instance with max loss: {max_loss_index}')
	print(f'Average loss: {np.average(loss)}')
	np.savez(output_filename, max_loss = max_loss, avg_loss = np.average(loss), N = loss.shape[0])
