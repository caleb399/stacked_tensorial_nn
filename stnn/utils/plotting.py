import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.cm import get_cmap


def plot_comparison(system, bf, rho, rho_pred, fontscale = 1, output_filename = 'comparison.png',
					wspace = -0.1, hspace = 0.5):
	"""
	Side-by-side comparison of contour plots for arguments 'rho' and 'rho_pred'.
	Also plots the boundary data (argument 'bf') as a contour plot.

	Args:
		system: PDESystem object
		bf (np.ndarray): 2D array representing the boundary data on the (x2, x3) grid.
		rho (np.ndarray): 2D array, known rho
		rho_pred (np.ndarray): 2D array, predicted rho
		fontscale (int, optional): A scaling factor for the font size in plots.
		output_filename (str, optional): name of the output file
		wspace (float, optional): horizontal spacing between subplots
		hspace (float, optional): vertical spacing between subplots

	Returns:
		Does not return a value. The figure is saved to a file with name given by 'output_filename'. The default
		is 'comparison.png'.
	"""
	# System parameters
	ell = system.params['ell']
	a2 = system.a2
	e = system.params['eccentricity']
	nx1, nx2, nx3 = system.params['nx1'], system.params['nx2'], system.params['nx3']

	# Get x, y grids from 'PDESystem' object
	x, y = system.get_xy_grids()

	# Relative error
	err = np.linalg.norm(rho - rho_pred)
	rel_err = err / np.linalg.norm(rho)

	# wrap around values for continuity
	rho = np.append(rho, rho[:, 0:1], axis = 1)
	rho_pred = np.append(rho_pred, rho_pred[:, 0:1], axis = 1)

	plots = [rho, rho_pred]

	# Color bar limits
	vmin = np.nanmin(rho_pred[:, :])
	vmax = np.nanmax(rho_pred[:, :])
	vmin = min(vmin, np.nanmin(rho[:, :]))
	vmax = max(vmax, np.nanmax(rho[:, :]))

	# Figure layout
	cbar_coords = (0.89, 0.11, 0.03, 0.77)
	fig = plt.figure(figsize = (24, 24))
	gs = fig.add_gridspec(11, 2, hspace = hspace, wspace = wspace)
	axs = [fig.add_subplot(gs[:11, 0]), fig.add_subplot(gs[:5, 1]), fig.add_subplot(gs[6:11, 1])]

	# boundary data contour plot
	bf_plot = np.nan * np.ones((2 * nx2, nx3))
	bf_plot[:nx2, :nx3 // 2] = bf[:nx2, :]
	bf_plot[nx2:, nx3 // 2:] = bf[nx2:, :]
	im = axs[0].imshow(bf_plot)
	axs[0].set_yticks([1, nx2 // 4, nx2 // 2 - 1, nx2 // 2 + 1, 3 * nx2 // 4, nx2])
	axs[0].set_yticklabels(['-pi', '0', 'pi', '', '0', 'pi'])
	axs[0].set_xticks([1, nx3 // 2, nx3])
	axs[0].set_xticklabels(['0', 'pi', '2pi'])
	axs[0].set_title('Boundary data\n\n', fontsize = fontscale * 48, y = 0.91)
	for label in axs[0].get_xticklabels() + axs[0].get_yticklabels():
		label.set_fontsize(fontscale * 48)
	divider = make_axes_locatable(axs[0])
	cax = divider.append_axes('bottom', size = "3%", pad = 1.5)
	cb = fig.colorbar(im, cax = cax, orientation = 'horizontal')
	cb.ax.tick_params(labelsize = fontscale * 48)
	cax.xaxis.set_ticks_position('bottom')

	# rho(x, y) contour plots
	titles = ['Direct Solution', 'Tensor network']
	for i, ax in enumerate(axs[1:]):
		z = plots[i]
		im = ax.contourf(x, y, z, levels = np.linspace(vmin, vmax, 100), cmap = get_cmap('hsv'))
		ax.set_title(titles[i], fontsize = fontscale * 48)
		for label in ax.get_xticklabels() + ax.get_yticklabels():
			label.set_fontsize(fontscale * 32)
		ax.set_aspect(1.0)

	cbar_ax = fig.add_axes(cbar_coords)
	cb = fig.colorbar(im, cax = cbar_ax, pad = 0.05)
	cb.ax.tick_params(labelsize = fontscale * 32)

	suptitle = f'ell = {ell:.3f}; a2 = {a2:.3f}; e = {e:.3f}; Relative error: {rel_err:.3f}'
	plt.suptitle(suptitle, fontsize = fontscale * 48, x = 0.1, y = 0.97, horizontalalignment = 'left')
	plt.savefig(output_filename)
	plt.close()
