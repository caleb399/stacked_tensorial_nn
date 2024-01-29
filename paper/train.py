import os
from datetime import datetime
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger

from stnn.nn.stnn import build_stnn
from stnn.data.preprocessing import load_training_data

"""
This script performs the first pass of training (200 epochs, regularization set to True,
and default weights initialization.

Running the script requires the training data, which can be downloaded from TBD (to be determined).
"""

if True:
	raise NotImplementedError('This script is designed to operate on the training datasets from the paper, ' 
							  'which have not yet been included with the repository.')


data_files = []

# training data gets loaded here

ell_min = 0.01
ell_max = 100.0
a2_min = 2.0
a2_max = 20.0

learning_rate = 0.005
weights_file = None # First training pass, no existing weights for initialization
epochs = 200
batch_size = 128
nx1, nx2, nx3 = 256, 64, 32
K = 20 # Number of tensor networks

# STNN model config, matches trial 5 in the paper
stnn_config = {
	'K': K,
	'use_regularization' : True,
	'regularization_strength' : 0.002,
	'ranks' : [1, 16, 16, 16, 16, 16, 7, 1],
	'shape1' : [4, 4, 4, 4, 4, 4, 4],
	'shape2' : [4, 2, 2, 2, 2, 2, 2],
	'nx1' : nx1,
	'nx2' : nx2,
	'nx3' : nx3,
	'W' : 2,
	'd' : 8
}

params_train, bf_train, rho_train, params_test, bf_test, rho_test \
	= load_training_data(data_files, nx2, nx3, ell_min, ell_max, a2_min, a2_max)

model = build_stnn(stnn_config)
if not weights_file is None:
	model.load_weights(weights_file, skip_mismatch = True, by_name = True)

model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.summary()

reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 20, verbose = 1, min_lr = 1e-7)
early_stop = EarlyStopping(monitor = 'val_loss', patience = 100, verbose = 1, restore_best_weights = True)
checkpoint = ModelCheckpoint('model_weights.{epoch:02d}.hdf5', monitor = 'val_loss', verbose = 1,
							 save_freq = 'epoch', period = 10, save_weights_only = True)
now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
csv_logger = CSVLogger(f'training_log_{now}.csv')
model.fit([params_train, bf_train], rho_train, epochs = epochs,
		  validation_data = ([params_test, bf_test], rho_test),
		  batch_size = batch_size, shuffle = True,
		  callbacks = [reduce_lr, checkpoint, csv_logger])

print('Accuracy after training: ', model.evaluate([params_test, bf_test], rho_test))

