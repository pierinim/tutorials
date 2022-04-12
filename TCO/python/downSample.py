import h5py
import sys
import numpy as np
import tensorflow as tf

f = h5py.File(sys.argv[1], "r")
ECAL = f.get("ECAL")
HCAL = f.get("HCAL")
energy = f.get("energy")

print(ECAL.shape)
print(HCAL.shape)

HCAL = HCAL[:,2:-2,2:-2,:]
ECAL = ECAL[:,1:-1,1:-1,:]

print(ECAL.shape)
print(HCAL.shape)

HCAL = np.sum(HCAL, axis=-1)
ECAL = np.sum(ECAL, axis=-1)

print(ECAL.shape)
print(HCAL.shape)

ECAL = np.reshape(ECAL, (ECAL.shape[0], 49, 49, 1))

x = ECAL
ecal_pool_2d = tf.keras.layers.AveragePooling2D(pool_size=(7, 7), strides=(7, 7), padding='valid')
x = ecal_pool_2d(x)
x = 49.*x

ECAL = np.array(x)
ECAL = np.reshape(ECAL, (ECAL.shape[0], 7, 7))
print(ECAL.shape)
print(HCAL.shape)

f = h5py.File(sys.argv[1].replace(".h5", "_DownSampled.h5"), "w")
f.create_dataset('ECAL', data=ECAL, compression='gzip')
f.create_dataset('HCAL', data=HCAL, compression='gzip')
f.create_dataset('energy', data=energy, compression='gzip')
f.close()

