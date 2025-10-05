import torch
import numpy as np
import h5py
# data = np.load('test.npz')
#
# data =data['test_x'][...,:4]
#
# print(data.shape)
#
# BJ_data=[]
#
#
#
# for i in range(data.shape[0]):
#     print(data[ i][0].shape)
#     BJ_data.append(data[i][0])
#
# print(data[-1][1:].shape)
#
# BJ_data =np.stack(BJ_data, axis=0)
#
# print(BJ_data.shape)
#
# data=np.concatenate([BJ_data,data[-1][1:]],axis=0)
#
# with h5py.File('BJ.h5', 'w') as f:
#     f.create_dataset('data', data=data)
#
# h = h5py.File('BJ.h5')
# data_seq = h["data"][:]
# print(data_seq.shape)

grpah =np.load("geo_adj.npy")
print(grpah.shape)

