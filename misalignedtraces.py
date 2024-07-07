import numpy as np
import h5py
from tqdm import tqdm

#######################################################
####                  Parameter                    ####
#######################################################

path2misaligned = '/home/cunchu/zy/file_run/masked_sm4_raw_traces_for_snr_0518.hdf5'
var_misaligned = 4
var_level = str(var_misaligned)
path2getmisaligned = 'masked_sm4_rawtraces_desyn'+var_level+'.hdf5'

randomseed = 42

######################################################

def shift_array(arr, units):
    shifted_arr = np.roll(arr, units)
    if units > 0:
        shifted_arr[:units] = 0
    elif units < 0:
        shifted_arr[:-units] = 0
    return shifted_arr



f = h5py.File(path2misaligned, 'r')
traces = np.array(f["rkgroup"]["traces"][:, :], dtype=np.float64)
label = np.array(f["rkgroup"]["sbox"][:, 0])
#traces = np.array(file_traces["rkgroup"]["traces"][:, :], dtype=np.float64)
#rkgroup = f["rkgroup"]
print("Success Loading Random Traces of Key Group!")
#traces_num = traces.shape[0]
#np.random.seed(randomseed)

#for i in range(traces_num):
#    f[i, :] = shift_array(traces[i, :], shift_stride[i])
f.close()
traces_num_random = traces.shape[0]
sample_num = traces.shape[1]




shift_stride = np.random.randint(0, var_misaligned, size=(traces_num_random))
#shift_stride = np.abs(np.random.normal(0, var_misaligned, (traces_num_random)))

file_get = h5py.File(path2getmisaligned, "w")

rkgroup_desyn = file_get.create_group("rkgroup")

rk_key = rkgroup_desyn.create_dataset("sbox", (traces_num_random, 1), dtype='i8')
#rk_key = label
rk_traces = rkgroup_desyn.create_dataset("traces", (traces_num_random, sample_num), dtype='f')
#rk_traces = traces
for i in tqdm(range(traces_num_random)):
    rk_traces[i, :] = shift_array(traces[i], shift_stride[i])
    rk_key[i, :] = label[i]
 
file_get.close()


