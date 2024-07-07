import h5py
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

current_data = datetime.now()
date_string = current_data.strftime("%Y-%m-%d")
path2profiling = '/home/cunchu/zy/file_run/masked_sm4_raw_traces_for_snr_0518.hdf5'
file_profiling = h5py.File(path2profiling, 'r')

samples = 20000
num_traces = 25600
want2attack = "sbox"
#label_profiling = np.array(file_profiling["rkgroup"]["roundkey"][:num_traces, 0])
#label_profiling = np.array(file_profiling["rkgroup"]["sbox"][:num_traces, 0])
label_profiling = np.array(file_profiling["rkgroup"][want2attack][:num_traces, 0])
traces_profiling = np.array(file_profiling["rkgroup"]["traces"][:num_traces, 0:samples], dtype=np.float64)
traces_profiling = np.expand_dims(traces_profiling, 2)



# Create a dictionary to store subsets
subsets = {label: [] for label in range(256)}

# Populate subsets with corresponding traces
for label, trace in zip(label_profiling, traces_profiling):
    subsets[label].append(trace)

# Convert lists to numpy arrays
for label in subsets:
    subsets[label] = np.array(subsets[label])

# Now, subsets[label] contains all traces for a specific label

mean_traces = np.zeros((256, samples))
for label in subsets:
    mean_traces[label] = np.mean(subsets[label], axis=0).flatten()

variance_matrices = np.zeros((256, samples))
for label in subsets:
    variance_matrices[label] = np.var(subsets[label], axis=0).flatten()


# var of the signal
var_signal = np.var(mean_traces, axis=0).flatten()

#mean of the noise

var_noise = np.mean(variance_matrices, axis=0).flatten()
# snr
snr = var_signal / var_noise

plt.plot(snr, label='SNR')
# 添加标签和标题
plt.xlabel('time')
plt.ylabel('SNR')
plt.title(want2attack+'_SNR_Result')
plt.legend()  # 显示图例
# 保存图形为文件（例如PNG）
plt.savefig(date_string+'_SNR_of_'+want2attack+'.png')
# 不显示图形，仅保存为文件
plt.close()
plt.show()