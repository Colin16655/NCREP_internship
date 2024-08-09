import numpy as np
from helper.data_loader import DataLoader
from helper.processor import ModalFrequencyAnalyzer
from helper.visualizer import Visualizer
from helper.folder_processor import FolderProcessor

# Set rcParams to customize plot appearance
import matplotlib.pyplot as plt
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.grid'] = True
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['legend.loc'] = 'upper right'
plt.rcParams['axes.titlesize'] = 20
plt.rcParams["figure.autolayout"] = True

### USER ###
folder_path = "data/Lello/Lello_2023_07_10_WholeDay"
location = "Lello_2023_07_10_WD_stairs"
selected_indices = [3, 4, 5, 6]  # Indices of the selected sensors : stair
batch_size = 1
n_mem = 4
n_modes = 4
pp_args = {'distance0' : 1,
           'distance1' : 10,  
           'distance2' : 2,
           'sigma' : 14,
           'nperseg' : 1024, # 256, 512, 1024, 2048, 4096, 8192
           'ranges_to_check' : [(10.5, 12.5), (15.5, 17.5), (21.5, 23.5)],
           'mac_threshold' : 0.9
        }
methods = [0, 1, 3]
ranges_display = [(8,13), (13,18), (18,24)]
### USER ###

# Define scaling factors for the sensor data
scaling_factors = np.array([0.4035*1000, 0.4023*1000, 0.4023*1000, 0.4023*1000, 0.4015*1000, 0.4014*1000, 0.4007*1000, 0.4016*1000])[selected_indices]


folder_processor = FolderProcessor(selected_indices, folder_path, batch_size, n_mem, n_modes, pp_args, scaling_factors, methods=methods, ranges_display=ranges_display)
folder_processor.process()
