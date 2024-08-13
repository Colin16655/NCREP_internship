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
folder_path = "data/Lello/LelloNight_Jul23_Apr24" # "data/Lello/Lello_2023_07_10_WholeDay"
selected_indices = [3, 4, 5, 6]  # Indices of the selected sensors : stair
batch_size = 6 # 1 3 6 for time window of 10 30 60 min
location = f"LelloNight_Jul23_Apr24_wind_{600*batch_size}" # f"Lello_2023_07_10_WD_stairs"
ranges_display = [(8,13), (13,18), (18,24)]
pp_args = {'distance0' : 1,
           'distance1' : 1,  
           'distance2' : 1,
           'sigma' : 8,
           'nperseg' : 1024, # 256, 512, 1024, 2048, 4096, 8192
           'ranges_to_check' : [(10.5, 12.5), (15.5, 17.5), (21.5, 23.5)],
           'mac_threshold' : 0.95,
           'n_mem' : 4,
           'n_modes' : len(ranges_display),
        }
methods = [1, 2, 3, 4]  # 0, 1, 2, 3
### USER ###

# Define scaling factors for the sensor data
scaling_factors = np.array([0.4035*1000, 0.4023*1000, 0.4023*1000, 0.4023*1000, 0.4015*1000, 0.4014*1000, 0.4007*1000, 0.4016*1000])[selected_indices]


folder_processor = FolderProcessor(selected_indices, location, folder_path, batch_size, pp_args, scaling_factors, methods=methods, ranges_display=ranges_display)
folder_processor.process()
