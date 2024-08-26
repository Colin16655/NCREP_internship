# Just ploting the time series (not really interesting)

import numpy as np
from helper.data_loader1 import DataLoader
from helper.visualizer import Visualizer


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
location = "Lello_2023_07_10_WholeDay_stairs"
batch_size = 3
selected_indices = [3, 4, 5, 6]
### USER ###

scaling_factors = np.array([0.4035*1000, 0.4023*1000, 0.4023*1000, 0.4023*1000, 0.4015*1000, 0.4014*1000, 0.4007*1000, 0.4016*1000])[selected_indices]

# Load the data
loader = DataLoader(selected_indices, folder_path=folder_path, batch_size=10, scaling_factors=scaling_factors)
time, data = loader[0]
print("Time shape:", time.shape)
print("Data shape:", data.shape)

# Initialize the Visualizer with time and frequencies
# Create a unique folder name based on hyperparameters
time_window_size = 600 * len(loader.file_paths[0])
folder_name = f"exp0_loc_{location}_wind_{time_window_size}"

visualizer = Visualizer(time, output_dir="PART_I/results")
labels=["Base X", "Base Y", "Base Z", "Stair 1 X", "Stair 1 Y", "Stair 1 Z", "Stair 2 Z", "Vitral Z"]
labels = [labels[i] for i in selected_indices]
visualizer.plot_data(data, "Original_sensor_data", folder_name, y_label="[A]", labels=labels)
visualizer.plot_data(data, "Detrended_scaled_sensor_data", folder_name, y_label="[mG]", labels=labels)
