# Just ploting the time series (not really interesting)

import numpy as np
from helper.data_loader import DataLoader, get_files_list
from helper.preprocessor import Preprocessor
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
location = "Lello_Jul23_stairs_WD"
file_paths = get_files_list(folder_path)[:10]
selected_indices = [0, 1, 2, 3, 4, 5, 6, 7]
### USER ###

# Load the data
loader = DataLoader(file_paths)
time, data = loader.load_data()
data = np.array([data[:, i] for i in selected_indices]).T
print("Time shape:", time.shape)
print("Data shape:", data.shape)

# Define scaling factors for the sensor data
scaling_factors = np.array([0.4035*1000, 0.4023*1000, 0.4023*1000, 0.4023*1000, 0.4015*1000, 0.4014*1000, 0.4007*1000, 0.4016*1000])
scaling_factors = np.array([scaling_factors[i] for i in selected_indices])

# Transform the data
transformer = Preprocessor(time, data, scaling_factors)
detrended_data = transformer.detrend_and_scale()

# Initialize the Visualizer with time and frequencies
# Create a unique folder name based on hyperparameters
time_window_size = 600 * len(file_paths)
folder_name = f"loc_{location}_wind_{time_window_size}"

visualizer = Visualizer(time, output_dir="results")
labels=["Base X", "Base Y", "Base Z", "Stair 1 X", "Stair 1 Y", "Stair 1 Z", "Stair 2 Z", "Vitral Z"]
labels = [labels[i] for i in selected_indices]
visualizer.plot_data(data, "Original_sensor_data", folder_name, y_label="[A]", labels=labels)
visualizer.plot_data(detrended_data, "Detrended_scaled_sensor_data", folder_name, y_label="[mG]", labels=labels)
