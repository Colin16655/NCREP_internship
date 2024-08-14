from process_folder import ProcessFolder
import numpy as np

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
# folder_path, location = "data/Lello/Jul23", "Lello_Jul23_stairs"
# folder_path, location = "data/Lello/Jul24", "Lello_Jul24_stairs"
folder_path, location = "data/Lello/Lello_2023_07_10_WholeDay", "Lello_2023_07_10_WholeDay_stairs"
# folder_path, location = "data/Lello/LelloNight_Jul23_Apr24", "LelloNight_Jul23_Apr24_stairs"

S = 6                            # Number of SDOs per time window
batch_size = 6                   # 1 for 10 min, 2 for 20 min, 3 for 30 min, 6 for 60 min time window
L = int(batch_size / S)          # length of the SDO; 1 for 10 min, 2 for 20 min, 3 for 30 min, 6 for 60 min time window
selected_indices = [3, 4, 5, 6]  # Indices of the selected sensors : stair
### USER ###

scaling_factors = np.array([0.4035*1000, 0.4023*1000, 0.4023*1000, 0.4023*1000, 0.4015*1000, 0.4014*1000, 0.4007*1000, 0.4016*1000])[selected_indices]

folder_processor = ProcessFolder(S, L, selected_indices, folder_path, batch_size, scaling_factors, location)
folder_processor.plot()
