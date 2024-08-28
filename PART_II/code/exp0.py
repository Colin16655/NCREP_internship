from process_folder import ProcessFolder
import numpy as np
from utils import save_figure

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

S = 5                            # Number of SDOs per time window
Ls = [6000*i for i in range(11, 22)]                            # * 0.01 seconds, so 1 to 11 minutes
selected_indices = [3, 4, 5, 6]  # Indices of the selected sensors : stair
### USER ###

fig, ax = plt.subplots(len(Ls), 1, figsize=(10, 1.2*len(Ls)), sharex=True)
labels = [0.01*L for L in Ls]

scaling_factors = np.array([0.4035*1000, 0.4023*1000, 0.4023*1000, 0.4023*1000, 0.4015*1000, 0.4014*1000, 0.4007*1000, 0.4016*1000])[selected_indices]
for i, L in enumerate(Ls):
    folder_processor = ProcessFolder(S, L, selected_indices, folder_path, scaling_factors, location)

    DI_values = folder_processor.DI_values
    time = np.linspace(0, 600*len(folder_processor.loader.file_paths), len(folder_processor.loader))
    # Separate the indices and values for positive and negative DI values
    indices = np.arange(len(DI_values))
    positive_indices = indices[DI_values > 0]
    negative_indices = indices[DI_values <= 0]

    positive_values = DI_values[DI_values > 0]
    negative_values = DI_values[DI_values <= 0]

    # Generate the corresponding time values for positive and negative indices
    positive_time = time[positive_indices]
    negative_time = time[negative_indices]

    if len(positive_time)   > 0 : ax[i].stem(positive_time, positive_values, linefmt='r-', markerfmt='ro', basefmt=" ")
    if len(negative_values) > 0 : ax[i].stem(negative_time, negative_values, linefmt='g-', markerfmt='go', basefmt=" ")

    ax[i].set_ylabel('DI')
        
    ax[i].set_title(f'L = {labels[i]} [s]')

ax[-1].set_xlabel('Time [s]')
fig.tight_layout()
save_figure(fig, f"NI_CB_DI", f"exp0_loc_{location}_S{S}_L_vary_p_{len(selected_indices)}", format='pdf')

