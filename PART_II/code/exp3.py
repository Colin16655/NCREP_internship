## PLOT the Alfredo data
from utils import save_figure
from alfredo import Alfredo
import numpy as np
from tqdm import tqdm
import sys

# Add the directory containing the package to sys.path
package_path = '/home/boubou/Documents/NCREP_internship/PART_I'
sys.path.append(package_path)

from helper.visualizer import Visualizer

# Set rcParams to customize plot appearance
import matplotlib.pyplot as plt
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.grid'] = True
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['legend.loc'] = 'upper right'
plt.rcParams['axes.titlesize'] = 11
plt.rcParams["figure.autolayout"] = True

paths = ["data/Alfredo/M_0_0_0_ambient/subset_signal.txt", 
         "data/Alfredo/M_0_0_1_ambient/subset_signal.txt", 
         "data/Alfredo/M_0_0_2_ambient/subset_signal.txt", 
         "data/Alfredo/M_0_1_1_ambient/subset_signal.txt", 
         "data/Alfredo/M_0_1_2_ambient/subset_signal.txt", 
         "data/Alfredo/M_1_0_0_ambient/subset_signal.txt", 
         "data/Alfredo/M_1_0_1_ambient/subset_signal.txt", 
         "data/Alfredo/M_1_0_2_ambient/subset_signal.txt", 
         "data/Alfredo/M_1_0_3_ambient/subset_signal.txt", 
         "data/Alfredo/M_1_1_2_ambient/subset_signal.txt", 
         "data/Alfredo/M_1_0_0_ambient_2m/subset_signal.txt", 
         "data/Alfredo/M_1_0_1_ambient_2m/subset_signal.txt", 
         "data/Alfredo/M_1_0_2_ambient_2m/subset_signal.txt",
         "data/Alfredo/M_2_0_0/subset_signal.txt",
         "data/Alfredo/M_2_0_1/subset_signal.txt",
         "data/Alfredo/M_2_0_2/subset_signal.txt",
         "data/Alfredo/M_2_0_3/subset_signal.txt", 
         "data/Alfredo/M_2_0_4/subset_signal.txt"]

names = ["M_0_0_0", "M_0_0_1", "M_0_0_2", "M_0_1_1", "M_0_1_2", "M_1_0_0", "M_1_0_1", "M_1_0_2", "M_1_0_3", "M_1_1_2", "M_1_0_0_2m", "M_1_0_1_2m", "M_1_0_2_2m", "M_2_0_0", "M_2_0_1", "M_2_0_2", "M_2_0_3", "M_2_0_4"]
labels = ["$x_2$", "$y_2$", "$x_1$", "$y_1$"]

for i, path in enumerate(paths):
    alfredo = Alfredo([path], S=5, k=2)
    alfredo.get_data()
    data = alfredo.merged_data
    time = np.arange(0, len(data)) / alfredo.fs
    fig, ax = plt.subplots(1, 1, figsize=(5, 2))
    for j in range(data.shape[1]):
        ax.plot(time, data[:, j], label=labels[j], linewidth=0.5)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Acceleration")
    ax.set_xlim([time[0], time[-1]])

    ax.legend(loc="upper center", ncol=len(ax.lines))
    save_figure(fig, f"{names[i]}", "exp3_Alfredo_time_domain", r'PART_II/results')

    # Frequency analysis
    # alfredo.apply_freq_analysis(folder_name="exp3_Alfredo_fqcy_domain", filename1='PSD_SVD'+names[i], filename2='PP_indices'+names[i], name=names[i])

