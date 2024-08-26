from yellow import Yellow
import numpy as np

# Add the directory containing the package to sys.path
# package_path = '/home/boubou/Documents/NCREP_internship/PART_I/helper'
# sys.path.append(package_path)

from visualizer import Visualizer

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

paths = ["data/Yellow/Ambient/shm01a.mat", 
         "data/Yellow/Ambient/shm02a.mat",
         "data/Yellow/Ambient/shm03a.mat",
         "data/Yellow/Ambient/shm04a.mat",
         "data/Yellow/Ambient/shm05a.mat",
         "data/Yellow/Ambient/shm06a.mat",
         "data/Yellow/Ambient/shm07a.mat",
         "data/Yellow/Ambient/shm08a.mat",
         "data/Yellow/Ambient/shm09a.mat"]

## USER 
S = 5
Ls = [200 * i for i in range(1, 12)] # 200 for 1s
## USER

folder_name = f"exp1_loc_Yellow_S{S}_L_vary_p_{16}"

# Frequency domain analysis to identify the modal frequencies


yellow = Yellow(paths, S)
yellow.get_data()
yellow.plot_acc(folder_name=folder_name)
yellow.apply_stat_analysis(Ls)
yellow.apply_freq_analysis(folder_name=folder_name)
