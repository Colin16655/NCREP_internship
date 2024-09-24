# Applies statistical analysis for different time window lengths (Ls).
# Finally, it applies frequency analysis and generates plots, saving the results to a specified folder.

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

# paths = ["data/Alfredo/M_1_0_0_ambient/subset_signal.txt", 
#          "data/Alfredo/M_1_0_1_ambient/subset_signal.txt", 
#          "data/Alfredo/M_1_0_2_ambient/subset_signal.txt", 
#          "data/Alfredo/M_1_0_3_ambient/subset_signal.txt"]
# name  = "M_1_0"

# paths = ["data/Alfredo/M_1_0_0_ambient/subset_signal.txt", 
#          "data/Alfredo/M_1_0_1_ambient/subset_signal.txt", 
#          "data/Alfredo/M_1_1_2_ambient/subset_signal.txt", 
#          "data/Alfredo/M_1_0_3_ambient/subset_signal.txt"]
# name  = "M_1_1"

# paths = ["data/Alfredo/M_0_0_0_ambient/subset_signal.txt", 
#          "data/Alfredo/M_0_0_1_ambient/subset_signal.txt", 
#          "data/Alfredo/M_0_0_2_ambient/subset_signal.txt"]
# name  = "M_0_0"

# paths = ["data/Alfredo/M_0_0_0_ambient/subset_signal.txt", 
#          "data/Alfredo/M_0_1_1_ambient/subset_signal.txt", 
#          "data/Alfredo/M_0_1_2_ambient/subset_signal.txt"]
# name  = "M_0_1"


# paths = ["data/Alfredo/M_2_0_0/subset_signal.txt",
#          "data/Alfredo/M_2_0_1/subset_signal.txt",
#          "data/Alfredo/M_2_0_2/subset_signal.txt",
#          "data/Alfredo/M_2_0_3/subset_signal.txt", 
#          "data/Alfredo/M_2_0_4/subset_signal.txt"]
# name  = "M_2_0"

# paths = ["data/Alfredo/M_1_0_0_ambient_2m/subset_signal.txt",
#          "data/Alfredo/M_1_0_1_ambient_2m/subset_signal.txt",
#          "data/Alfredo/M_1_0_2_ambient_2m/subset_signal.txt"]
# name  = "M_1_0_2m"

paths = ["data/Alfredo/M_3_0_0/subset_signal.txt",
         "data/Alfredo/M_3_0_1/subset_signal.txt",
         "data/Alfredo/M_3_0_2/subset_signal.txt",
         "data/Alfredo/M_3_0_3/subset_signal.txt",
         "data/Alfredo/M_3_0_4/subset_signal.txt",
         "data/Alfredo/M_3_0_5/subset_signal.txt",
         "data/Alfredo/M_3_0_6/subset_signal.txt"]
name  = "M_3_0"

## USER 
S = 5
k = 2
folder_name = f"exp2_loc_{name}_S_{S}_k_{k}_L_vary_p_{4}"
Ls = [10000, 11000, 12000, 12500, 13000, 13500, 14000, 14500, 15000] # 20 for 1s

ranges_display = [(1e-10,10), (10,25), (25,40)]
pp_args = {'distance0' : 1,
           'distance1' : 1,  
           'distance2' : 1,
           'sigma' : 8,
           'nperseg' : 512, # 256, 512, 1024, 2048, 4096, 8192
        #    'ranges_to_check' : [(10.5, 12.5), (15.5, 17.5), (21.5, 23.5)],
           'mac_threshold' : 0.9,
           'n_mem' : 4,
           'n_modes' : len(ranges_display),
        }
batchsize = 120240
## USER


# Frequency domain analysis to identify the modal frequencies


alfredo = Alfredo(paths, S, k)
alfredo.get_data()
# alfredo.plot_acc(folder_name=folder_name)
alfredo.apply_stat_analysis(Ls, folder_name)

alfredo.apply_freq_analysis_PART_I(pp_args=pp_args, batchsize=batchsize, ranges_display=ranges_display, methods=[2], folder_name=folder_name)