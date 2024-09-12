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

# paths = ["data/Alfredo/M_1_0_0_ambient_2m/subset_signal.txt", 
#          "data/Alfredo/M_1_0_1_ambient_2m/subset_signal.txt", 
#          "data/Alfredo/M_1_0_2_ambient_2m/subset_signal.txt"]
# names = ["M_1_0_0", "M_1_0_1", "M_1_0_2", "M_1_0_3"]
# model_name = "M_1_0_2m"

# paths = ["data/Alfredo/M_1_0_0_ambient/subset_signal.txt", 
#          "data/Alfredo/M_1_0_1_ambient/subset_signal.txt", 
#          "data/Alfredo/M_1_0_2_ambient/subset_signal.txt", 
#          "data/Alfredo/M_1_0_3_ambient/subset_signal.txt"]
# names = ["M_1_0_0", "M_1_0_1", "M_1_0_2", "M_1_0_3"]
# model_name = "M_1_0"

# paths = ["data/Alfredo/M_1_0_0_ambient/subset_signal.txt", 
#          "data/Alfredo/M_1_0_1_ambient/subset_signal.txt", 
#          "data/Alfredo/M_1_1_2_ambient/subset_signal.txt", 
#          "data/Alfredo/M_1_0_3_ambient/subset_signal.txt"]
# names = ["M_1_1_0", "M_1_1_1", "M_1_1_2", "M_1_1_3"]
# model_name = "M_1_1"

# paths = ["data/Alfredo/M_0_0_0_ambient/subset_signal.txt", 
#          "data/Alfredo/M_0_0_1_ambient/subset_signal.txt", 
#          "data/Alfredo/M_0_0_2_ambient/subset_signal.txt"]
# names = ["M_0_0_0", "M_0_0_1", "M_0_0_2"]
# model_name = "M_0_0"

paths = ["data/Alfredo/M_2_0_0_0_bis/subset_signal.txt",
         "data/Alfredo/M_2_0_1_0_bis/subset_signal.txt",
         "data/Alfredo/M_2_0_2_bis/subset_signal.txt",
         "data/Alfredo/M_2_0_3_bis/subset_signal.txt", 
         "data/Alfredo/M_2_0_4_bis/subset_signal.txt"]
names = ["M_2_0_0", "M_2_0_1", "M_2_0_2", "M_2_0_3", "M_2_0_4"]
model_name = "M_2_0"

colors, styles = ["r", "k", "b", "g"], ["--", "-", ":", "-."]
labels = ["$x_2$", "$y_2$", "$x_1$", "$y_1$"]

fig_fft, ax_fft = plt.subplots(len(paths), 1, figsize=(5, 2*len(paths)))
fig_pp, ax_pp = plt.subplots(len(paths), 1, figsize=(5, 2*len(paths)))
fig_psd, ax_psd = plt.subplots(len(paths), 1, figsize=(5, 2*len(paths)))
band = (0.5, 50)
for i, path in enumerate(paths):
    alfredo = Alfredo([path], S=5, k=2)
    alfredo.get_data()
    # ax.legend(loc="upper center", ncol=len(ax.lines))
    # save_figure(fig, f"{names[i]}", "exp3_Alfredo_time_domain")

    alfredo.detect_damages_PART_I(ax_fft=ax_fft[i], ax_psd=ax_psd[i], ax_pp=ax_pp[i], colors=colors, style='-', nperseg=1024, 
                                  folder_name="exp4_Alfredo_fqcy_domain_" + model_name, filename1='PSD_SVD'+names[i], filename2='PP_indices'+names[i], name=names[i], 
                                  band=band)

filenames = ["FFT_ALL", "PP_ALL", "PSD_ALL"]
ylabels = ["FFT - |.|", "PP indices", "PSD"]
for i, (axs, fig) in enumerate(zip([ax_fft, ax_pp, ax_psd], [fig_fft, fig_pp, fig_psd])):
    axs[-1].set_xlabel("F [Hz]")
    axs[0].legend()
    for j, ax in enumerate(axs):
        ax.set_title(f"{names[j]}")
        ax.set_ylabel(ylabels[i])
        ax.set_xlim(band[0], band[1])
        # ax.set_ylim(1e-3, 13)
    fig.tight_layout()
    save_figure(fig, file_name=filenames[i], folder_name="exp4_Alfredo_fqcy_domain_" + model_name, output_dir="PART_II/results")