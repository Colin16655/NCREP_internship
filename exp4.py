import numpy as np
from helper.data_loader import DataLoader
from helper.preprocessor import Preprocessor
from helper.processor import FrequencyAnalyzer
from helper.visualizer import Visualizer
from scipy.ndimage import gaussian_filter1d
import os
from helper.process import get_vibration_frequencies, get_vibration_frequencies_pyoma, get_vibration_frequencies_best

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

def get_files_list(folder_path):
    """
    Retrieve a list of all files in a specified folder, sorted in chronological order based on filenames.

    This function lists all files in the given folder and sorts them lexicographically by filename. 
    It then returns the full file paths for each file in the sorted order.

    Parameters:
        folder_path (str): The path to the folder containing the files.

    Returns:
        list of str: A list of full file paths, sorted by filename.
    """
    # List all files in the folder
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    # Sort files by filename (assumes filenames can be sorted lexicographically)
    files.sort()
    # Create full file paths
    sorted_file_paths = [os.path.join(folder_path, file_name) for file_name in files]
    return sorted_file_paths

def plot_frequencies_best(file_paths, location, time_window_factor, n_modes=4, show=False):
    """
    Plots detected frequencies in specified frequency ranges.

    Parameters:
        file_paths (list): List of file paths to process.
        location (str): Location name or identifier used in processing.
    """

    # Initialize lists to collect frequencies
    freqs = np.full((n_modes, int(len(file_paths)/time_window_factor)), np.nan)
    dt = len(file_paths)/6 / (int(len(file_paths)/time_window_factor) - 1) 
    idx = 0
    results = None
    for i in range(0, len(file_paths)+1-time_window_factor, time_window_factor):
        file_path = [file_paths[j] for j in range(i, i+time_window_factor)]
        # Retrieve frequencies from the file
        _, results = get_vibration_frequencies_best(file_path, location, results_prev=results, p=idx, dt=dt)
        # for freq in results.keys():
            # print(freq, results[freq][0])
            
        # Append frequencies within the specified ranges
        for f in results.keys():
            j = results[f][0]
            freqs[j, idx] = f
        idx += 1
    print(freqs)
    
    time = np.linspace(0, len(file_paths)/6, len(freqs[0]))

    # Create the plots
    visualizer = Visualizer(time, output_dir="results")
    location = "Lello_Jul23_stairs"
    processing_method = "Welch"
    nperseg = 2048 # 256, 512, 1024, 2048, 4096, 8192
    folder_name = f"loc_{location}_wd{600*time_window_factor}_meth_{processing_method}_nperseg_{nperseg}"

    fig, ax = plt.subplots(n_modes, 1, figsize=(3.33*n_modes, 7), sharex=True)

    line_styles = ['-', '--', '-.', ':']
    colors = ['blue', 'green', 'red', 'black']
    for i in range(n_modes):
        ax[i].plot(time, freqs[i], label=f"Mode {i+1}") 
        ax[i].set_ylabel('Frequency (Hz)')
        ax[i].legend()

    ax[-1].set_xlabel('Time [h]')

    fig.tight_layout()
    if show: fig.show()
    print(folder_name)
    visualizer._save_figure(fig, "Detected_Frequencies", folder_name)

# Example usage
folder_path = "data/Lello/Lello_2023_07_02_WholeDay"
folder_path = "data/Lello/LelloNight_Jul23_Apr24"
location = "Lello_Jul23_stairs_night"
time_window_factor = 1 # 2, 4
file_paths = get_files_list(folder_path)
# file_paths = ["data/Lello/LelloNight_Jul23_Apr24/data_2023_07_02_050019_0.csv",
#               "data/Lello/LelloNight_Jul23_Apr24/data_2023_07_02_051032_0.csv",
#               "data/Lello/LelloNight_Jul23_Apr24/data_2023_07_02_052045_0.csv",
#               "data/Lello/LelloNight_Jul23_Apr24/data_2023_07_02_053057_0.csv",
#               "data/Lello/LelloNight_Jul23_Apr24/data_2023_07_02_054110_0.csv",
#               "data/Lello/LelloNight_Jul23_Apr24/data_2023_07_02_055122_0.csv"]
print(len(file_paths[0:20]))
plot_frequencies_best(file_paths[0:80], location, time_window_factor)
