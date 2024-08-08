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

def plot_frequencies(file_paths, location, time_window_factor, show=False):
    """
    Plots detected frequencies in specified frequency ranges.

    Parameters:
        file_paths (list): List of file paths to process.
        location (str): Location name or identifier used in processing.
    """

    # Initialize lists to collect frequencies
    freqs_8_13 = np.full((4, int(len(file_paths)/time_window_factor)), np.nan)
    freqs_13_18 = np.full((4,int(len(file_paths)/time_window_factor)), np.nan)
    freqs_18_24 = np.full((4,int(len(file_paths)/time_window_factor)), np.nan)
    idx = 0
    for i in range(0, len(file_paths)+1-time_window_factor, time_window_factor):
        file_path = [file_paths[j] for j in range(i, i+time_window_factor)]
        # Retrieve frequencies from the file
        vibration_f_Colin = get_vibration_frequencies(file_path, location)
        vibration_f_FDD, vibration_f_EFDD = get_vibration_frequencies_pyoma(file_path, location)
        for j, vibration_f in enumerate([vibration_f_Colin, vibration_f_FDD, vibration_f_EFDD]):
            # Convert vibration_f to a numpy array for easier manipulation
            vibration_f = np.array(vibration_f)

            # Define masks for each frequency range
            mask_8_13 = (vibration_f >= 8) & (vibration_f < 13)
            mask_13_18 = (vibration_f >= 13) & (vibration_f < 18)
            mask_18_24 = (vibration_f >= 18) & (vibration_f <= 24)
            
            # Append frequencies within the specified ranges
            if len(vibration_f[mask_8_13]) > 0: freqs_8_13[j, idx] = vibration_f[mask_8_13]
            if len(vibration_f[mask_13_18]) > 0: freqs_13_18[j ,idx] = vibration_f[mask_13_18]
            if len(vibration_f[mask_18_24]) > 0 : freqs_18_24[j, idx] = vibration_f[mask_18_24]
        idx += 1
    
    time = np.linspace(0, len(file_paths)/6, len(freqs_8_13[0]))

    # Create the plots
    visualizer = Visualizer(time, output_dir="results")
    location = "Lello_Jul23_stairs"
    processing_method = "Welch"
    nperseg = 2048 # 256, 512, 1024, 2048, 4096, 8192
    folder_name = f"loc_{location}_wd{600*time_window_factor}_meth_{processing_method}_nperseg_{nperseg}"

    fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    line_styles = ['-', '--', '-.']
    colors = ['blue', 'green', 'red']
    for j, (label, line_style, c) in enumerate(zip(["Colin", "FDD", "EFDD"], line_styles, colors)):
        # Plot for 8-13 Hz
        ax[0].plot(time, freqs_8_13[j], label=label, color=c, linestyle=line_style)  
        # Plot for 13-18 Hz
        ax[1].plot(time, freqs_13_18[j], label=label, color=c, linestyle=line_style)
        # Plot for 18-24 Hz
        ax[2].plot(time, freqs_18_24[j], label=label, color=c, linestyle=line_style)
    ax[0].set_ylabel('Frequency (Hz)')
    ax[0].legend()
    ax[1].set_ylabel('Frequency (Hz)')
    ax[1].legend()
    ax[2].set_ylabel('Frequency (Hz)')
    ax[2].legend()

    ax[-1].set_xlabel('Time [h]')

    fig.tight_layout()
    if show: fig.show()
    visualizer._save_figure(fig, "Detected_Frequencies", folder_name)

# Example usage
folder_path = "data/Lello/Lello_2023_07_10_WholeDay"
location = "Lello_Jul23_stairs_WD"
time_window_factor = 2 # 2, 4
file_paths = get_files_list(folder_path)
plot_frequencies(file_paths, location, time_window_factor)
