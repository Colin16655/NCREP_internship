import numpy as np
from helper.data_loader import DataLoader
from helper.signal_transformer import SignalTransformer
from helper.frequency_analyzer import FrequencyAnalyzer
from helper.visualizer import Visualizer
from scipy.ndimage import gaussian_filter1d
import os
from helper.process import get_vibration_frequencies

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

def iterate_files_in_order(folder_path):
    """
    Iterates over all files in a given folder in chronological order based on filenames.

    Parameters:
        folder_path (str): Path to the folder containing the files.
    """
    # List all files in the folder
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    # Sort files by filename (assumes filenames can be sorted lexicographically)
    files.sort()

    # Create full file paths
    sorted_file_paths = [os.path.join(folder_path, file_name) for file_name in files]

    return sorted_file_paths

def plot_frequencies(file_paths, location):
    """
    Plots detected frequencies in specified frequency ranges.

    Parameters:
        file_paths (list): List of file paths to process.
        location (str): Location name or identifier used in processing.
    """

    # Initialize lists to collect frequencies
    freqs_8_13 = np.full(len(file_paths), np.nan)
    freqs_13_18 = np.full(len(file_paths), np.nan)
    freqs_18_24 = np.full(len(file_paths), np.nan)

    for i, file_path in enumerate(file_paths):
        # Retrieve frequencies from the file
        vibration_f = get_vibration_frequencies([file_path], location)
        
        # Convert vibration_f to a numpy array for easier manipulation
        vibration_f = np.array(vibration_f)

        # Define masks for each frequency range
        mask_8_13 = (vibration_f >= 8) & (vibration_f < 13)
        mask_13_18 = (vibration_f >= 13) & (vibration_f < 18)
        mask_18_24 = (vibration_f >= 18) & (vibration_f <= 24)
        
        # Append frequencies within the specified ranges
        if len(vibration_f[mask_8_13]) > 0: freqs_8_13[i] = vibration_f[mask_8_13]
        if len(vibration_f[mask_13_18]) > 0: freqs_13_18[i] = vibration_f[mask_13_18]
        if len(vibration_f[mask_18_24]) > 0 : freqs_18_24[i] = vibration_f[mask_18_24]
    
    time = np.linspace(0, len(file_paths)/6, len(file_paths))

    # Create the plots
    visualizer = Visualizer(time, output_dir="results")
    location = "Lello_Jul24_stairs"
    time_window_size = 600 * len(file_paths)
    processing_method = "Welch"
    nperseg = 2048 # 256, 512, 1024, 2048, 4096, 8192
    folder_name = f"loc_{location}_wd_meth_{processing_method}_nperseg_{nperseg}"

    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    
    # Plot for 8-13 Hz
    ax[0].plot(time, freqs_8_13, label='8-13 Hz', color='blue')
    ax[0].set_title('Detected Frequencies between 8 and 13 Hz')
    ax[0].set_ylabel('Frequency (Hz)')
    ax[0].set_xlabel('Index')
    ax[0].legend()
    
    # Plot for 13-18 Hz
    ax[1].plot(time, freqs_13_18, label='13-18 Hz', color='green')
    ax[1].set_title('Detected Frequencies between 13 and 18 Hz')
    ax[1].set_ylabel('Frequency (Hz)')
    ax[1].set_xlabel('Index')
    ax[1].legend()
    
    # Plot for 18-24 Hz
    ax[2].plot(time, freqs_18_24, label='18-24 Hz', color='red')
    ax[2].set_title('Detected Frequencies between 18 and 24 Hz')
    ax[2].set_ylabel('Frequency (Hz)')
    ax[2].set_xlabel('Index')
    ax[2].legend()


    plt.tight_layout()
    plt.show()
    visualizer._save_figure(fig, "Detected_Frequencies", folder_name)

# Example usage
folder_path = "data/Lello/Lello_2023_07_10_WholeDay"
location = "Lello_Jul23_stairs_WD"
file_paths = iterate_files_in_order(folder_path)
print(len(file_paths))
plot_frequencies(file_paths, location)
