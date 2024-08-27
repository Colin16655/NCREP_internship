import numpy as np
from io import StringIO
import sys
from sklearn.preprocessing import StandardScaler
import os

# Add the directory containing the package to sys.path
package_path = '/home/boubou/Documents/NCREP_internship/PART_I/helper'
sys.path.append(package_path)

# import the package
from processor import ModalFrequencyAnalyzer, PeakPicker
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

# List of file paths
paths = ['data/chimney/1.txt', 'data/chimney/2.txt']

## USER 
nperseg = 2048
fs      = 200 # Hz
band    = (1e-10, 10) # Hz
n_modes = 6 # Number of modes to extract
## USER

output_dir  = "PART_chimney/results"

def get_data(paths):
    # Initialize variables to hold data
    data1, data2 = None, None

    # Iterate over each file path and assign the data to the respective variables
    for i, path in enumerate(paths):
        # Read the file and replace commas with periods
        with open(path, 'r') as file:
            # Read the lines and replace commas with periods
            lines = [line.replace(',', '.') for line in file]
        
        # Convert the processed lines to a NumPy array
        data = np.genfromtxt(StringIO(''.join(lines)))
        
        # Assign the data to the respective variable
        if i == 0:
            data1 = data
        elif i == 1:
            data2 = data

    # Print the shapes of the arrays to confirm
    print("Shape of data1:", data1.shape)
    print("Shape of data2:", data2.shape)

    # time axis
    t1 = np.arange(data1.shape[0]) / fs
    t2 = np.arange(data2.shape[0]) / fs

    return t1, data1, t2, data2

def preprocess(t1, data1, t2, data2):
    # Preprocessing (remove DC component)
    detrended_data1 = data1 - np.mean(data1, axis=0)
    detrended_data2 = data2 - np.mean(data2, axis=0)

    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Plot original data1
    axs[0, 0].plot(t1, data1, label=['x', 'y'])
    axs[0, 0].set_title('Original Data 1')
    axs[0, 0].set_ylabel('Amplitude')
    axs[0, 0].legend()

    # Plot preprocessed data1 (scaled and detrended)
    axs[1, 0].plot(t1, detrended_data1, label=['x', 'y'])
    axs[1, 0].set_title('Preprocessed Data 1')
    axs[1, 0].set_xlabel('Time [s]')
    axs[1, 0].set_ylabel('Amplitude')
    axs[1, 0].legend()

    # Plot original data2
    axs[0, 1].plot(t2, data2, label=['x', 'y'])
    axs[0, 1].set_title('Original Data 2')
    axs[0, 1].legend()

    # Plot preprocessed data2 (scaled and detrended)
    axs[1, 1].plot(t2, detrended_data2, label=['x', 'y'])
    axs[1, 1].set_title('Preprocessed Data 2')
    axs[1, 1].set_xlabel('Time [s]')
    axs[1, 1].legend()

    # Adjust layout
    fig.tight_layout()
    
    vis = Visualizer(None, output_dir=output_dir)
    vis._save_figure(fig, "data", "", format='pdf')
    return detrended_data1, detrended_data2

def process(time, data, folder_name, data_name):
    analyzer = ModalFrequencyAnalyzer(dt=1/fs, time=time, data=data)

    # Computation to initialize analyzer
    visualizer = Visualizer(time, output_dir=output_dir)

    frequencies, fft_data = analyzer.compute_fft(output_dir=os.path.join(output_dir, folder_name), filename='fft_'+data_name)
    visualizer.plot_fft(frequencies, fft_data, folder_name, labels=["x", "y"], band=band)

    freqs, psd_matrix = analyzer.compute_psd_matrix(nperseg=nperseg)
    visualizer.plot_psd(freqs, psd_matrix, folder_name, linear=False, band=band)
    analyzer.compute_coherence_matrix()
    U_corr, S_corr, V_corr = analyzer.perform_svd_coherence()
    U_PSD, S_PSD, V_PSD  = analyzer.perform_svd_psd()
    P1, P2, P3 = analyzer.compute_pp_index()

    # Peak picking 
    peak_picker = PeakPicker(analyzer)
    peaks, _ = peak_picker.identify_peaks_2(S_PSD[:, 0], U_PSD, band=band, distance=1, mac_threshold=0.9, n_modes=n_modes-1, p=0, dt=1/fs, output_dir=output_dir, folder_name=folder_name)

    visualizer.plot_sigmas(freqs, S_PSD, peaks, folder_name, filename='PSD_SVD', plot_smooth=False, band=band)
    visualizer.plot_pp_index(freqs, [P1, P2, P3], peaks, folder_name, filename='PP_indices', plot_smooth=False, band=band)   
    print("Identified modal frequencies : ", freqs[peaks.astype(int)])         

def main():
    t1, data1, t2, data2 = get_data(paths)
    data1_scaled, data2_scaled = preprocess(t1, data1, t2, data2)
    process(t1, data1_scaled, folder_name = f"chimney_data1_nperseg_{nperseg}", data_name="data1")
    process(t2, data2_scaled, folder_name = f"chimney_data2_nperseg_{nperseg}", data_name="data2")

if __name__ == "__main__":
    main()