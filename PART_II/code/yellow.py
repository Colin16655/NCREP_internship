import scipy.io
import numpy as np
from utils import save_figure
import matplotlib.pyplot as plt
from process_array import ProcessArray

import sys
import os

# Add the directory containing the package to sys.path
package_path = '/home/boubou/Documents/NCREP_internship/PART_I'
sys.path.append(package_path)

# import the package
from helper.processor import ModalFrequencyAnalyzer, PeakPicker
from helper.visualizer import Visualizer

class Yellow:
    """
    A class to apply statistical and frequency analysis to the Yellow structure data.
    """
    def __init__(self, paths, S, k):
        """
        Initialize with file paths and parameters.

        Args:
            paths (list): List of file paths for the data.
            S (int): # of SDOs.
            k (int): # of clusters.
        """
        self.paths = paths
        self.S = S
        self.k = k
        self.names = ["DA01", "DA02", "DA03", "DA04", "DA05", "DA06", "DA07", "DA08", "DA09", "DA10", "DA11", "DA12", "DA13", "DA14", "DA15", "DA16"]
        self.ref_data = None
        self.dam_data = None
        self.merged_data = None
        self.fsdasy = None
        self.fs = None
        self.ref_data = None
        self.dam_data = None

        self.analyzer = None

    def get_data(self):
        """
        Load and preprocess data from provided paths.

        Extract reference and damaged data, and compute the sampling frequency.
        """
        paths = self.paths
        _, dasy, _, _ = self._load_mat_file(paths[0])

        self.ref_data = np.array([dasy[name][0][0].flatten()[:45568] for name in self.names]) # !!! USER
        self.ref_data -= np.mean(self.ref_data, axis=1, keepdims=True)
        
        self.dam_data = np.zeros((len(paths) - 1, len(self.names), len(self.ref_data[0])))

        for i, path in enumerate(paths[1:]):
            dasy_descr, dasy, filedescription, fsdasy = self._load_mat_file(path)
            for j, name in enumerate(self.names):
                self.dam_data[i, j, :] = dasy[name][0][0].flatten()[:45568]     
                self.dam_data[i, j, :] -= np.mean(self.dam_data[i, j, :])
        # Extract the sampling frequency
        self.fs = fsdasy[0][0]  # Extracting the actual value from the array
    
    
    def plot_acc(self, folder_name):
        """
        Plot acceleration data for reference and damaged datasets.

        Args:
            folder_name (str): Folder name for saving the plot.
        """
        # Generate the time vector
        time_ref = np.arange(0, len(self.ref_data[0])) / self.fs
        time_dam = np.arange(0, len(self.dam_data[0][0])) / self.fs
            
        labels = ["dam 1", "dam 2", "dam 3", "dam 4", "dam 5", "dam 6", "dam 7", "dam 8"]

        fig0, ax0 = plt.subplots(2, len(labels)//2, figsize=(4*len(labels)//2, 2 * 2), sharex=True)
        fig1, ax1 = plt.subplots(1, 1, figsize=(4, 2))

        for i, label in enumerate(labels):
            ax0[i//4, i%4].plot(time_dam, self.dam_data[i][0], label=label)
            ax0[i//4, i%4].legend()
            ax0[i//4, i%4].set_ylabel('Acc [g]')
            ax0[i//4, i%4].set_xlim([time_dam[0], time_dam[-1]])

        ax1.plot(time_ref, self.ref_data[0], label='ref')
        ax1.legend()
        ax1.set_ylabel('Acc [g]')
        ax1.set_xlabel('Time [s]')
        ax1.set_xlim([time_ref[0], time_ref[-1]])

        for i in range(4) : ax0[1, i].set_xlabel('Time [s]')
        fig0.tight_layout()
        save_figure(fig0, f"acc_dam", folder_name, output_dir="PART_II/results", format='pdf')
        save_figure(fig1, f"acc_ref", folder_name, output_dir="PART_II/results", format='pdf')

    def apply_stat_analysis(self, Ls, dam, folder_name):
        """
        Apply statistical analysis on merged reference and damaged data.

        Args:
            Ls (list): List of window lengths for analysis.
            dam (int): Index of the damaged dataset to analyze.
            folder_name (str): Folder name for saving the results.
        """
        # Merge ref with dam data
        self.merged_data = np.concatenate((self.ref_data.T, self.dam_data[dam].T), axis=0) # USER
        i0 = len(self.ref_data[0])
        i1 = len(self.merged_data) // 6
        i2 = 2 * i1
        temp = self.merged_data[:i1]
        self.merged_data[:i1] = self.merged_data[i1:i2]
        self.merged_data[i1:i2] = temp
        # Process merged array
        fig, ax = plt.subplots(len(Ls), 1, figsize=(5, 1*len(Ls)), sharex=True)
        labels = [1/200 * L for L in Ls]

        DI_ALL_valuses = []

        for i, L in enumerate(Ls):
            processor = ProcessArray(self.S, L, self.k, self.merged_data)

            # plot NI, CB, DI
            processor.plot(f"NI_CB_DI_L_{labels[i]}", folder_name, len(self.merged_data) / self.fs)

            DI_values = processor.DI_values
            DI_ALL_valuses.append(DI_values)

            time = np.linspace(0, len(self.merged_data) / self.fs, len(DI_values))
            # Separate the indices and values for positive and negative DI values
            indices = np.arange(len(DI_values))
            positive_indices = indices[DI_values > 0]
            negative_indices = indices[DI_values <= 0]

            positive_values = DI_values[DI_values > 0]
            negative_values = DI_values[DI_values <= 0]

            # Generate the corresponding time values for positive and negative indices
            positive_time = time[positive_indices] + L/self.fs
            negative_time = time[negative_indices] + L/self.fs

            if len(positive_time)   > 0 : 
                max = np.max(positive_values)
                ax[i].stem(positive_time, positive_values, linefmt='r-', markerfmt='ro', basefmt=" ")
            else: max = 0
            if len(negative_values) > 0 : 
                min = np.min(negative_values)
                ax[i].stem(negative_time, negative_values, linefmt='g-', markerfmt='go', basefmt=" ")
            else: min = 0
            ax[i].vlines(i1 / self.fs, ymin=min, ymax=max, color='grey', linestyle='--')
            ax[i].vlines(i2 / self.fs, ymin=min, ymax=max, color='grey', linestyle='--')
            ax[i].vlines(i0 / self.fs, ymin=min, ymax=max, color='red', linestyle='--', label='damage')
            
            ax[i].set_ylabel('DI')                
            ax[i].set_title(f'L = {labels[i]} [s]')
            ax[i].set_xlim([time[0], time[-1]])
            
        ax[-1].set_xlabel('Time [s]')
        fig.tight_layout()
        save_figure(fig, f"NI_CB_DI_dam_{dam+1}", folder_name, output_dir="PART_II/results", format='pdf')
    
    def apply_freq_analysis(self, nperseg=1024, plot=True, folder_name=""):
        """
        Perform frequency analysis on the reference data.

        Args:
            nperseg (int): Number of samples per segment for FFT (default is 1024).
            plot (bool): Whether to plot results (default is True).
            folder_name (str): Folder name for saving the plots.
        """
        self.analyzer = ModalFrequencyAnalyzer(dt=1/self.fs, data=self.ref_data.T)

        # Computation to initialize analyzer
        time = np.linspace(0, len(self.merged_data) / self.fs, len(self.merged_data))
        visualizer = Visualizer(time, output_dir="PART_II/results")
        visualizer.plot_data(self.merged_data, f"merged_acc_dam_{0}", folder_name=folder_name, y_label="Acceleration")
        freqs, psd_matrix = self.analyzer.compute_psd_matrix(nperseg=nperseg)
        visualizer.plot_psd(freqs, psd_matrix, folder_name, linear=False)
        self.analyzer.compute_coherence_matrix()
        U_corr, S_corr, V_corr = self.analyzer.perform_svd_coherence()
        U_PSD, S_PSD, V_PSD  = self.analyzer.perform_svd_psd()
        P1, P2, P3 = self.analyzer.compute_pp_index()

        # Peak picking 
        band = (1e-10, 30)
        peak_picker = PeakPicker(self.analyzer)
        peaks, _ = peak_picker.identify_peaks_2(S_PSD[:, 0], U_PSD, band=band, distance=1, mac_threshold=0.95, n_modes=4, p=16, dt=1/self.fs)
        # print("Detected peaks: ", freqs[peaks.astype(int)])
        visualizer.plot_sigmas(freqs, S_PSD, peaks, folder_name, filename='PSD_SVD', plot_smooth=False, band=band, legend=False)
        visualizer.plot_pp_index(freqs, [P1, P2, P3], peaks, folder_name, filename='PP_indices', plot_smooth=False, band=band)
 
    @staticmethod
    def _load_mat_file(relative_path):
        """
        Load a MATLAB .mat file and extract relevant data.

        Args:
            relative_path (str): Path to the .mat file.

        Returns:
            tuple: Extracted data including descriptions, structure, file info, and sampling frequency.
        """
        # Construct the full file path
        file_path = os.path.join(relative_path)
            
        # Load the .mat file
        data = scipy.io.loadmat(file_path)
            
        # Extracting the variables from the loaded data
        dasy_descr = data.get('dasy_descr')  # Channel descriptions
        dasy = data.get('dasy')              # Data structure
        filedescription = data.get('filedescription')  # Experiment description
        fsdasy = data.get('fsdasy')          # Sampling frequency
            
        return dasy_descr, dasy, filedescription, fsdasy
    