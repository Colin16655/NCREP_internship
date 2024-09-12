import numpy as np
import matplotlib.pyplot as plt
from utils import save_figure
from process_array import ProcessArray

import sys
import os
import csv

# Add the directory containing the package to sys.path
package_path = '/home/boubou/Documents/NCREP_internship/PART_I'
sys.path.append(package_path)

# Import the necessary classes
from helper.processor import ModalFrequencyAnalyzer, PeakPicker
from helper.visualizer import Visualizer

class Alfredo:
    def __init__(self, paths, S, k, location="Alfredo"):
        self.paths = paths
        self.S = S
        self.k = k
        self.names = ["$x_2$", "$y_2$", "$x_1$", "$y_1$"]
        self.merged_data = None
        self.fs = None
        self.dt = None
        self.location = location

        self.analyzer = None
        self.I = []

    def get_data(self):
        all_data = []
        temp = 0
        for path in self.paths:
            dt, data = self._extract_data_from_txt(path)

            # Apply the Hamming window independently to each channel
            window = np.hamming(len(data))
            for i in range(data.shape[1]):  # Loop through each channel
                data[:, i] = data[:, i] * window
            
            # Calculate the sampling frequency based on dt (assuming the same dt for all files)
            if self.fs is None:
                self.fs = 1.0 / dt

            # Append the data from each file to the all_data list
            all_data.append(data)
            self.I.append(temp + len(data))
            temp += len(data)
        
        # Concatenate all the data along the vertical axis e(time axis)
        self.merged_data = np.vstack(all_data)
        self.I = np.array(self.I[:-1])
        

    def plot_acc(self, folder_name):
        time = np.arange(0, len(self.merged_data)) / self.fs
            
        labels = ["Signal 1", "Signal 2", "Signal 3", "Signal 4"]

        fig, ax = plt.subplots(2, len(labels)//2, figsize=(4*len(labels)//2, 2 * 2), sharex=True)

        for i, label in enumerate(labels):
            ax[i//2, i%2].plot(time, self.merged_data[:, i], label=label)
            ax[i//2, i%2].legend()
            ax[i//2, i%2].set_ylabel('Acc [g]')
            ax[i//2, i%2].set_xlim([time[0], time[-1]])

        ax[1, 0].set_xlabel('Time [s]')
        ax[1, 1].set_xlabel('Time [s]')
        
        fig.tight_layout()
        save_figure(fig, f"acc_signals", folder_name, format='pdf')

    def apply_stat_analysis(self, Ls, folder_name):
        # Process merged array
        fig, ax = plt.subplots(len(Ls), 1, figsize=(5, 1*len(Ls)), sharex=True)
        labels = [self.dt * L for L in Ls]

        DI_ALL_valuses = []

        for i, L in enumerate(Ls):
            processor = ProcessArray(self.S, L, self.k, self.merged_data, self.location, np.copy(self.I))
            # plot NI, CB, DI
            processor.plot(f"NI_CB_DI_L_{labels[i]}", len(self.merged_data) / self.fs)

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
            positive_time = time[positive_indices]
            negative_time = time[negative_indices]

            if len(positive_time)   > 0 : 
                max = np.max(positive_values)
                ax[i].stem(positive_time, positive_values, linefmt='r-', markerfmt='ro', basefmt=" ")
            else: max = 0
            if len(negative_values) > 0 : 
                min = np.min(negative_values)
                ax[i].stem(negative_time, negative_values, linefmt='g-', markerfmt='go', basefmt=" ")
            else: min = 0
            for ii in processor.I:
                ax[i].vlines(ii * (int(len(self.merged_data) / self.fs / len(DI_values))+1), ymin=min, ymax=max, color='red', linestyle='--')
        
            ax[i].set_ylabel('DI')                
            ax[i].set_title(f'L = {labels[i]} [s]')
            ax[i].set_xlim([time[0], time[-1]])
            
        ax[-1].set_xlabel('Time [s]')
        fig.tight_layout()
        save_figure(fig, f"NI_CB_DI", folder_name, output_dir="PART_II/results", format='pdf')
    
    def apply_freq_analysis(self, nperseg=512, plot=True, folder_name="", filename1='PSD_SVD', filename2='PP_indices', name='', band=(1e-10, 25)):
        self.analyzer = ModalFrequencyAnalyzer(dt=1/self.fs, data=self.merged_data)

        # Computation to initialize analyzer
        time = np.linspace(0, len(self.merged_data) / self.fs, len(self.merged_data))
        visualizer = Visualizer(time, output_dir="PART_II/results")
        freqs_fft, fft_data = self.analyzer.compute_fft()

        ## PLOT FFT - START
        fig, ax = plt.subplots(1, 1, figsize=(5.5, 3))
        labels = ["$x_2$", "$y_2$", "$x_1$", "$y_1$"]
        for i in range(self.merged_data.shape[1]):
            ax.semilogy(freqs_fft, np.abs(fft_data[:, i]), label=labels[i], linewidth=0.5)
        ax.legend()
        ax.set_ylabel("FFT - Amplitude")
        ax.set_xlim(band[0], band[1])
        ax.set_ylim(1e-3, 13)
        ax.set_xlabel("Frequency [Hz]")
        fig.tight_layout()

        visualizer._save_figure(fig, "FFT_results", folder_name)
        ## PLOT FFT - END
        # visualizer.plot_data(self.data, f"merged_acc_dam_{0}", folder_name=folder_name, y_label="Acceleration")
        freqs, psd_matrix = self.analyzer.compute_psd_matrix(nperseg=nperseg)
        visualizer.plot_psd(freqs, psd_matrix, folder_name, linear=False)
        self.analyzer.compute_coherence_matrix()
        U_corr, S_corr, V_corr = self.analyzer.perform_svd_coherence()
        U_PSD, S_PSD, V_PSD  = self.analyzer.perform_svd_psd()
        P1, P2, P3 = self.analyzer.compute_pp_index()

        # Peak picking 
        peak_picker = PeakPicker(self.analyzer)
        peaks, _ = peak_picker.identify_peaks_2(S_PSD[:, 0], U_PSD, band=band, distance=1, mac_threshold=0.9, n_modes=4, p=0, dt=1/self.fs, folder_name=folder_name, name=name)
        temp = []
        for peak in peaks : 
            if not np.isnan(peak) : 
                temp.append(peak)
        peaks = np.array(temp)
        print("Detected peaks: ", freqs[peaks.astype(int)])
        visualizer.plot_sigmas(freqs, S_PSD, peaks, folder_name, filename=filename1, plot_smooth=False, band=band, legend=True)
        visualizer.plot_pp_index(freqs, [P1, P2, P3], peaks, folder_name, filename=filename2, plot_smooth=False, band=band)

    def detect_damages_PART_I(self, ax_fft, ax_psd, ax_pp, colors, style, nperseg=1024, folder_name="", filename1='PSD_SVD', filename2='PP_indices', name='', band=(1e-10, 25)):
        self.analyzer = ModalFrequencyAnalyzer(dt=1/self.fs, data=self.merged_data)

        # Computation to initialize analyzer
        time = np.linspace(0, len(self.merged_data) / self.fs, len(self.merged_data))
        visualizer = Visualizer(time, output_dir="PART_II/results")
        freqs_fft, fft_data = self.analyzer.compute_fft()

        ## PLOT FFT - START
        labels = ["$x_2$", "$y_2$", "$x_1$", "$y_1$"]
        for i in range(self.merged_data.shape[1]):
            ax_fft.semilogy(freqs_fft, np.abs(fft_data[:, i]), label=labels[i], linewidth=0.5, color=colors[i], linestyle=style)
        ## PLOT FFT - END
        freqs, psd_matrix = self.analyzer.compute_psd_matrix(nperseg=nperseg)
        self.analyzer.compute_coherence_matrix()
        U_corr, S_corr, V_corr = self.analyzer.perform_svd_coherence()
        U_PSD, S_PSD, V_PSD  = self.analyzer.perform_svd_psd()
        P1, P2, P3 = self.analyzer.compute_pp_index()

        # Peak picking 
        peak_picker = PeakPicker(self.analyzer)
        peaks, _ = peak_picker.identify_peaks_2(S_PSD[:, 0], U_PSD, band=band, distance=1, mac_threshold=0.9, n_modes=4, p=0, dt=1/self.fs, folder_name=folder_name, name=name)
        temp = []
        for peak in peaks : 
            if not np.isnan(peak) : 
                temp.append(peak)
        peaks = np.array(temp)
        visualizer.plot_sigmas(freqs, S_PSD, peaks, folder_name, filename=filename1, plot_smooth=False, band=band, legend=False, ax=ax_psd)
        visualizer.plot_pp_index(freqs, [P1, P2, P3], peaks, folder_name, filename=filename2, plot_smooth=False, band=band, ax=ax_pp, legend=False, style=style)

    def _extract_data_from_txt(self, file_path):
        data_section = False
        data = []

        with open(file_path, 'r') as txtfile:
            lines = txtfile.readlines()
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                # Identify the dt line and extract dt value
                if line.startswith("dt:"):
                    # Skip the next line to get the actual dt value
                    i += 1
                    dt_line = lines[i].strip()
                    self.dt = float(dt_line.replace(',', '.'))
                
                # Once the "data:" section is reached, start reading data
                if line.startswith("data:"):
                    data_section = True
                    i += 1  # Move to the first line of data
                    line = lines[i].strip()
                
                # Read the data lines
                if data_section and line:
                    # Split the line into individual values and convert them to float
                    data.append([float(x.replace(',', '.')) for x in line.split()])
                
                i += 1
        
        # Convert data to a numpy array
        data_array = np.array(data)
        return self.dt, data_array
