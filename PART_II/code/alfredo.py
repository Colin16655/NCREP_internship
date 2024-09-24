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
    """
    A class to apply statistical and frequency analysis to Alfredo data.
    """
    def __init__(self, paths, S, k):
        """
        Initialize Alfredo class with file paths, parameters, and location.
        
        Args:
            paths (list): List of file paths.
            S (int): # of SDOs for the statistical analysis.
            k (int): # of clusters for the statistical analysis.
            location (str, optional): Location identifier. Defaults to "Alfredo".
        """
        self.paths = paths
        self.S = S
        self.k = k

        self.merged_data = None
        self.fs = None
        self.dt = None # 1/fs
        self.analyzer = None
        self.I = []

    def get_data(self):
        """
        Load and process data from paths, applying a Hamming window.
        """
        all_data = []
        temp = 0
        for path in self.paths:
            dt, data = self._extract_data_from_txt(path)

            # Apply the Hamming window independently to each channel
            window = np.hamming(len(data))
            for i in range(data.shape[1]):  # Loop through each channel
                data[:, i] = data[:, i] * window
            
            # Calculate the sampling frequency based on dt (same dt for all files)
            if self.fs is None:
                self.fs = 1.0 / dt

            all_data.append(data)
            self.I.append(temp + len(data)) # for displaying the red dashed vertical lines (damages)
            temp += len(data)
        
        # Concatenate all the data along the vertical axis
        self.merged_data = np.vstack(all_data)
        self.I = np.array(self.I[:-1])   

    def plot_acc(self, folder_name):
        """
        Plot acceleration signals over time and save the figure.

        Args:
            folder_name (str): Folder to save the plot.
        """
        time = np.arange(0, len(self.merged_data)) / self.fs
            
        labels = ["Signal 1", "Signal 2", "Signal 3", "Signal 4"] # For Alfredo, 4 channels

        fig, ax = plt.subplots(2, len(labels)//2, figsize=(4*len(labels)//2, 2 * 2), sharex=True)

        for i, label in enumerate(labels):
            ax[i//2, i%2].plot(time, self.merged_data[:, i], label=label)
            ax[i//2, i%2].legend()
            ax[i//2, i%2].set_ylabel('Acc [g]')
            ax[i//2, i%2].set_xlim([time[0], time[-1]])

        ax[1, 0].set_xlabel('Time [s]')
        ax[1, 1].set_xlabel('Time [s]')
        
        fig.tight_layout()
        save_figure(fig, f"acc_signals", folder_name, output_dir="PART_II/results", format='pdf')

    def apply_stat_analysis(self, Ls, folder_name):
        """
        Apply statistical analysis on the data for different values of L.

        Args:
            Ls (list): List of L values for analysis.
            folder_name (str): Folder to save the plots.
        """
        fig, ax = plt.subplots(len(Ls), 1, figsize=(5, 1*len(Ls)), sharex=True)
        labels = [self.dt * L for L in Ls]

        DI_ALL_valuses = []

        for k, L in enumerate(Ls):
            processor = ProcessArray(self.S, L, self.k, self.merged_data)
            # plot NI, CB, DI
            processor.plot(f"NI_CB_DI_L_{labels[k]}", folder_name, len(self.merged_data) / self.fs)

            DI_values = processor.DI_values
            DI_ALL_valuses.append(DI_values)
            time = L * self.dt * np.arange(len(processor.DI_values))
            # Separate the indices and values for positive and negative DI values
            indices = np.arange(len(DI_values))
            positive_indices = indices[DI_values > 0]
            negative_indices = indices[DI_values <= 0]

            positive_values = DI_values[DI_values > 0]
            negative_values = DI_values[DI_values <= 0]

            # Generate the corresponding time values for positive and negative indices
            positive_time = time[positive_indices] + L*self.dt
            negative_time = time[negative_indices] + L*self.dt

            if len(positive_time)   > 0 : 
                max = np.max(positive_values)
                ax[k].stem(positive_time, positive_values, linefmt='r-', markerfmt='ro', basefmt=" ")
            else: max = 0
            if len(negative_values) > 0 : 
                min = np.min(negative_values)
                ax[k].stem(negative_time, negative_values, linefmt='g-', markerfmt='go', basefmt=" ")
            else: min = 0
            for i in self.I:
                ax[k].vlines(i * self.dt, ymin=min, ymax=max, color='red', linestyle='--')
        
            ax[k].set_ylabel('DI')                
            ax[k].set_title(f'L = {labels[k]} [s]')
            ax[k].set_xlim([time[0], time[-1]])
            
        ax[-1].set_xlabel('Time [s]')
        fig.tight_layout()
        save_figure(fig, f"NI_CB_DI", folder_name, output_dir="PART_II/results", format='pdf')
    
    def apply_freq_analysis(self, nperseg=512, plot=True, folder_name="", filename1='PSD_SVD', filename2='PP_indices', name='', band=(1e-10, 25)):
        """
        Perform frequency analysis using PSD and peak picking.

        Args:
            nperseg (int, optional): Number of segments for PSD. Defaults to 512.
            plot (bool, optional): Whether to plot results. Defaults to True.
            folder_name (str, optional): Folder to save results. Defaults to "".
            filename1 (str, optional): Filename for PSD. Defaults to 'PSD_SVD'.
            filename2 (str, optional): Filename for peak picking indices. Defaults to 'PP_indices'.
            name (str, optional): Name of the analysis case (M_0_0_0 for example). Defaults to ''.
            band (tuple, optional): Frequency band for analysis. Defaults to (1e-10, 25).
        """
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
        """
        Detect damages using frequency analysis and visualizations from Part I.

        Args:
            ax_fft, ax_psd, ax_pp (matplotlib axes): Axes to plot FFT, PSD, and PP results.
            colors (list): Colors for plotting.
            style (str): Line style for plotting.
            nperseg (int, optional): Number of segments for PSD. Defaults to 1024.
            folder_name (str, optional): Folder to save results. Defaults to "".
            filename1 (str, optional): Filename for PSD. Defaults to 'PSD_SVD'.
            filename2 (str, optional): Filename for peak picking indices. Defaults to 'PP_indices'.
            name (str, optional): Name of the analysis case (M_0_0_0 for example). Defaults to ''.
            band (tuple, optional): Frequency band for analysis. Defaults to (1e-10, 25).
        """
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

    def apply_freq_analysis_PART_I(self, pp_args, batchsize, ranges_display, methods, folder_name):
        """
        Applies frequency analysis in batches using various peak-picking methods and visualizes the results.

        Args:
            pp_args (dict): Peak-picking arguments (e.g., nperseg, distance, sigma).
            batchsize (int): Number of data points per batch.
            ranges_display (list): Frequency ranges to display and track.
            methods (list): List of method indices for peak-picking.
            folder_name (str): Folder name to save plots.

        Returns:
            None
        """
        analyzer = ModalFrequencyAnalyzer(dt=1/self.fs)
        peak_picker = PeakPicker(analyzer)
        num_batches = len(self.merged_data) // batchsize

        detected_freqs = np.full((len(ranges_display), 1, num_batches), np.nan)
        f_window = (ranges_display[0][0], ranges_display[-1][-1])

        results = None
        for idx in range(num_batches):
            start_idx = idx * batchsize
            end_idx = min((idx + 1) * batchsize, len(self.merged_data))
            analyzer.data = self.merged_data[start_idx:end_idx]
            # Computation to initialize analyzer
            analyzer.compute_psd_matrix(nperseg=pp_args['nperseg'])
            analyzer.compute_coherence_matrix()
            analyzer.perform_svd_coherence()
            analyzer.perform_svd_psd()
            analyzer.compute_pp_index()
            for i, method in enumerate(methods):
                # Compute frequencies (assuming this returns a numpy array of frequencies)
                if method == 0:
                    label = 'method 0'
                    peaks = peak_picker.identify_peaks_0(analyzer.P3, distance=pp_args['distance0']) 
                elif method == 1:
                    label = 'method 1'
                    peaks = peak_picker.identify_peaks_1(analyzer.P3, analyzer.S_psd[:, 0], distance=pp_args['distance1'], sigma=pp_args['sigma'], ranges_to_check=pp_args['ranges_to_check']) 
                elif method == 2 :
                    label = 'method 2'
                    peaks, results = peak_picker.identify_peaks_2(analyzer.S_psd[:, 0], analyzer.U_psd, band=f_window, distance=pp_args['distance2'], mac_threshold=pp_args['mac_threshold'],
                                                                   n_modes=pp_args['n_modes'], n_mem=pp_args['n_mem'], results_prev=results, p=idx, dt=self.dt, plotdebug=True) 
                elif method == 3:
                    label = 'PyOMA'
                    peaks = peak_picker.identify_peaks_pyoma()
                elif method == 4 :
                    label = 'method 2'
                    peaks, results = peak_picker.identify_peaks_2(analyzer.P3, analyzer.U_psd, band=f_window, distance=pp_args['distance2'], mac_threshold=pp_args['mac_threshold'], 
                                                                  n_modes=pp_args['n_modes'], n_mem=pp_args['n_mem'], results_prev=results, p=idx, dt=self.dt, plotdebug=True)
                
                else:
                    raise ValueError(f"Unsupported method {method}")

                if method != 3 : 
                    mode_freqs = []
                    for peak in peaks:
                        if np.isnan(peak):
                            mode_freqs.append(np.nan)
                        else:
                            peak = int(peak)
                            mode_freqs.append(analyzer.freq_psd[peak])
                    mode_freqs = np.array(mode_freqs)

                else : mode_freqs = peaks
                # print("m", mode_freqs)

                for j, band in enumerate(ranges_display):
                    band_min, band_max = band
                    mask = (mode_freqs >= band_min) & (mode_freqs < band_max)
                    # in case multiple modes are detected in the same band, we keep the closest to the previous value
                    if len(mode_freqs[mask]) > 1 : 
                        if not np.isnan(detected_freqs[j, i, idx]):
                            val = mode_freqs[ np.argmin(np.abs(np.array(mode_freqs[mask]) - detected_freqs[j, i, idx])) ]
                        else : val = mode_freqs[mask][0]
                    else : val = mode_freqs[mask]
                    if len(mode_freqs[mask]) > 0: 
                        detected_freqs[j, i, idx] = val 
                    
        # print(detected_freqs)

        # ---- PLOT ----
        file_duration = batchsize * self.dt / 60 # 1 file duration in minutes
        time = np.linspace(0, num_batches*file_duration, len(detected_freqs[0][0])+1)

        # Create the plots
        visualizer = Visualizer(time, output_dir="PART_II/results")

        fig, ax = plt.subplots(1, 1, figsize=(3.33, 3))

        linestyles_list = ['-', '--', '-.', ':']
        linestyles = [linestyles_list[i % len(linestyles_list)] for i in range(len(methods))]
        colors_list = ['blue', 'green', 'red', 'black']  
        colors = [colors_list[i % len(colors_list)] for i in range(len(methods))]

        labels = np.array(["method 0", "method 1", "method 2 PSD", "PyOMA", "method 2 PP"])[methods]

        for i in range(len(ranges_display)):
            for j, (label, linestyle) in enumerate(zip(labels, linestyles)):
                y = np.concatenate((detected_freqs[i][j], [np.nan]))
                ax.step(time, y, where='post', label=label, color=colors_list[i], linestyle=linestyles_list[i])  
            ax.set_ylabel('f (Hz)')
            # ax.legend()       

        ax.set_xlabel('Time [min]')

        fig.tight_layout()
        visualizer._save_figure(fig, "Detected_Frequencies", folder_name)

    def _extract_data_from_txt(self, file_path):    
        """
        Extracts sampling interval and data from a specified text file.
        
        Args:
            file_path (str): Path to the text file containing data.

        Returns:
            tuple: Sampling interval (dt) and data as a numpy array.
        """
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
