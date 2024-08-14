import numpy as np
import os
import matplotlib.pyplot as plt
from helper.data_loader import DataLoader
from helper.processor import ModalFrequencyAnalyzer, PeakPicker
from helper.visualizer import Visualizer
from tqdm import tqdm

class FolderProcessor:
    """
    Processes data from a folder of CSV files, computes vibration frequencies through time, and plots the results.

    This class handles the loading of data from multiple CSV files in batches, computes vibration frequencies
    for each batch using the `ModalFrequencyAnalyzer`, and plots the detected frequencies across different
    frequency ranges.
    """

    def __init__(self, selected_indices, location, folder_path, batch_size, pp_args, scaling_factors, methods=[1], ranges_display=[(8,13), (13,18), (18,24)]):
        """
        Initializes the FolderProcessor with parameters for processing data.

        Args:
            selected_indices (list of int): Indices of the columns to be selected from the sensor data.
            folder_path (str): Path to the folder containing the CSV files.
            batch_size (int): Number of files to include in each mini-batch.
            n_mem (int): Number of memory elements for the frequency analysis.
            n_modes (int): Number of modes for the frequency analysis.
            pp_args (dict): Arguments for peak picking.
            scaling_factors (numpy.ndarray): Factors used to scale the sensor data.
            method (list of int): indicates which methods to use for the peak picking (0, 1 or 2 for method 0, 1 or 2 and 3 for PyOMA)
        """
        self.folder_path = folder_path
        self.batch_size = batch_size
        self.pp_args = pp_args
        self.scaling_factors = scaling_factors
        self.loader = DataLoader(selected_indices, folder_path=folder_path, batch_size=batch_size, scaling_factors=scaling_factors)
        self.analyzer = ModalFrequencyAnalyzer()
        self.peak_picker = PeakPicker(self.analyzer)
        self.methods = methods
        self.ranges_display = ranges_display
        self.location = location
        self.n_channels = len(self.ranges_display)
        self.n_methods = len(self.methods)
        self.n_files = len(self.loader)
        print(self.n_files)
        self.detected_freqs = np.full((self.n_channels, self.n_methods, self.n_files), np.nan)

    def process(self):
        """
        Processes each batch of data from the folder, computes vibration frequencies, and stores them for plotting.

        Iterates through batches of data loaded from CSV files, uses the `ModalFrequencyAnalyzer` to compute
        vibration frequencies for each batch, and stores these frequencies in the `frequency_data` attribute.
        """
        # Initialize arrays to collect frequencies
        f_window = (self.ranges_display[0][0], self.ranges_display[-1][-1])
        results = None

        dt = 1/6 * self.batch_size # 10 minutes : 1 file duration in hours

        for idx, (time, data) in enumerate(tqdm(self.loader, desc="Processing Batches", unit="batch")):
            self.analyzer.time = time
            print(idx)
            # self.analyzer.data = data

            # # Computation to initialize analyzer
            # self.analyzer.compute_psd_matrix(nperseg=self.pp_args['nperseg'])
            # self.analyzer.compute_coherence_matrix()
            # self.analyzer.perform_svd_coherence()
            # self.analyzer.perform_svd_psd()
            # self.analyzer.compute_pp_index()
            # for i, method in enumerate(self.methods):
            #     # Compute frequencies (assuming this returns a numpy array of frequencies)
            #     if method == 0:
            #         label = 'method 0'
            #         peaks = self.peak_picker.identify_peaks_0(self.analyzer.P3, distance=self.pp_args['distance0']) 
            #     elif method == 1:
            #         label = 'method 1'
            #         peaks = self.peak_picker.identify_peaks_1(self.analyzer.P3, self.analyzer.S_psd[:, 0], distance=self.pp_args['distance1'], sigma=self.pp_args['sigma'], ranges_to_check=self.pp_args['ranges_to_check']) 
            #     elif method == 2 :
            #         label = 'method 2'
            #         peaks, results = self.peak_picker.identify_peaks_2(self.analyzer.S_psd[:, 0], self.analyzer.U_psd, band=f_window, distance=self.pp_args['distance2'], mac_threshold=self.pp_args['mac_threshold'], n_modes=self.pp_args['n_modes'], n_mem=self.pp_args['n_mem'], results_prev=results, p=idx, dt=dt) 
            #     elif method == 3:
            #         label = 'PyOMA'
            #         peaks = self.peak_picker.identify_peaks_pyoma()
            #     elif method == 4 :
            #         label = 'method 2'
            #         peaks, results = self.peak_picker.identify_peaks_2(self.analyzer.P3, self.analyzer.U_psd, band=f_window, distance=self.pp_args['distance2'], mac_threshold=self.pp_args['mac_threshold'], n_modes=self.pp_args['n_modes'], n_mem=self.pp_args['n_mem'], results_prev=results, p=idx, dt=dt)
                
            #     else:
            #         raise ValueError(f"Unsupported method {method}")

            #     if method != 3 : mode_freqs, mode_shapes = self.peak_picker.identify_mode_shapes(self.analyzer.U_psd, peaks)
            #     else : mode_freqs = peaks

            #     for j, band in enumerate(self.ranges_display):
            #         band_min, band_max = band
            #         mask = (mode_freqs >= band_min) & (mode_freqs < band_max)
            #         # in case multiple modes are detected in the same band, we keep the closest to the previous value
            #         if len(mode_freqs[mask]) > 1 : 
            #             if not np.isnan(self.detected_freqs[j, i, idx]):
            #                 val = mode_freqs[ np.argmin(np.abs(np.array(mode_freqs[mask]) - self.detected_freqs[j, i, idx])) ]
            #             else : val = mode_freqs[mask][0]
            #         else : val = mode_freqs[mask]
            #         if len(mode_freqs[mask]) > 0: 
            #             self.detected_freqs[j, i, idx] = val 
                    
            # if idx > 25 : break # WIP - for tests with partial 5 AM data

        self.plot_frequencies(show=False)

    def plot_frequencies(self, show=False):
        """
        Plots detected frequencies in specified frequency ranges.

        Parameters:
            file_paths (list): List of file paths to process.
            location (str): Location name or identifier used in processing.
        """
        file_duration = 1/6 * self.batch_size # 10 minutes : 1 file duration in hours
        time = np.linspace(0, self.n_files*file_duration, self.n_files)

        # Create the plots
        visualizer = Visualizer(time, output_dir="PART_I/results")
        processing_method = "Welch"
        nperseg = self.pp_args['nperseg']
        folder_name = f"loc_{self.location}_wd{600*self.batch_size}_meth_{processing_method}_nperseg_{nperseg}_sigma_{self.pp_args['sigma']}"

        fig, ax = plt.subplots(self.n_channels, 1, figsize=(3.33*self.n_channels, 7), sharex=True)

        linestyles_list = ['-', '--', '-.', ':']
        linestyles = [linestyles_list[i % len(linestyles_list)] for i in range(self.n_methods)]
        colors_list = ['blue', 'green', 'red', 'black']  
        colors = [colors_list[i % len(colors_list)] for i in range(self.n_methods)]

        labels = np.array(["method 0", "method 1", "method 2 PSD", "PyOMA", "method 2 PP"])[self.methods]

        for i in range(self.n_channels):
            for j, (label, linestyle, c) in enumerate(zip(labels, linestyles, colors)):
                ax[i].plot(time, self.detected_freqs[i][j], label=label, color=c, linestyle=linestyle)  
            ax[i].set_ylabel('f (Hz)')
            ax[i].legend()       

        ax[-1].set_xlabel('Time [h]')

        fig.tight_layout()
        if show: fig.show()
        visualizer._save_figure(fig, "Detected_Frequencies", folder_name)
        print(folder_name)