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

    def __init__(self, selected_indices, folder_path, batch_size, n_mem, n_modes, pp_args, scaling_factors, methods=[1], ranges_display=[(8,13), (13,18), (18,24)]):
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
        self.n_mem = n_mem
        self.n_modes = n_modes
        self.pp_args = pp_args
        self.scaling_factors = scaling_factors
        self.loader = DataLoader(selected_indices, folder_path=folder_path, batch_size=batch_size, scaling_factors=scaling_factors)
        self.analyzer = ModalFrequencyAnalyzer()
        self.peak_picker = PeakPicker(self.analyzer)
        self.frequency_data = []
        self.methods = methods
        self.ranges_display = ranges_display

    def process(self):
        """
        Processes each batch of data from the folder, computes vibration frequencies, and stores them for plotting.

        Iterates through batches of data loaded from CSV files, uses the `ModalFrequencyAnalyzer` to compute
        vibration frequencies for each batch, and stores these frequencies in the `frequency_data` attribute.
        """
        # Initialize arrays to collect frequencies
        n_channels = len(self.ranges_display)
        n_methods = len(self.methods)
        n_files = len(self.loader)
        freqs = np.full((n_channels, n_methods, n_files), np.nan)

        band = (self.ranges_display[0][0], self.ranges_display[-1][-1])
        results = None

        for idx, (time, data) in enumerate(tqdm(self.loader, desc="Processing Batches", unit="batch")):
            self.analyzer.time = time
            self.analyzer.data = data

            # Computation to initialize analyzer
            self.analyzer.compute_psd_matrix(nperseg=self.pp_args['nperseg'])
            self.analyzer.compute_coherence_matrix()
            self.analyzer.perform_svd_coherence()
            self.analyzer.perform_svd_psd()
            self.analyzer.compute_pp_index()
            for i, method in enumerate(self.methods):
                # Compute frequencies (assuming this returns a numpy array of frequencies)
                if method == 0:
                    label = 'method 0'
                    peaks = self.peak_picker.identify_peaks_0(self.analyzer.P3, distance=self.pp_args['distance0']) 
                elif method == 1:
                    label = 'method 1'
                    peaks = self.peak_picker.identify_peaks_1(self.analyzer.P3, self.analyzer.S_psd[:, 0], distance=self.pp_args['distance1'], sigma=self.pp_args['sigma'], ranges_to_check=self.pp_args['ranges_to_check']) 
                elif method == 2 :
                    label = 'method 2'
                    peaks, results = self.peak_picker.identify_peaks_2(self.analyzer.S_psd[:, 0], self.analyzer.U_psd, band=band, distance=self.pp_args['distance2'], mac_threshold=self.pp_args['mac_threshold'], n_modes=self.n_modes, results_prev=results) # WIP 
                elif method == 3:
                    label = 'PyOMA'
                    peaks = self.peak_picker.identify_peaks_pyoma()
                else:
                    raise ValueError(f"Unsupported method {method}")
                
                if method != 3 : mode_freqs, mode_shapes = self.peak_picker.identify_mode_shapes(self.analyzer.U_psd, peaks)
                else : mode_freqs = peaks

                for j, band in enumerate(self.ranges_display):
                    band_min, band_max = band
                    mask = (mode_freqs >= band_min) & (mode_freqs < band_max)
                    if len(mode_freqs[mask]) > 0: freqs[j, i, idx] = mode_freqs[mask]

        print(freqs) # PLOT INSTEAD OF PRINTING - WIP