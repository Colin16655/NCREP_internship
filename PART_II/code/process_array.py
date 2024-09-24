import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
from process_tw import ProcessTW
from utils import save_figure
from process_folder import ProcessFolder

class ProcessArray(ProcessFolder):
    """
    Class for the statistical analysis of a time series data array."""
    def __init__(self, S, L, k, data_array):
        """
        Initialize with the time series data array and desired window size.

        Args:
            S (int): The number of SDOs (Statistical Deformation Operators) per time window.
            L (int): Batch size for processing.
            k (int): Parameter used in the k-medoids clustering algorithm.
            data_array (np.ndarray): Array containing time series data.
        """
        # Replace DataLoader with the array data
        self.data_array = data_array
        self.p = self.data_array.shape[1]
        self.S = S
        self.L = L
        self.num_batches = len(self.data_array) // self.L     
        self.NI_values = np.full(self.num_batches+1, np.nan)
        self.CB_values = np.full(self.num_batches+1, np.nan)
        self.DI_values = np.full(self.num_batches+1, np.nan)  
        
        # Directly process the data array instead of using DataLoader
        self.process(k)

    def process(self, k):
        """
        Process the data array in batches, compute NI, CB, and DI, and store results.

        Args:
            k (int): Parameter used in the k-medoids clustering algorithm.
        """
        analysis = ProcessTW(batch=None, S=self.S, p=self.p)
        
        num_batches = len(self.data_array) // self.L
        for idx in range(num_batches):
            start_idx = idx * self.L
            end_idx = (idx + 1) * self.L

            batch = self.data_array[start_idx:end_idx]

            analysis.batch = batch
            analysis.compute_Q()
            if idx >= self.S - 1:
                analysis.compute_D()
                # analysis.plot_D("trash", idx)
                analysis.apply_k_medoids(k=k)
                NI = analysis.compute_NI()
                self.NI_values[idx] = NI
            if idx >= self.S:
                CB = self.compute_CB(self.NI_values[max(idx-self.S+1, self.S-1):idx+1])
                DI = NI - CB
                self.CB_values[idx] = CB
                self.DI_values[idx] = DI
