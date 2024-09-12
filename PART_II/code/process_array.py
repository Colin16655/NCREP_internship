import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
from process_tw import ProcessTW
from utils import save_figure
from process_folder import ProcessFolder

class ProcessArray(ProcessFolder):
    def __init__(self, S, L, k, data_array, location, I):
        """
        Initialize with the time series data array and desired window size.

        Args:
            data_array (np.ndarray): Array containing time series data.
            S (int): The number of SDOs per time window.
            L (int): Batch size.
            scaling_factors (list): Scaling factors for data normalization.
            location (str): Location or identifier for the processing context.
        """
        # Replace DataLoader with the array data
        self.data_array = data_array
        self.p = self.data_array.shape[1]
        # plt.plot(self.data_array[:, 0])
        # plt.show()
        self.S = S
        self.L = L
        self.location = location
        self.folder_name = f"loc_{self.location}_S{S}_L_{L}_p_{self.p}"   
        self.num_batches = len(self.data_array) // self.L     
        self.NI_values = np.full(self.num_batches+1, np.nan)
        self.CB_values = np.full(self.num_batches+1, np.nan)
        self.DI_values = np.full(self.num_batches+1, np.nan)  
        self.I = I
        
        # Directly process the data array instead of using DataLoader
        self.process(k)

    def process(self, k):
        """
        Process the data array in batches, compute NI, CB, and DI, and store results.
        """
        analysis = ProcessTW(batch=None, S=self.S, p=self.p)
        
        num_batches = len(self.data_array) // self.L
        for idx in range(num_batches + 1):
            start_idx = idx * self.L
            end_idx = min((idx + 1) * self.L, len(self.data_array))

            for i, current_I in enumerate(self.I):
                if start_idx <= current_I <= end_idx: self.I[i] = idx

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
