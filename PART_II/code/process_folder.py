import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib.pyplot as plt
from data_loader2 import DataLoader
from process_tw import ProcessTW
from tqdm import tqdm
from utils import save_figure

class ProcessFolder:
    def __init__(self, S, L, k, selected_indices, folder_path, scaling_factors, location):
        """
        Initialize with the folder path containing time series data and the desired window size.

        Args:
            folder_path (str): Path to the folder containing time series data.
            window_size (int): The number of SDOs per time window.
        """
        self.loader = DataLoader(selected_indices, folder_path=folder_path, batch_size=L, scaling_factors=scaling_factors)
        # for f in self.loader.get_files_list(folder_path):
            # print(f)
        self.S = S
        self.p = len(selected_indices) # Number of sensors
        self.L = L
        self.NI_values = np.full(len(self.loader), np.nan)
        self.CB_values = np.full(len(self.loader), np.nan)
        self.DI_values = np.full(len(self.loader), np.nan)
        self.location = location
        self.folder_name = f"loc_{self.location}_S{S}_L_{L}_p_{self.p}"

        self.process(k)

    def process(self, k):
        """
        Process each batch, compute NI, CB, and DI, and store results.
        """
        analysis = ProcessTW(batch=None, S=self.S, p=self.p)

        for idx, batch in enumerate(tqdm(self.loader, desc="Processing Batches", unit="batch")):
            # apply Hamming window to the batch
            window = np.hamming(len(batch))
            for i in range(batch.shape[1]):  # Loop through each channel
                batch[:, i] = batch[:, i] * window
            
            analysis.batch = batch
            analysis.compute_Q()
            if idx >= self.S-1:
                analysis.compute_D()
                # if idx % 100 == 0: analysis.plot_D(self.folder_name, idx)
                analysis.apply_k_medoids(k)
                NI = analysis.compute_NI()
                self.NI_values[idx] = NI
            if idx >= self.S:
                CB = self.compute_CB(self.NI_values[max(idx-self.S+1, self.S-1):idx+1])
                DI = NI - CB
                self.CB_values[idx] = CB
                self.DI_values[idx] = DI

    def compute_CB(self, ni_values):
        """
        Calculate the upper confidence boundary for NI values.

        Args:
            ni_values (list): List of NI values.

        Returns:
            float: Upper confidence boundary.
        """
        mean = np.median(ni_values)
        
        inner_medians = np.zeros(len(ni_values))
        
        # Calculate the nested median of absolute differences
        for i, ni_i in enumerate(ni_values):
            differences = np.abs(ni_i - ni_values)
            inner_medians[i] = np.median(differences)
        
        # Outer median of the inner medians
        std = 1.1926 * np.median(inner_medians) / np.sqrt(self.S)

        degrees_of_freedom = self.S - 1
        t_value = t.ppf(.99, degrees_of_freedom)  # 99% confidence interval
        return mean + t_value * std

    def plot(self, filename=f"NI_CB_DI", T=600):
        """
        Plot the evolution of the NI, CB, NI+CB, DI indices over time.
        """
        if T==600 : T = T*len(self.NI_values) # for exp0
        fig, ax = plt.subplots(4, 1, figsize=(10, 10))
        # Share the x-axis among axes 1 to 4
        for i in range(1, 4):
            ax[i].sharex(ax[1])

        # for i in range(self.p):
            # ax[0].plot(data[:, i], color=colors[i], label=labels[i])
        time = np.linspace(0, T, len(self.NI_values))
        ax[0].plot(time, self.NI_values, color='k')
        ax[1].plot(time, self.CB_values, color='b', linestyle='-.')
        ax[2].plot(time, self.NI_values, color='k', label=('NI'))
        ax[2].plot(time, self.CB_values, color='b', linestyle='-.', label=('CB'))

        # Separate the indices and values for positive and negative DI values
        indices = np.arange(len(self.DI_values))
        positive_indices = indices[self.DI_values > 0]
        negative_indices = indices[self.DI_values < 0]

        positive_values = self.DI_values[self.DI_values > 0]
        negative_values = self.DI_values[self.DI_values < 0]

        # Generate the corresponding time values for positive and negative indices
        positive_time = time[positive_indices]
        negative_time = time[negative_indices]

        # Plot the positive DI values in red on the stem plot
        if len(positive_time) > 0: ax[3].stem(positive_time, positive_values, linefmt='r-', markerfmt='ro', basefmt=" ", use_line_collection=True)

        # Plot the negative DI values in green on the stem plot
        if len(negative_time) > 0: ax[3].stem(negative_time, negative_values, linefmt='g-', markerfmt='go', basefmt=" ", use_line_collection=True)

        # ax[0].set_ylabel(f'Accel. [mG]')
        ax[0].set_ylabel('NI')
        ax[1].set_ylabel('CB')
        ax[2].set_ylabel('NI & CB')
        ax[3].set_ylabel('DI')
        
        # ax[0].legend()
        ax[2].legend()

        ax[-1].set_xlabel('Time Window')
        fig.tight_layout()
        save_figure(fig, filename, self.folder_name, output_dir=r"PART_II/results", format='pdf')