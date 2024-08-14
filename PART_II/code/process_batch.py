import numpy as np
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids
from utils import save_figure

class ProcessBatch:
    def __init__(self, batch, S, p):
        self.batch = batch
        self.fft_batch = None
        self.Q = np.full((S, 2, 3, p), np.nan) # shape is (S, 2, 3, p)
        self.D = None
        self.Q_k = None
        self.medoid_indices_ = None
        self.NI = None
        self.Q_idx = 0
        self.S = S
        self.p = p
        self.mean_RMS = np.full((2, S), np.nan) # for compute_D

    def compute_Q(self):
        """
        Compute the Q array from time series data (batch).

        Returns:
            np.ndarray: Computed Q array.
        """
        if self.batch is None:
            raise ValueError("No batch is provided.")
        Q_time = np.percentile(self.batch, [25, 50, 75], axis=0)
        self.fft_batch = np.abs(np.fft.rfft(self.batch, axis=0))
        Q_freq = np.percentile(self.fft_batch, [25, 50, 75], axis=0)
        self.Q[self.Q_idx, :, :, :] = np.stack((Q_time, Q_freq), axis=0)

        # for compute_D
        self.mean_RMS[0, self.Q_idx] = np.mean([np.sqrt(np.mean(np.square(self.batch[:, l]))) for l in range(self.p)])
        self.mean_RMS[1, self.Q_idx] = np.mean([np.sqrt(np.mean(np.square(self.fft_batch[:, l]))) for l in range(self.p)])

        self.Q_idx = (self.Q_idx + 1) % self.S
        return self.Q

    def compute_D(self):
        """
        Compute the distance matrix D for the SDO (Q[., ., :, :]).

        Returns:
            np.ndarray: Computed distance matrix D.
        """
        self.D = np.zeros((self.S, self.S))
        for i in range(self.S):
            for j in range(i + 1, self.S):
                delta_T = np.array([np.linalg.norm(self.Q[i, 0, k, :] - self.Q[j, 0, k, :]) for k in range(3)])
                delta_F = np.array([np.linalg.norm(self.Q[i, 1, k, :] - self.Q[j, 1, k, :]) for k in range(3)])
                delta_T_norm = np.linalg.norm(delta_T) / max(self.mean_RMS[0, i], self.mean_RMS[0, j])
                delta_F_norm = np.linalg.norm(delta_F) / max(self.mean_RMS[1, i], self.mean_RMS[1, j])
                self.D[i, j] = self.D[j, i] = delta_T_norm + delta_F_norm
        return self.D

    def apply_k_medoids(self, k):
        """
        Apply the k-medoids clustering algorithm to the Q array using the distance matrix D.

        Args:
            k (int): Number of clusters.
        """
        if self.D is None:
            raise ValueError("Distance matrix D is not computed. Please run compute_D() first.")
        # Use precomputed distance matrix D for k-medoids clustering
        kmedoids = KMedoids(n_clusters=k, metric='precomputed', random_state=0)
        kmedoids.fit(self.D)
        # Update self.Q_k with the cluster centers
        self.Q_k = self.Q[kmedoids.medoid_indices_]
        self.medoid_indices_ = kmedoids.medoid_indices_
        # print("  -- choosen : ", kmedoids.medoid_indices_)
        return self.Q_k

    def compute_NI(self):
        """
        Compute the Novelty Index (NI) from the distance matrix D.

        Returns:
            float: Computed Novelty Index.
        """
        if self.Q_k is None:
            raise ValueError("Q_k is not computed. Please run apply_k_medoids() first.")
        self.NI = 0.0
        for i in range(len(self.medoid_indices_)):
            for j in range(i+1, len(self.medoid_indices_)):
                i = self.medoid_indices_[i]
                j = self.medoid_indices_[j]
                self.NI = max(self.NI, self.D[i, j])
        return self.NI

    def plot_D(self, folder_name, idx):
        """
        Plot the distance matrix D to visualize clustering requirements.

        Args:
            foalder_name (str): Name of the folder to save the plot.
        """
        if self.D is None:
            raise ValueError("Distance matrix D is not computed. Please run compute_NI() first.")
        fig, ax = plt.subplots(figsize=(8, 8))
        cax = ax.imshow(self.D, cmap='hot', vmin=0, vmax=2)
        cbar = fig.colorbar(cax)
        ax.set_xlabel("Object j")
        ax.set_ylabel("Object i")
        ax.set_xticks(range(0, self.S))
        ax.set_yticks(range(0, self.S))
        ax.set_xticklabels(range(1, self.S+1))
        ax.set_yticklabels(range(1, self.S+1))
        fig.tight_layout()
        save_figure(fig, f"D_{idx}", folder_name, format='png')