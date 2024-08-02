import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d


class Visualizer:
    def __init__(self, time, output_dir="figures"):
        """
        Initializes the Visualizer with time, frequencies, and an optional output directory.

        Parameters:
            time (numpy.ndarray): Time array for the time series data.
            frequencies (numpy.ndarray): Frequencies array for the FFT analysis.
            output_dir (str): Base directory where figures will be saved. Default is "figures".
        """
        self.time = time
        self.output_dir = output_dir

    def _create_output_dir(self, folder_name):
        """
        Creates a directory for saving figures based on the provided folder name.

        Parameters:
            folder_name (str): Name of the folder to create.
        """
        path = os.path.join(self.output_dir, folder_name)
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def _save_figure(self, figure, file_name, folder_name):
        """
        Saves the figure in PDF formats.

        Parameters:
            figure (matplotlib.figure.Figure): The figure to save.
            file_name (str): The base name for the saved files.
            folder_name (str): The folder where the files will be saved.
        """
        output_path = self._create_output_dir(folder_name)
        figure.savefig(os.path.join(output_path, f"{file_name}.pdf"), format='pdf', bbox_inches='tight')
        plt.close(figure)

    def plot_data(self, data, filename, folder_name="", y_label="Amplitude", labels=None):
        """
        Plots the time series data and saves the figure.

        Parameters:
            data (numpy.ndarray): The time series data to plot.
            filename (str): Filename of the plot.
            folder_name (str): Name of the folder where the figure will be saved. Defaults to "".
        """
        fig, ax = plt.subplots(len(data[0]), 1, sharex=True, figsize=(10, 20))
        for i in range(len(data[0])):
            if labels is not None : 
                ax[i].plot(self.time, data[:, i], label=labels[i])
                ax[i].legend()
            else : ax[i].plot(self.time, data[:, i])
            ax[i].set_ylabel(y_label)
            ax[i].set_xlim(self.time[0], self.time[-1])
        ax[-1].set_xlabel("Time [s]")
        # fig.suptitle(filename)
        fig.tight_layout()

        self._save_figure(fig, filename, folder_name)

    def plot_fft(self, frequencies, fft_data, folder_name="", labels=None):
        """
        Plots the FFT results and saves the figure.

        Parameters:
            fft_data (numpy.ndarray): FFT data to plot.
            folder_name (str): Name of the folder where the figure will be saved. Defaults to "".
        """
        fig, ax = plt.subplots(8, 3, figsize=(15, 20))
        for i in range(8):
            if labels is not None:
                ax[i, 0].plot(frequencies, np.real(fft_data[:, i]), label=labels[i])
                ax[i, 0].legend()
            else : ax[i, 0].plot(frequencies, np.real(fft_data[:, i]))
            ax[i, 0].set_ylabel("FFT - Re")
            if labels is not None:
                ax[i, 1].plot(frequencies, np.imag(fft_data[:, i]), label=labels[i])
                ax[i, 1].legend()
            else : ax[i, 1].plot(frequencies, np.imag(fft_data[:, i]))
            ax[i, 1].set_ylabel("FFT - Im")
            if labels is not None:
                ax[i, 2].plot(frequencies, np.abs(fft_data[:, i]), label=labels[i])
                ax[i, 2].legend()
            else : ax[i, 2].plot(frequencies, np.abs(fft_data[:, i]))
            ax[i, 2].set_ylabel("FFT - Amplitude")
        ax[-1, 0].set_xlabel("Frequency [Hz]")
        ax[-1, 1].set_xlabel("Frequency [Hz]")
        ax[-1, 2].set_xlabel("Frequency [Hz]")
        plt.tight_layout()

        self._save_figure(fig, "FFT_results", folder_name)

    def plot_psd(self, frequencies, psd_matrix, folder_name="", labels=None):
        """
        Plots the Power Spectral Density (PSD) and saves the figure.

        Parameters:
            psd_matrix (numpy.ndarray): PSD matrix data.
            folder_name (str): Name of the folder where the figure will be saved. Defaults to "".
        """
        fig, ax = plt.subplots(2, 1, figsize=(10, 10))
        for i in range(len(psd_matrix[0])):
            if labels is not None:
                ax[0].semilogy(frequencies, np.real(psd_matrix[:, i, i]), label=labels[i])
                ax[0].legend()
            else:
                ax[0].semilogy(frequencies, np.real(psd_matrix[:, i, i]))
            if labels is not None:
                ax[1].plot(frequencies, np.real(psd_matrix[:, i, i]), label=labels[i])
                ax[1].legend()
            else:
                ax[1].plot(frequencies, np.real(psd_matrix[:, i, i]))
        ax[1].set_xlabel("Frequency [Hz]")
        ax[0].set_ylabel("PSD")
        ax[1].set_ylabel("PSD")
        ax[0].set_xlim(8, 24)
        ax[1].set_xlim(8, 24)
        fig.tight_layout()
        self._save_figure(fig, "PSD_results", folder_name)

    def plot_sigmas(self, frequencies, S_PSD, S_corr, folder_name=""):
        """
        Plots the singular values of the PSD matrix as a function of frequency and saves the figure.

        Parameters:
            singular_values (numpy.ndarray): The singular values array from SVD.
            frequencies (numpy.ndarray): The frequency array.
            filename (str): Filename of the plot.
            folder_name (str): Name of the folder where the figure will be saved. Defaults to "".
        """
        num_singular_values = S_PSD.shape[1]  # Number of singular values
        fig, ax = plt.subplots(2, 2, figsize=(20, 10))
        for k, singular_values in enumerate([S_PSD, S_corr]):
            for i in range(num_singular_values):
                ax[0, k].semilogy(frequencies, singular_values[:, i], label=f'sigma {i+1}')
                ax[1, k].plot(frequencies, singular_values[:, i], label=f'sigma {i+1}')
            
            ax[1, k].set_xlabel("Frequency [Hz]")
            ax[0, k].set_ylabel("Singular Value")
            ax[1, k].set_ylabel("Singular Value")
            ax[0, k].set_xlim(8, 24)
            ax[1, k].set_xlim(8, 24)
            ax[0, k].legend()
            ax[1, k].legend()
        fig.suptitle("Left : PSD, Right : Correlation", fontsize=20)
        fig.tight_layout()
        self._save_figure(fig, "SVD_results", folder_name)

    def plot_pp_indexc(self, freqs, P1, P2, P3, peaks, folder_name=""):
        """
        Plots the PP indices as a function of frequency and saves the figure.

        Parameters:
            freqs (numpy.ndarray): The frequency array.
            P1 (numpy.ndarray): The first PP index.
            P2 (numpy.ndarray): The second PP index.
            P3 (numpy.ndarray): The third PP index.
            folder_name (str): Name of the folder where the figure will be saved. Defaults to "".
        """
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.semilogy(freqs, P1, label="P1")
        ax.semilogy(freqs, P2, label="P2")
        ax.semilogy(freqs, P3, label="P3")

        # Scatter plot for P3 at the peak points
        ax.scatter(freqs[peaks], P3[peaks], color='red', marker='x', label='Peaks')

        ax.set_ylabel("PP index")
        ax.set_xlabel("Frequency [Hz]")
        ax.legend()
        ax.set_xlim(8, 24)

        fig.tight_layout()
        self._save_figure(fig, "PP_indices_results", folder_name)

    def plot_pp_index(self, freqs, P1, P2, P3, peaks, folder_name="", sigma=14):
        """
        Plots the PP indices as a function of frequency and saves the figure.

        Parameters:
            freqs (numpy.ndarray): The frequency array.
            P1 (numpy.ndarray): The first PP index.
            P2 (numpy.ndarray): The second PP index.
            P3 (numpy.ndarray): The third PP index.
            peaks (numpy.ndarray): Indices of the peak points.
            folder_name (str): Name of the folder where the figure will be saved. Defaults to "".
            sigma (float): Standard deviation for Gaussian kernel. Defaults to 2.
        """
        # Apply Gaussian smoothing
        P1_smooth = gaussian_filter1d(P1, sigma=sigma)
        P2_smooth = gaussian_filter1d(P2, sigma=sigma)
        P3_smooth = gaussian_filter1d(P3, sigma=sigma)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot non-smoothed curves
        ax.semilogy(freqs, P1, label="P1 (Raw)", linestyle='dotted', color='orange')
        ax.semilogy(freqs, P2, label="P2 (Raw)", linestyle='dotted', color='green')
        ax.semilogy(freqs, P3, label="P3 (Raw)", linestyle='dotted', color='blue')

        # Plot smoothed curves
        ax.semilogy(freqs, P1_smooth, label="P1 (Smoothed)", linestyle='solid', color='orange')
        ax.semilogy(freqs, P2_smooth, label="P2 (Smoothed)", linestyle='solid', color='green')
        ax.semilogy(freqs, P3_smooth, label="P3 (Smoothed)", linestyle='solid', color='blue')

        # Scatter plot for P3 at the peak points
        ax.scatter(freqs[peaks], P3[peaks], color='red', marker='x', label='Peaks')

        ax.set_ylabel("PP index")
        ax.set_xlabel("Frequency [Hz]")
        ax.legend()
        ax.set_xlim(8, 24)
        ax.grid(True)

        fig.tight_layout()
        self._save_figure(fig, "PP_indices_results", folder_name)

    def plot_PCA(self, data, folder_name=""):
        # Apply PCA
        pca = PCA(n_components=3)  # For 3D visualization
        principal_components_3d = pca.fit_transform(data)

        # Extract the first 2 principal components for 2D visualization
        pca_2d = PCA(n_components=2)
        principal_components_2d = pca_2d.fit_transform(data)

        # Plotting 3D PCA
        fig = plt.figure(figsize=(12, 6))

        # 3D plot
        ax = fig.add_subplot(121, projection='3d')
        ax.scatter(principal_components_3d[:, 0], principal_components_3d[:, 1], principal_components_3d[:, 2], c='r', marker='o')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title('3D PCA of Mode Shapes')

        # 2D plot
        ax2 = fig.add_subplot(122)
        ax2.scatter(principal_components_2d[:, 0], principal_components_2d[:, 1], c='b', marker='o')
        ax2.set_xlabel('PC1')
        ax2.set_ylabel('PC2')
        ax2.set_title('2D PCA of Mode Shapes')

        fig.tight_layout()
        self._save_figure(fig, "PCA", folder_name)

    def plot_MAC_matrix(self, MAC, mode_frequency, peaks, folder_name=""):
        # Create the plot
        fig, ax = plt.subplots(figsize=(8, 8))
        cax = ax.imshow(MAC, cmap='viridis')

        # Add color bar
        cbar = fig.colorbar(cax)
        cbar.set_label('MAC Value')

        # Set ticks and labels based on the modal frequencies
        ax.set_xticks(np.arange(len(peaks)))
        ax.set_yticks(np.arange(len(peaks)))
        ax.set_xticklabels([f"{freq:.2f} Hz" for freq in mode_frequency])
        ax.set_yticklabels([f"{freq:.2f} Hz" for freq in mode_frequency])

        # Rotate the tick labels for better readability if needed
        plt.xticks(rotation=45)

        # Add titles and labels
        ax.set_title("MAC Matrix")
        ax.set_xlabel("Mode Frequency [Hz]")
        ax.set_ylabel("Mode Frequency [Hz]")

        # Show plot
        fig.tight_layout()
        self._save_figure(fig, "MAC_matrix", folder_name)