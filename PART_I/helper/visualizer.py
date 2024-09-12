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
        self.colors = ['black', 'r', 'b', 'g']
        self.styles = ['-', '--', '-.', ':']

    def get_color_array(self, n):

        # Get the default color cycle from Matplotlib
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        # Repeat the color cycle if n is greater than the length of the default color cycle
        colors = [color_cycle[i % len(color_cycle)] for i in range(n)]
        return colors

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

    def _save_figure(self, figure, file_name, folder_name, format='pdf'):
        """
        Saves the figure in PDF formats.

        Parameters:
            figure (matplotlib.figure.Figure): The figure to save.
            file_name (str): The base name for the saved files.
            folder_name (str): The folder where the files will be saved.
        """
        output_path = self._create_output_dir(folder_name)
        if format == 'png' : figure.savefig(os.path.join(output_path, f"{file_name}.png"), bbox_inches='tight')
        else : figure.savefig(os.path.join(output_path, f"{file_name}.pdf"), format='pdf', bbox_inches='tight')
        plt.close(figure)

    def plot_data(self, data, filename, folder_name="", y_label="Amplitude", labels=None, show=False, figwidth=3.33):
        """
        Plots the time series data and saves the figure.

        Parameters:
            data (numpy.ndarray): The time series data to plot.
            filename (str): Filename of the plot.
            folder_name (str): Name of the folder where the figure will be saved. Defaults to "".
        """
        fig, ax = plt.subplots(len(data[0]), 1, sharex=True, figsize=(figwidth, 10/4*len(data[0])))
        for i in range(len(data[0])):
            if labels is not None : 
                ax[i].plot(self.time[int(3/8*len(self.time)) : int(4/8*len(self.time))], data[int(3/8*len(self.time)) : int(4/8*len(self.time)), i], label=labels[i])
                ax[i].legend()
            else : ax[i].plot(self.time, data[:, i])
            ax[i].set_ylabel(y_label)
            # ax[i].set_xlim(self.time[0], self.time[-1])
        ax[-1].set_xlabel("Time [s]")
        # fig.suptitle(filename)
        fig.tight_layout()
        self._save_figure(fig, filename, folder_name)
        if show: fig.show()

    def plot_fft(self, frequencies, fft_data, folder_name="", labels=None, show=False, band=(8,24)):
        """
        Plots the FFT results and saves the figures.

        This function generates two sets of plots:
        1. A grid of subplots showing the real part, imaginary part, and amplitude of the FFT data for each channel.
        2. A set of subplots showing the amplitude of the FFT data for a specific subset of channels (channels 3 to 6).

        Parameters:
            frequencies (numpy.ndarray): The frequency values corresponding to the FFT data.
            fft_data (numpy.ndarray): The FFT data to plot, with dimensions (num_frequencies, num_channels).
            folder_name (str): Name of the folder where the figures will be saved. Defaults to "" (current directory).
            labels (list of str or None): Optional list of labels for the channels. If provided, it should be the same length as the number of channels in `fft_data`.
            show (bool): If True, the figures will be displayed interactively. Defaults to False.

        Returns:
            None
        """
        n_sensors = len(fft_data[0])
        fig, ax = plt.subplots(n_sensors, 3, figsize=(15, 20/8*n_sensors))
        for i in range(n_sensors):
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
        fig.tight_layout()

        self._save_figure(fig, "FFT_complete_results", folder_name)

        fig, ax = plt.subplots(n_sensors, 1, figsize=(3.33, 2.5*n_sensors))
        for i in range(n_sensors):
            if labels is not None:
                ax[i].semilogy(frequencies, np.abs(fft_data[:, i]), label=labels[i])
                ax[i].legend()
            else : ax[i].semilogy(frequencies, np.abs(fft_data[:, i]))
            ax[i].set_ylabel("FFT - Amplitude")
            ax[i].set_xlim(band[0], band[1])
            ax[i].set_ylim(1e-3, 13)
        ax[-1].set_xlabel("Frequency [Hz]")
        fig.tight_layout()

        self._save_figure(fig, "FFT_results", folder_name)
        if show: fig.show()

    def plot_psd(self, frequencies, psd_matrix, folder_name="", labels=None, show=False, linear=True, band=(8,24)):
        """
        Plots the Power Spectral Density (PSD) and saves the figures.

        This function generates two plots:
        1. A semi-logarithmic plot showing the real part of the PSD data over frequency.
        2. A linear plot showing the real part of the PSD data over frequency.

        Parameters:
            frequencies (numpy.ndarray): Array of frequency values corresponding to the PSD data.
            psd_matrix (numpy.ndarray): 3D array containing the PSD data, with dimensions (num_frequencies, num_channels, num_channels).
            folder_name (str): Name of the folder where the figures will be saved. Defaults to "" (current directory).
            labels (list of str or None): Optional list of labels for the channels. If provided, it should match the number of channels in `psd_matrix`.
            show (bool): If True, the figures will be displayed interactively. Defaults to False.

        Returns:
            None
        """
        if linear: fig, ax = plt.subplots(2, 1, figsize=(5, 5))
        else: 
            fig, ax = plt.subplots(1, 1, figsize=(5, 2.5))
            ax = [ax]
        for i in range(len(psd_matrix[0])):
            if labels is not None:
                ax[0].semilogy(frequencies, np.real(psd_matrix[:, i, i]), label=labels[i])
                ax[0].legend(framealpha=0.5)
            else:
                ax[0].semilogy(frequencies, np.real(psd_matrix[:, i, i]))
            if linear:
                if labels is not None:
                    ax[1].plot(frequencies, np.real(psd_matrix[:, i, i]), label=labels[i])
                    ax[1].legend(framealpha=0.5)
                else:
                    ax[1].plot(frequencies, np.real(psd_matrix[:, i, i]))
        if linear :
            ax[1].set_ylabel("PSD")
            ax[1].set_xlim(band[0], band[1])
        ax[-1].set_xlabel("Frequency [Hz]")
        ax[0].set_ylabel("PSD")
        ax[0].set_xlim(band[0], band[1])
        fig.tight_layout()
        self._save_figure(fig, "PSD_results", folder_name)
        if show: fig.show()

    def plot_coherence(self, frequencies, coherence_matrix, peaks, folder_name, filename='Coherence', show=False):
        """
        Plots the coherence matrix (diagonal elements only) on a semi-logarithmic scale and saves the figures.

        Parameters:
            frequencies (numpy.ndarray): Array of frequency values corresponding to the coherence data.
            coherence_matrix (numpy.ndarray): 3D array containing the coherence data, with dimensions (num_frequencies, num_channels, num_channels).
            peaks (numpy.ndarray): Array containing the detected peak indices.
            folder_name (str): Name of the folder where the figures will be saved.
            filename (str): Name of the file to save the plot as. Defaults to 'Coherence'.
            show (bool): If True, the figures will be displayed interactively. Defaults to False.

        Returns:
            None
        """
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        colors = self.get_color_array(6)
        idx = 0
        names = ["1X", "1Y", "1Z", "2Z"]
        for i in range(len(coherence_matrix[0])):
            for j in range(i+1, len(coherence_matrix[0])):
                ax.semilogy(frequencies, np.real(coherence_matrix[:, i, j]), color=colors[idx], linewidth=1.5, label=f"{names[i]}-{names[j]}")
                idx += 1
        ax.semilogy(frequencies, np.ones_like(frequencies), linestyle='--', color='black')
        ax.set_ylabel("Coherence")
        ax.set_xlabel("f [Hz]")
        ax.set_xlim(8, 24)
        ax.legend(loc='lower center', ncol=3, framealpha=0.5)
        fig.tight_layout()

        self._save_figure(fig, filename, folder_name)
        if show:
            plt.show()

    def plot_sigmas(self, frequencies, S_PSD, peaks, folder_name="", sigma=8, filename='PSD_SVD_results', plot_li=False, plot_smooth=True, show=False, band=(8,24), legend=True, ax=None):
        """
        Plots the singular values of the PSD matrix as a function of frequency and saves the figure.

        Parameters:
            frequencies (numpy.ndarray): The frequency array.
            S_PSD (numpy.ndarray): The singular values array from the PSD matrix.
            peaks (numpy.ndarray): Indices of the peak points.
            folder_name (str): Name of the folder where the figure will be saved. Defaults to "".
            sigma (float): Standard deviation for Gaussian kernel. Defaults to 8.
            plot_li (bool): Whether to plot the additional linear plots. Defaults to False.
            show (bool): Whether to display the plot. Defaults to False.
        """
        num_singular_values = S_PSD.shape[1]  # Number of singular values

        # Define frequency range for x-limits
        freq_min, freq_max = band

        if plot_li:
            fig, ax = plt.subplots(2, 1, figsize=(10, 10))  # Adjusted figsize to accommodate additional plots
            ax0 = ax[0]
        else:
            if ax is not None: ax0 = ax
            else: fig, ax0 = plt.subplots(1, 1, figsize=(5.5, 3))

        colors = self.get_color_array(num_singular_values)

        min_val = np.inf
        max_val = -np.inf

        # Filter data within the specified frequency range
        freq_mask = (frequencies >= freq_min) & (frequencies <= freq_max)
        freq_filtered = frequencies[freq_mask]

        for i in range(num_singular_values):
            original_values = S_PSD[:, i]
            filtered_values = gaussian_filter1d(original_values, sigma=sigma)

            # Apply frequency mask to the original and filtered values
            original_values_filtered = original_values[freq_mask]
            filtered_values_filtered = filtered_values[freq_mask]

            # Update min and max values within the frequency range
            min_val = min(min_val, np.min(original_values_filtered), np.min(filtered_values_filtered))
            max_val = max(max_val, np.max(original_values_filtered), np.max(filtered_values_filtered))

            # Plot non-smoothed curves
            if plot_smooth: ax0.semilogy(freq_filtered, original_values_filtered, label=f'$\sigma_{i+1}$ (Raw)', linestyle='dotted', color=colors[i])
            else: 
                ax0.semilogy(freq_filtered, original_values_filtered, label=f'$\sigma_{i+1}$ (Raw)', linestyle='solid', color=colors[i])
            # Plot smoothed curves
            if plot_smooth: ax0.semilogy(freq_filtered, filtered_values_filtered, linestyle='solid', color=colors[i])

            if plot_li:
                ax[1].plot(freq_filtered, original_values_filtered, label=f'$\sigma_{i+1}$ (Raw)', linestyle='dotted', color=colors[i])
                ax[1].plot(freq_filtered, filtered_values_filtered, linestyle='solid', color=colors[i])

        # Scatter plot for first sigma at the peak points
        if peaks is not None:
            peak_values = S_PSD[peaks.astype(int), 0]
            ax0.scatter(frequencies[peaks.astype(int)], peak_values, color='black', marker='x', label='Peaks')

        # Update min and max values to include peak values and slightly increase max_val to avoid cropping
        if peaks is not None : min_val = min(min_val, np.min(peak_values))
        if peaks is not None : max_val = max(max_val, np.max(peak_values))

        # Increase max_val slightly to avoid cropping the peaks
        y_margin = (max_val - min_val) * 0.2  # 20% margin
        max_val += y_margin

        # Set y-limits based on min and max values within the frequency range
        # ax0.set_ylim(min_val, max_val)
        ax0.set_ylabel("Singular Value")
        if plot_li:
            ax[1].set_ylim(min_val, max_val)
            ax[1].set_xlabel("Frequency [Hz]")
            ax[1].set_ylabel("Singular Value")
            ax[1].set_xlim(freq_min, freq_max)
            ax[1].legend(framealpha=0.5)
        else:
            ax0.set_xlim(freq_min, freq_max)
            if ax is None: ax0.set_xlabel("Frequency [Hz]")

        if legend: ax0.legend(framealpha=0.5)
        if ax is None: 
            fig.tight_layout()
            self._save_figure(fig, filename, folder_name)
            if show: fig.show()
        
    def plot_pp_index(self, freqs, PPS, peaks, folder_name="", filename='PP_indices_results', 
                      sigma=14, plot_smooth=True, show=False, band=(8,24), ax=None, legend=True, style='-'):
        """
        Plots the PP indices as a function of frequency and saves the figure.

        Parameters:
            freqs (numpy.ndarray): The frequency array.
            PPS (list of numpy.ndarray): List of PP indices.
            peaks (numpy.ndarray): Indices of the peak points.
            folder_name (str): Name of the folder where the figure will be saved. Defaults to "".
            sigma (float): Standard deviation for Gaussian kernel. Defaults to 14.
            show (bool): Whether to display the plot. Defaults to False.
        """
        num_singular_values = len(PPS)  # Number of PP indices
        colors = self.get_color_array(num_singular_values)  # Get colors for each PP index

        # Apply Gaussian smoothing
        PPS_smooth = [gaussian_filter1d(P, sigma=sigma) for P in PPS]

        if ax is None : fig, ax = plt.subplots(figsize=(7, 4))

        # Plot non-smoothed and smoothed curves
        for i, P in enumerate(PPS):
            if plot_smooth: label_raw = f"$P_{i+1}$ (Raw)"
            else: label_raw = f"$P_{i+1}$"
            color = colors[i % len(colors)]
            if plot_smooth: ax.semilogy(freqs, P, label=label_raw, linestyle='dotted', color=color)
            else: ax.semilogy(freqs, P, label=label_raw, linestyle='solid', color=color)
            if plot_smooth: ax.semilogy(freqs, PPS_smooth[i], linestyle='solid', color=color)

        # Scatter plot for the peaks (using the first PP index for peaks visualization)
        if peaks is not None:
            if len(PPS) > 0:
                ax.scatter(freqs[peaks.astype(int)], PPS[-1][peaks.astype(int)], color='black', marker='x', label='Peaks')

        ax.set_ylabel("PP index", fontsize=18)
        ax.set_xlabel("Frequency [Hz]", fontsize=18)
        if legend : ax.legend(framealpha=0.5)
        ax.set_xlim(band[0], band[1])
        ax.grid(True)

        if ax is None:
            fig.tight_layout()
            self._save_figure(fig, filename, folder_name)
            if show: fig.show()

    def plot_PCA(self, data, folder_name="", show=False):
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
        if show: fig.show()

    def plot_MAC_matrix(self, MAC, mode_frequency, peaks, folder_name="", filename='MAC_matrix', show=False):
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
        if show: plt.show()
        self._save_figure(fig, filename, folder_name, format='png')
