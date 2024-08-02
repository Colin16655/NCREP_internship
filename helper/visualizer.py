import matplotlib.pyplot as plt
import numpy as np
import os

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

    def plot_pp_index(self, freqs, P1, P2, P3, peaks, folder_name=""):
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
