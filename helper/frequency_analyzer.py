import numpy as np
import scipy.signal as signal

class FrequencyAnalyzer:
    """
    A class to analyze the frequency components of signal data.

    Attributes:
        detrended_data (numpy.ndarray): The detrended and scaled sensor data.
        time (numpy.ndarray): The time vector corresponding to the sensor data.
        fft_data (numpy.ndarray): The FFT of the detrended sensor data.
        psds (numpy.ndarray): The Power Spectral Densities of the sensor data.
        psd_matrix (numpy.ndarray): The PSD matrix for selected sensors.
        U (numpy.ndarray): Left singular vectors from SVD.
        S (numpy.ndarray): Singular values from SVD.
        V (numpy.ndarray): Right singular vectors from SVD (V = U.T since the psd_matrix is symetric).
    """

    def __init__(self, detrended_data, time):
        """
        Initializes the FrequencyAnalyzer with detrended data and time vector.

        Parameters:
            detrended_data (numpy.ndarray): The detrended and scaled sensor data.
            time (numpy.ndarray): The time vector corresponding to the sensor data.
        """
        self.detrended_data = detrended_data
        self.time = time
        self.fft_data = None
        self.psd_matrix = None
        self.correlation_matrix = None
        self.U = None
        self.S = None
        self.V = None

    def compute_fft(self):
        """
        Computes the FFT of the detrended data.

        Returns:
            tuple: Frequencies and the FFT of the detrended data.
        """
        n_samples = self.detrended_data.shape[0]
        sampling_dt = np.mean(np.diff(self.time))
        frequencies = np.fft.rfftfreq(n_samples, d=sampling_dt)
        self.fft_data = np.fft.rfft(self.detrended_data, axis=0)
        return frequencies, self.fft_data

    def compute_psd_matrix(self, fs, selected_indices, nperseg=1024):
        """
        Computes the PSD matrix for selected sensors using Welch's method.

        Parameters:
            fs (float): Sampling frequency.
            selected_indices (list of int): Indices of the selected sensors.
            nperseg (int): Length of each segment for Welch's method.

        Returns:
            numpy.ndarray: The PSD matrix for the selected sensors.
        """
        self.psd_matrix = np.zeros((nperseg // 2 + 1, len(selected_indices), len(selected_indices)), dtype=complex)
        for i in range(len(selected_indices)):
            freqs, psd = signal.welch(self.detrended_data[:, selected_indices[i]], fs=fs, nperseg=nperseg)
            self.psd_matrix[:, i, i] = psd
        for i in range(len(selected_indices)):
            for j in range(i + 1, len(selected_indices)):
                self.psd_matrix[:, i, j] = signal.csd(self.detrended_data[:, selected_indices[i]], self.detrended_data[:, selected_indices[j]], fs=fs, nperseg=nperseg)[1]
                self.psd_matrix[:, j, i] = self.psd_matrix[:, i, j] # might be smarter to just compute the upper triangle
        return freqs, self.psd_matrix
    
        
    def compute_correlation_matrix(self, psd_matrix):
        """
        Computes the correlation matrix from the given PSD matrix.

        Parameters:
            psd_matrix (numpy.ndarray): The PSD matrix for which to compute the correlation matrix.

        Returns:
            numpy.ndarray: The correlation matrix.
        """ 
        self.correlation_matrix = np.zeros_like(psd_matrix)
        for i in range(len(psd_matrix)):
            psd_conjugate = np.conjugate(psd_matrix[i])
            NUM = psd_matrix[i] * psd_conjugate 
            diag = np.real(np.diagonal(psd_matrix[i]))
            DEN = np.outer(diag, diag)    
            self.correlation_matrix[i] = NUM / DEN
        return self.correlation_matrix

    def perform_svd(self, matrix):
        """
        Performs Singular Value Decomposition (SVD) on the PSD matrix.

        Returns:
            tuple: U, S, V matrices from SVD.
        """
        self.U, self.S, self.V = np.linalg.svd(matrix, full_matrices=False)
        return self.U, self.S, self.V
    
    def get_pp_index(self, correlation_matrix, S_corr):
        """
        Compute the PP index.

        Parameters:
            correlation_matrix (numpy.ndarray): The correlation matrix of shape (n_frqcies, n_sensors, n_sensors).
            S_corr (numpy.ndarray): The singular values corresponding to the correlation matrix of shape (n_frqcies, n_sensors).

        Returns:
            tuple: A tuple containing three numpy arrays:
                - P1 (numpy.ndarray): 1st index.
                - P2 (numpy.ndarray): 2nd index
                - P3 (numpy.ndarray): 3rd index.
        """
        P1 = np.zeros(len(correlation_matrix))
        P2 = np.zeros(len(correlation_matrix))
        P3 = np.zeros(len(correlation_matrix))
        for k in range(len(correlation_matrix)):
            P1[k] = np.sum([np.abs(1 / np.log(correlation_matrix[k, i, j])) for i in range(len(correlation_matrix[0])) for j in range(i+1, len(correlation_matrix[0]))])
            P2[k] = np.abs(1 / np.log(S_corr[k, 0]/len(correlation_matrix[0])))
            P3[k] = np.prod(1 / S_corr[k, 1:])
        return P1, P2, P3

    def identify_peaks(self, freqs, S, band=(8, 24), distance=10):
        """
        Identifies peaks in the signal S within a specified frequency band.

        Parameters:
            freqs (numpy.ndarray): Array of frequencies.
            S (numpy.ndarray): Signal data from which to identify peaks.
            band (tuple): A tuple specifying the frequency band (min, max) within which to find peaks.
            distance (int): Minimum distance between peaks (in terms of number of data points).

        Returns:
            numpy.ndarray: Indices of the peaks found within the specified frequency band.
        """
        # Extract indices within the specified frequency band
        ban_min, band_max = band
        min_idx, max_idx = self.extract_indices_within_band(freqs, ban_min, band_max)
        
        # Slice the frequency and signal arrays to the specified band
        band_freqs = freqs[min_idx:max_idx + 1]
        band_S = S[min_idx:max_idx + 1]
        
        # Identify peaks within the band
        peaks, _ = signal.find_peaks(band_S, distance=distance)
        
        # Adjust peak indices to match the original frequency array
        peaks = peaks + min_idx
        return peaks

    def extract_indices_within_band(self, freqs, ban_min, band_max):
        """
        Extracts the indices min_idx and max_idx such that the interval [freqs[min_idx], freqs[max_idx]]
        is within the range [ban_min, band_max].

        Parameters:
            freqs (numpy.ndarray): The array of frequencies.
            ban_min (float): The minimum frequency of the desired band.
            band_max (float): The maximum frequency of the desired band.

        Returns:
            tuple: (min_idx, max_idx) where freqs[min_idx] and freqs[max_idx] define the interval within the band.
        """
        # Ensure freqs is a numpy array
        freqs = np.array(freqs)
        
        # Find the indices where frequencies fall within the band
        min_idx = np.searchsorted(freqs, ban_min, side='left')
        max_idx = np.searchsorted(freqs, band_max, side='right') - 1
        
        # Check if indices are within the bounds of the frequency array
        if min_idx >= len(freqs) or max_idx >= len(freqs) or min_idx > max_idx:
            raise ValueError("The specified band does not overlap with the frequency array.")
        
        return min_idx, max_idx

    def identify_mode_shapes(self, frequencies, U, peaks):
        """
        Identifies the mode frequencies and the mode shapes at the peaks of the singular values.

        Parameters:
            frequencies (numpy.ndarray): The frequency vector.
            peaks (list of int): Indices of the peak frequencies in the singular values.
            bandwidth (int, optional): The bandwidth around the peak for averaging the mode shapes. Defaults to 5.

        Returns:
            tuple:
                numpy.ndarray: The identified mode frequencies.
                list of numpy.ndarray: The mode shapes corresponding to the identified mode frequencies.
        """
        mode_shapes = []
        mode_fqcies = []
        for peak in peaks:
            mode_shape = np.real(U[peak, :, 0])
            mode_shapes.append(mode_shape)
            mode_fqcies.append(frequencies[peak])
        return np.array(mode_fqcies), np.array(mode_shapes)
    
    def compute_mac(self, mode_shape1, mode_shape2):
        """
        Computes the Modal Assurance Criterion (MAC) between two mode shapes.

        Parameters:
            mode_shape1 (numpy.ndarray): The first mode shape.
            mode_shape2 (numpy.ndarray): The second mode shape.

        Returns:
            float: The MAC value between the two mode shapes.
        """
        return np.abs(np.dot(mode_shape1, mode_shape2.conj().T) / (np.linalg.norm(mode_shape1) * np.linalg.norm(mode_shape2)))
    

# def estimate_damping_and_mode_shapes(self, peaks, bandwidth=5):
#     """
#     Estimates the damping ratios and mode shapes from the identified spectral peaks.

#     Parameters:
#         peaks (numpy.ndarray): Indices of the identified peaks.
#         bandwidth (int): Number of frequency bins around the peak to consider for fitting.

#     Returns:
#         damping_ratios (list): Estimated damping ratios for each peak.
#         mode_shapes (list): Estimated mode shapes for each peak.
#     """
#     damping_ratios = []
#     mode_shapes = []
#     for peak in peaks:
#         lower_bound = max(0, peak - bandwidth)
#         upper_bound = min(len(self.frequencies), peak + bandwidth)
#         freq_band = self.frequencies[lower_bound:upper_bound]
#         sv_band = self.singular_values[lower_bound:upper_bound, 0]
#         time_corr = ifft(sv_band)
#         time = np.arange(len(time_corr)) / self.fs
#         popt, _ = curve_fit(self.fit_exponential_decay, time, np.abs(time_corr), maxfev=10000)
#         damping_ratio = popt[1] / (2 * np.pi * self.frequencies[peak])
#         damping_ratios.append(damping_ratio)
#         mode_shape = np.real(self.psd_matrix[peak, :, 0])
#         mode_shapes.append(mode_shape)
#     return damping_ratios, mode_shapes