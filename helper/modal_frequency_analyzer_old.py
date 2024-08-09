import numpy as np
import scipy.signal as signal
from scipy.ndimage import gaussian_filter1d
import math
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


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

    def __init__(self, detrended_data,=None time=None):
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
        self.coherence_matrix = None
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
                self.psd_matrix[:, j, i] = np.conj(self.psd_matrix[:, i, j]) # might be smarter to just compute the upper triangle
        return freqs, self.psd_matrix
        
    def compute_coherence_matrix(self, psd_matrix):
        """
        Computes the coherence matrix from the given PSD matrix.

        Parameters:
            psd_matrix (numpy.ndarray): The PSD matrix for which to compute the coherence matrix.

        Returns:
            numpy.ndarray: The coherence matrix.
        """ 
        self.coherence_matrix = np.zeros_like(psd_matrix)
        for i in range(len(psd_matrix)):
            psd_conjugate = np.conjugate(psd_matrix[i])
            NUM = psd_matrix[i] * psd_conjugate 
            diag = np.real(np.diagonal(psd_matrix[i]))
            DEN = np.outer(diag, diag)    
            self.coherence_matrix[i] = NUM / DEN
        return self.coherence_matrix

    def perform_svd(self, matrix):
        """
        Performs Singular Value Decomposition (SVD) on the PSD matrix.

        Returns:
            tuple: U, S, V matrices from SVD.
        """
        self.U, self.S, self.V = np.linalg.svd(matrix, full_matrices=False)
        return self.U, self.S, self.V
    
    def get_pp_index(self, coherence_matrix, S_corr):
        """
        Compute the PP index.

        Parameters:
            coherence_matrix (numpy.ndarray): The coherence matrix of shape (n_frqcies, n_sensors, n_sensors).
            S_corr (numpy.ndarray): The singular values corresponding to the coherence matrix of shape (n_frqcies, n_sensors).

        Returns:
            tuple: A tuple containing three numpy arrays:
                - P1 (numpy.ndarray): 1st index.
                - P2 (numpy.ndarray): 2nd index
                - P3 (numpy.ndarray): 3rd index.
        """
        P1 = np.zeros(len(coherence_matrix))
        P2 = np.zeros(len(coherence_matrix))
        P3 = np.zeros(len(coherence_matrix))
        for k in range(len(coherence_matrix)):
            P1[k] = np.sum([np.abs(1 / np.log(coherence_matrix[k, i, j])) for i in range(len(coherence_matrix[0])) for j in range(i+1, len(coherence_matrix[0]))])
            P2[k] = np.abs(1 / np.log(S_corr[k, 0]/len(coherence_matrix[0])))
            P3[k] = np.prod(1 / S_corr[k, 1:])
        return P1, P2, P3

    def identify_peaks(self, freqs, S, band=(8, 24), distance=10, sigma=14):
        """
        Identifies peaks in the signal S within a specified frequency band, and refines the peak positions 
        by checking the non-smoothed data within a 1 Hz range before and after each identified peak.

        Parameters:
            freqs (numpy.ndarray): Array of frequencies.
            S (numpy.ndarray): Signal data from which to identify peaks.
            band (tuple): A tuple specifying the frequency band (min, max) within which to find peaks.
            distance (int): Minimum distance between peaks (in terms of number of data points).
            sigma (float): Standard deviation for Gaussian smoothing.

        Returns:
            numpy.ndarray: Indices of the refined peaks found within the specified frequency band.
        """
        # Extract indices within the specified frequency band
        ban_min, band_max = band
        min_idx, max_idx = self.extract_indices_within_band(freqs, ban_min, band_max)
        
        # Slice the frequency and signal arrays to the specified band
        band_freqs = freqs[min_idx:max_idx + 1]
        band_S = S[min_idx:max_idx + 1]
        
        # Smooth the signal and identify peaks in the smoothed signal
        S_smooth = gaussian_filter1d(band_S, sigma=sigma)
        smooth_peaks, _ = signal.find_peaks(S_smooth, distance=distance)
        
        # Convert 1 Hz range to number of samples
        num_samples_1hz = int(np.round(1 / (freqs[1] - freqs[0])))
       
        refined_peaks = []
        for peak in smooth_peaks:
            # Calculate the range to check in the original signal
            peak_freq = band_freqs[peak]
            search_range = (max(peak - num_samples_1hz, 0), min(peak + num_samples_1hz, len(band_S) - 1))

            # Extract the relevant section from the original signal
            original_section = S[min_idx:max_idx + 1][search_range[0]:search_range[1] + 1]
            original_freqs_section = band_freqs[search_range[0]:search_range[1] + 1]

            # Find the peak in the original data section
            refined_peak = np.argmax(original_section) #signal.find_peaks(original_section, distance=distance)
            
            # Map new peak back to the global index
            refined_peaks.append(search_range[0] + refined_peak + min_idx)

        # Convert list to numpy array and ensure unique peaks
        refined_peaks = np.unique(np.array(refined_peaks))
        
        return refined_peaks

    def identify_peaks_bis(self, freqs, PP, PSD_sigma, band=(8, 24), distance=10, sigma=14, curvature_threshold=-0.004, ranges_to_check=[(10.5, 12.5), (15.5, 17.5), (21.5, 23.5)]):
        """
        Identifies peaks in the PP and PSD_sigma signals within a specified frequency band, refines the peak positions 
        by checking the non-smoothed data within the given ranges, and selects the peak with the largest curvature.

        Parameters:
            freqs (numpy.ndarray): Array of frequencies.
            PP (numpy.ndarray): Signal data from which to identify peaks for PP.
            PSD_sigma (numpy.ndarray): Signal data from which to identify peaks for PSD_sigma.
            band (tuple): A tuple specifying the frequency band (min, max) within which to find peaks.
            distance (int): Minimum distance between peaks (in terms of number of data points).
            sigma (float): Standard deviation for Gaussian smoothing.
            curvature_threshold (float): Threshold for the curvature to consider a peak.
            ranges_to_check (list of tuples): List of frequency ranges to check for peak refinement.

        Returns:
            numpy.ndarray: List of indices of the refined peaks, selecting the best one from either PP or PSD_sigma.
        """

        def get_peaks_and_curvatures(signal_data):
            # Extract indices within the specified frequency band
            band_min, band_max = band
            min_idx = np.searchsorted(freqs, band_min)
            max_idx = np.searchsorted(freqs, band_max)
            band_freqs = freqs[min_idx:max_idx + 1]
            band_S = signal_data[min_idx:max_idx + 1]
            
            # Smooth the signal and identify peaks in the smoothed signal
            S_smooth = gaussian_filter1d(band_S, sigma=sigma)
            smooth_peaks, _ = signal.find_peaks(S_smooth, distance=distance)

            # Compute the curvature of the smoothed signal
            curvature = np.gradient(np.gradient(S_smooth))
            
            # Refine peaks within specified ranges
            refined_peaks = []
            curvatures = []
            for peak in smooth_peaks:
                # if curvature[peak] < curvature_threshold:
                # print('  -- ', peak)
                peak_freq = band_freqs[peak]
                for r in ranges_to_check:
                    if r[0] <= peak_freq <= r[1]:
                        # Extract the original signal within the range
                        range_min_idx = np.searchsorted(freqs, r[0])
                        range_max_idx = np.searchsorted(freqs, r[1])
                        original_section = signal_data[range_min_idx:range_max_idx + 1]
                        # Find the peak in the original data section
                        if len(original_section) > 0:
                            local_peak = np.argmax(original_section)
                            refined_peak_idx = range_min_idx + local_peak
                            refined_peaks.append(refined_peak_idx)
                            curvatures.append(curvature[peak])

            return np.array(refined_peaks), np.array(curvatures)

        # Get refined peaks and curvatures for both PP and PSD_sigma
        refined_peaks_PP, curvatures_PP = get_peaks_and_curvatures(PP)
        # print("refined_peaks_PP", refined_peaks_PP, freqs[refined_peaks_PP])
        refined_peaks_PSD_sigma, curvatures_PSD_sigma = get_peaks_and_curvatures(PSD_sigma)
        # print("refined_peaks_PSD_sigma", refined_peaks_PSD_sigma, freqs[refined_peaks_PSD_sigma])
        # Initialize a list to store the best peaks for each range
        selected_peaks = []

        # For each range, select the peak with the highest curvature from both PP and PSD_sigma
        for i, r in enumerate(ranges_to_check):
            # print('r = ', r, ranges_to_check)
            range_min_freq, range_max_freq = r
            peaks_in_range_PP = []
            curvatures_in_range_PP = []
            peaks_in_range_PSD_sigma = []
            curvatures_in_range_PSD_sigma = []
            
            # Get PP peaks and curvatures within the current range
            for i, peak in enumerate(refined_peaks_PP):
                if range_min_freq <= freqs[peak] <= range_max_freq:
                    peaks_in_range_PP.append(peak)
                    curvatures_in_range_PP.append(curvatures_PP[i])
            
            # Get PSD_sigma peaks and curvatures within the current range
            for i, peak in enumerate(refined_peaks_PSD_sigma):
                # print(' -t', range_min_freq, freqs[peak], range_max_freq)
                if range_min_freq <= freqs[peak] <= range_max_freq:
                    # print('   oo ', peak)
                    peaks_in_range_PSD_sigma.append(peak)
                    curvatures_in_range_PSD_sigma.append(curvatures_PSD_sigma[i])
            
            # Combine peaks and curvatures from both signals
            all_peaks = np.array(peaks_in_range_PP + peaks_in_range_PSD_sigma)
            all_curvatures = np.array(curvatures_in_range_PP + curvatures_in_range_PSD_sigma)

            # Select the peak with the highest curvature within the current range
            if len(all_peaks) > 0:
                best_peak = all_peaks[np.argmax(all_curvatures)]
                selected_peaks.append(best_peak)
                # ranges_to_check[i] = (math.floor(freqs[best_peak]), math.ceil(freqs[best_peak]))
                
        return np.unique(np.array(selected_peaks)), ranges_to_check
    
    def identify_peaks_bis_bis(self, freqs, S, U, band=(8, 24), distance=2, sigma=14, mac_threshold=0.9, n_modes=4, time=0, results_prev=None, 
                               p=None, dt=None):
        """
        Detects all local maxima within a given frequency band, computes the MAC matrix using provided eigenvectors,
        groups peaks based on MAC threshold, and refines peak locations by fitting bell curves.

        Parameters:
            freqs (array-like): Array of frequency values.
            S (array-like): Signal values corresponding to the frequencies.
            U (array-like): Array of eigenvectors (mode shapes) for the frequencies.
            band (tuple): Frequency band within which to detect peaks (default is (8, 24)).
            distance (int): Minimum distance between peaks (default is 10).
            sigma (int): Standard deviation for Gaussian kernel to smooth the signal (default is 14).
            mac_threshold (float): Threshold for MAC value to consider peaks as the same mode (default is 0.8).

        Returns:
            selected_peaks (array-like): Indices of the final refined peaks.
        """
        
        # Step 1: Detect all local maxima in the raw PSD array
        band_min_idx = np.searchsorted(freqs, band[0], side='left')
        band_max_idx = np.searchsorted(freqs, band[1], side='right')
        raw_peaks, _ = signal.find_peaks(S[band_min_idx:band_max_idx], distance=distance)
        raw_peaks += band_min_idx  # Adjust peak indices back to original array indices
        raw_peaks = np.linspace(band_min_idx, band_max_idx, num=band_max_idx-band_min_idx, dtype=int)
        # Step 2: Compute the MAC matrix for the detected peaks using eigenvectors U
        mode_shapes = U[raw_peaks, :, 0]
        MAC = np.zeros((len(raw_peaks), len(raw_peaks)))
        for i in range(len(raw_peaks)):
            for j in range(i, len(raw_peaks)):  # Ensure we don't recompute MAC[i, j] and MAC[j, i]
                MAC[i, j] = MAC[j, i] = self.compute_mac(mode_shapes[i], mode_shapes[j])

        # Apply thresholding to the MAC matrix: set values below threshold to 0
        MAC_modified = np.where(MAC >= mac_threshold, MAC, 0)
        
        # Step 3: Identify which peaks belong to the same mode using a MAC threshold
        modes = []
        used_indices = set()
        for i in range(len(raw_peaks)):
            if i in used_indices:
                continue
            mode_group = [i]
            for j in range(i + 1, len(raw_peaks)):
                if MAC[i, j] > mac_threshold:
                    mode_group.append(j)
                    used_indices.add(j)
                else:
                    break
            modes.append(mode_group)
            # print("\n",i,mode_group)
        
        # Step 4: Obtain the frequency range for each mode, ignore single peaks (!if distance is too high -> risk of missing the peak!)
        mode_ranges = []
        for mode_group in modes:
            if len(mode_group) > 1:
                min_idx = min(raw_peaks[mode_group])
                max_idx = max(raw_peaks[mode_group])
                mode_ranges.append((min_idx, max_idx))

        # Plotting
        fig0, ax0 = plt.subplots(1, 1, figsize=(14, 6))
        fig1, ax1 = plt.subplots(1, 1, figsize=(20, 20))

        time = p*dt
        
        # Plot S array
        for min_idx, max_idx in mode_ranges:
            ax0.semilogy(freqs[min_idx:max_idx+1], S[min_idx:max_idx+1], label=f'Mode Range: {freqs[min_idx]:.1f} to {freqs[max_idx]:.1f}')
        ax0.semilogy(freqs, S, color='black', linestyle='--', alpha=0.3)
        # ax0.scatter(freqs[raw_peaks], S[raw_peaks], color='red', label='Peaks')
        ax0.set_xlim([band[0], band[1]])
        ax0.set_xlabel('Frequency')
        ax0.set_ylabel('PSD')
        ax0.set_title('Signal with Detected Peaks and Mode Ranges')
        # ax0.legend(framealpha=0.1, loc='lower center', fontsize=12)
        
        # Plot MAC matrix
        freq_indices = np.arange(len(raw_peaks))
        freq_labels = [f'{f:.1f}' for f in freqs[raw_peaks]]  # Round frequency labels
        
        cax = ax1.imshow(MAC_modified, interpolation='none', aspect='equal')  # Use gray colormap for binary matrix
        fig1.colorbar(cax, ax=ax1, label='MAC Value')
        ax1.set_xticks(freq_indices)
        ax1.set_xticklabels(freq_labels, rotation=90)
        ax1.set_yticks(freq_indices)
        ax1.set_yticklabels(freq_labels)
        ax1.set_title('MAC Matrix')
        ax1.set_xlabel('Frequency')
        ax1.set_ylabel('Frequency')
        fig0.suptitle("Time = "+str(time), fontsize=16)
        fig0.tight_layout()
        fig0.savefig('PSD'+str(p)+'.png')
        fig1.suptitle("Time = "+str(time), fontsize=16)
        fig1.tight_layout()
        fig1.savefig('MAC_Matrix'+str(p)+'.png')

        # # Step 5: Fit a bell curve on each identified range
        # def gaussian(x, a, x0, sigma):
        #     return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
        
        # selected_peaks = []
        # for min_idx, max_idx in mode_ranges:
        #     freqs_range = freqs[min_idx:max_idx+1]
        #     S_range = S[min_idx:max_idx+1]
        #     popt, _ = curve_fit(gaussian, freqs_range, S_range, p0=[max(S_range), freqs_range[np.argmax(S_range)], 1])
        #     selected_peak = np.searchsorted(freqs, popt[1])
        #     selected_peaks.append(selected_peak)

        # Step 5: for each detected range, select the peak with the highest value
        selected_peaks = []
        for min_idx, max_idx in mode_ranges:
            selected_peak = np.argmax(S[min_idx:max_idx+1]) + min_idx
            selected_peaks.append(selected_peak)
        selected_peaks = np.array(selected_peaks)

        # Only keep the n_modes highest peaks among the selected peaks
        selected_peaks = selected_peaks[np.argsort(S[selected_peaks])[::-1][:n_modes]]
        
        # Step 6: if time > time_ref : only keep the peaks matching the n_modes that are the most frequent through time
        #         else : keep all the peaks in results, concatenate the new peaks corresponding to new modes of vibration

        # Create a dictionary to hold the final results (for a faster algorithm, replace the dic by a numpy array, trick : concatenate)
        # def score(f1, f2, shape1, shape2):
        #     return np.abs(f1-f2) / self.compute_mac(shape1, shape2)
        results = {}

        if results_prev is None:
            for i, peak in enumerate(selected_peaks):
                peak_freq = freqs[peak]
                peak_mode_shape = U[peak, :, 0]
                results[peak_freq] = (i, peak_mode_shape)    
            return selected_peaks, results
        else:
            available_freqs = list(results_prev.keys())
            for peak in selected_peaks:
                peak_freq = freqs[peak]
                peak_mode_shape = U[peak, :, 0]
                for freq in available_freqs:
                    if self.compute_mac(peak_mode_shape, results_prev[freq][1]) > mac_threshold:
                        results[peak_freq] = (results_prev[freq][0], peak_mode_shape)
                        break
            return selected_peaks, results

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