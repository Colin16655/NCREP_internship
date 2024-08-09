import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import PyOMA as oma

class ModalFrequencyAnalyzer:
    """
    A class for performing modal frequency analysis, including SVD on both PSD and coherence matrices.
    """

    def __init__(self, data=None, time=None):
        """
        Initializes the ModalFrequencyAnalyzer with data and time vector.

        Parameters:
            data (numpy.ndarray): Sensor data, shape (n_samples, n_sensors).
            time (numpy.ndarray): Time vector corresponding to the sensor data.
        """
        self.data = data
        self.time = time
        self.fft_data = None
        self.psd_matrix = None
        self.coherence_matrix = None
        self.freq_fft = None    # frequencies for FFT
        self.freq_psd = None    # frequencies for PSD
        self.U_psd = None       # shape (len(freq_psd), n_sensors, n_sensors)
        self.S_psd = None       # shape (len(freq_psd), n_sensors)
        self.V_psd = None       # shape (len(freq_psd), n_sensors, n_sensors)
        self.U_coherence = None # shape (len(freq_psd), n_sensors, n_sensors)
        self.S_coherence = None # shape (len(freq_psd), n_sensors, n_sensors)
        self.V_coherence = None # shape (len(freq_psd), n_sensors, n_sensors)
        self.P1 = None
        self.P2 = None
        self.P3 = None

    def compute_fft(self):
        """
        Computes the FFT of the data.

        Returns:
            tuple: Frequencies and the FFT of the data.
        """
        n_samples = self.data.shape[0]
        sampling_dt = np.mean(np.diff(self.time))
        self.freq_fft = np.fft.rfftfreq(n_samples, d=sampling_dt)
        self.fft_data = np.fft.rfft(self.data, axis=0)
        return self.freq_fft, self.fft_data

    def compute_psd_matrix(self, nperseg=1024):
        """
        Computes the PSD matrix for selected sensors using Welch's method.

        Parameters:
            fs (float): Sampling frequency.
            nperseg (int): Length of each segment for Welch's method.

        Returns:
            tuple: Frequencies and the PSD matrix for the selected sensors.
        """
        fs = 1 / np.mean(np.diff(self.time))
        n_sensors = len(self.data[0])
        self.psd_matrix = np.zeros((nperseg // 2 + 1, n_sensors, n_sensors), dtype=complex)
    
        # Calculate the PSD for each sensor (diagonal elements of the matrix)
        self.freq_psd, psd = signal.welch(self.data[:, 0], fs=fs, nperseg=nperseg)
        self.psd_matrix[:, 0, 0] = psd  # Store the PSD for the first sensor
        
        for i in range(1, n_sensors):
            _, psd = signal.welch(self.data[:, i], fs=fs, nperseg=nperseg)
            self.psd_matrix[:, i, i] = psd
        for i in range(n_sensors):
            for j in range(i+1, n_sensors):
                _, csd = signal.csd(self.data[:, i], self.data[:, j], fs=fs, nperseg=nperseg)
                self.psd_matrix[:, i, j] = csd
                self.psd_matrix[:, j, i] = np.conj(csd)

        return self.freq_psd, self.psd_matrix

    def compute_coherence_matrix(self):
        """
        Computes the coherence matrix from the PSD matrix.

        Returns:
            numpy.ndarray: The coherence matrix.
        """
        if self.psd_matrix is None:
            raise ValueError("PSD matrix not computed. Call compute_psd_matrix() first.")
        
        num_freqs, num_sensors, _ = self.psd_matrix.shape

        self.coherence_matrix = np.zeros_like(self.psd_matrix)
        for i in range(num_freqs):
            psd_conjugate = np.conjugate(self.psd_matrix[i])
            numerator = self.psd_matrix[i] * psd_conjugate 
            diagonal = np.real(np.diagonal(self.psd_matrix[i]))
            denominator = np.outer(diagonal, diagonal)    
            self.coherence_matrix[i] = numerator / denominator
        return self.coherence_matrix
    
    def perform_svd_psd(self):
        """
        Performs Singular Value Decomposition (SVD) on the PSD matrix.

        Returns:
            tuple: U, S, V matrices from SVD of the PSD matrix.
        """
        if self.psd_matrix is None:
            raise ValueError("PSD matrix not computed. Call compute_psd_matrix() first.")
        
        self.U_psd, self.S_psd, self.V_psd = np.linalg.svd(self.psd_matrix, full_matrices=False)
        return self.U_psd, self.S_psd, self.V_psd

    def perform_svd_coherence(self):
        """
        Performs Singular Value Decomposition (SVD) on the coherence matrix.

        Returns:
            tuple: U, S, V matrices from SVD of the coherence matrix.
        """
        if self.coherence_matrix is None:
            raise ValueError("Coherence matrix not computed. Call compute_coherence_matrix() first.")
        
        self.U_coherence, self.S_coherence, self.V_coherence = np.linalg.svd(self.coherence_matrix, full_matrices=False)
        return self.U_coherence, self.S_coherence, self.V_coherence

    def compute_pp_index(self):
        """
        Computes the PP index from the singular values.

        Returns:
            tuple: P1, P2, and P3 indices.
        """
        if self.coherence_matrix is None:
            raise ValueError("Coherence matrix not computed. Call compute_coherence_matrix() first.")
        
        num_freqs = len(self.coherence_matrix)
        self.P1 = np.zeros(num_freqs)
        self.P2 = np.zeros(num_freqs)
        self.P3 = np.zeros(num_freqs)
        for k in range(num_freqs):
            coherence = self.coherence_matrix[k]
            self.P1[k] = np.sum([np.abs(1 / np.log(coherence[i, j])) for i in range(len(coherence)) for j in range(i+1, len(coherence))])
            self.P2[k] = np.abs(1 / np.log(self.S_coherence[k, 0] / len(coherence)))
            self.P3[k] = np.prod(1 / self.S_coherence[k, 1:])
        return self.P1, self.P2, self.P3
    

class PeakPicker:
    """
    A class to perform peak picking given a ModalFrequencyAnalyzer instance that has been initialized.

    Attributes:
        analyzer (ModalFrequencyAnalyzer): Instance of the ModalFrequencyAnalyzer class.
        frequencies (numpy.ndarray): Array of frequency values corresponding to the signal data.
        signal_data (numpy.ndarray): Signal values corresponding to the frequencies.
    """

    def __init__(self, analyzer):
        """
        Initializes the PeakPicker with a ModalFrequencyAnalyzer instance, frequencies, and signal data.

        Parameters:
            analyzer (ModalFrequencyAnalyzer): An initialized instance of the ModalFrequencyAnalyzer class.

        Raises:
            ValueError: If the analyzer is not properly initialized.
        """
        self.analyzer = analyzer

    @staticmethod
    def compute_mac(mode_shape1, mode_shape2):
        """
        Computes the Modal Assurance Criterion (MAC) between two mode shapes.

        Parameters:
            mode_shape1 (numpy.ndarray): The first mode shape.
            mode_shape2 (numpy.ndarray): The second mode shape.

        Returns:
            float: The MAC value between the two mode shapes.
        """
        return np.abs(np.dot(mode_shape1.conj().T, mode_shape2))**2 / (np.dot(mode_shape1.conj().T, mode_shape1) * np.dot(mode_shape2.conj().T, mode_shape2)).real

    @staticmethod
    def extract_indices_within_band(freqs, band_min, band_max):
        """
        Extracts indices of the frequency array that fall within the specified frequency band.

        Parameters:
            freqs (numpy.ndarray): Array of frequency values.
            band_min (float): Minimum frequency of the band.
            band_max (float): Maximum frequency of the band.

        Returns:
            tuple: Indices corresponding to the specified frequency band.

        Raises:
            ValueError: If the specified band does not overlap with the frequency array.
        """
        min_idx = np.searchsorted(freqs, band_min, side='left')
        max_idx = np.searchsorted(freqs, band_max, side='right') - 1
        if min_idx >= len(freqs) or max_idx >= len(freqs) or min_idx > max_idx:
            raise ValueError("The specified band does not overlap with the frequency array.")
        return min_idx, max_idx

    def identify_mode_shapes(self, U, peaks):
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
            mode_shape = U[peak, :, 0]
            mode_shapes.append(mode_shape)
            mode_fqcies.append(self.analyzer.freq_psd[peak])
        return np.array(mode_fqcies), np.array(mode_shapes)

    def identify_peaks_0(self, S, band=(8, 24), distance=10, sigma=14):
        """
        Identifies peaks in the signal S within a specified frequency band, and refines the peak positions 
        by checking the non-smoothed data within a 1 Hz range before and after each identified peak.

        Parameters:
            S (numpy.ndarray): Signal data from which to identify peaks.
            band (tuple): A tuple specifying the frequency band (min, max) within which to find peaks.
            distance (int): Minimum distance between peaks (in terms of number of data points).
            sigma (float): Standard deviation for Gaussian smoothing.

        Returns:
            numpy.ndarray: Indices of the refined peaks found within the specified frequency band.
        """
        # Access frequencies from the analyzer instance
        freqs = self.analyzer.freq_psd
        
        # Check if freq is available
        if freqs is None:
            raise ValueError("Frequencies for PSD not computed. Ensure compute_psd_matrix() has been called.")
        
        # Ensure S matches the frequency array length
        if len(S) != len(freqs):
            raise ValueError("Length of S does not match the length of the frequency array.")
        
        # Extract the indices within the specified frequency band
        band_min, band_max = band
        min_idx, max_idx = self.extract_indices_within_band(freqs, band_min, band_max)
        band_freqs = freqs[min_idx:max_idx + 1]
        band_S = S[min_idx:max_idx + 1]
        
        # Smooth the signal data
        S_smooth = gaussian_filter1d(band_S, sigma=sigma)
        smooth_peaks, _ = signal.find_peaks(S_smooth, distance=distance)
        
        # Refine peak locations
        num_samples_1hz = int(np.round(1 / (freqs[1] - freqs[0])))
        refined_peaks = []
        for peak in smooth_peaks:
            # Calculate the range to check in the original signal
            search_range = (max(peak - num_samples_1hz, 0), min(peak + num_samples_1hz, len(band_S) - 1))
            # Extract the relevant section from the original signal
            original_section = S[min_idx:max_idx + 1][search_range[0]:search_range[1] + 1]
            # Find the peak in the original data section
            refined_peak = np.argmax(original_section) #signal.find_peaks(original_section, distance=distance)
            # Map new peak back to the global index
            refined_peaks.append(search_range[0] + refined_peak + min_idx)
        return np.unique(np.array(refined_peaks))

    def identify_peaks_1(self, PP, PSD_sigma, band=(8, 24), distance=10, sigma=14, curvature_threshold=-0.004, ranges_to_check=[(10.5, 12.5), (15.5, 17.5), (21.5, 23.5)]):
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
        # Access frequencies from the analyzer instance
        freqs = self.analyzer.freq_psd

        def get_peaks_and_curvatures(signal_data):
            # Extract indices within the specified frequency band
            band_min, band_max = band
            min_idx, max_idx = self.extract_indices_within_band(freqs, band_min, band_max)
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
                if range_min_freq <= freqs[peak] <= range_max_freq:
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
                
        return np.unique(np.array(selected_peaks))

    def identify_peaks_2(self, S, U, band=(8, 24), distance=2, mac_threshold=0.9, n_modes=4, results_prev=None, 
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
            mac_threshold (float): Threshold for MAC value to consider peaks as the same mode (default is 0.8).

        Returns:
            selected_peaks (array-like): Indices of the final refined peaks.
        """
        # Access frequencies from the analyzer instance
        freqs = self.analyzer.freq_psd
        
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

        # # Plotting
        # fig0, ax0 = plt.subplots(1, 1, figsize=(14, 6))
        # fig1, ax1 = plt.subplots(1, 1, figsize=(20, 20))

        # time = p*dt
        
        # # Plot S array
        # for min_idx, max_idx in mode_ranges:
        #     ax0.semilogy(freqs[min_idx:max_idx+1], S[min_idx:max_idx+1], label=f'Mode Range: {freqs[min_idx]:.1f} to {freqs[max_idx]:.1f}')
        # ax0.semilogy(freqs, S, color='black', linestyle='--', alpha=0.3)
        # # ax0.scatter(freqs[raw_peaks], S[raw_peaks], color='red', label='Peaks')
        # ax0.set_xlim([band[0], band[1]])
        # ax0.set_xlabel('Frequency')
        # ax0.set_ylabel('PSD')
        # ax0.set_title('Signal with Detected Peaks and Mode Ranges')
        # # ax0.legend(framealpha=0.1, loc='lower center', fontsize=12)
        
        # # Plot MAC matrix
        # freq_indices = np.arange(len(raw_peaks))
        # freq_labels = [f'{f:.1f}' for f in freqs[raw_peaks]]  # Round frequency labels
        
        # cax = ax1.imshow(MAC_modified, interpolation='none', aspect='equal')  # Use gray colormap for binary matrix
        # fig1.colorbar(cax, ax=ax1, label='MAC Value')
        # ax1.set_xticks(freq_indices)
        # ax1.set_xticklabels(freq_labels, rotation=90)
        # ax1.set_yticks(freq_indices)
        # ax1.set_yticklabels(freq_labels)
        # ax1.set_title('MAC Matrix')
        # ax1.set_xlabel('Frequency')
        # ax1.set_ylabel('Frequency')
        # fig0.suptitle("Time = "+str(time), fontsize=16)
        # fig0.tight_layout()
        # fig0.savefig('PSD'+str(p)+'.png')
        # fig1.suptitle("Time = "+str(time), fontsize=16)
        # fig1.tight_layout()
        # fig1.savefig('MAC_Matrix'+str(p)+'.png')

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

    def identify_peaks_pyoma(self):
        data = self.analyzer.data
        fs = 1 / np.mean(np.diff(self.analyzer.time))
        # Apply FDD method directly
        FDD = oma.FDDsvp(data, fs)

        # Define approximate peaks identified from the plot
        FreQ = [11.29, 16.05, 22.54]

        # Extract the modal properties
        # Res_FDD = oma.FDDmodEX(FreQ, FDD[1])
        Res_EFDD = oma.EFDDmodEX(FreQ, FDD[1], method='EFDD')
        # Res_FSDD = oma.EFDDmodEX(FreQ, FDD[1], method='FSDD', npmax = 35, MAClim=0.95)
        plt.close()
        return Res_EFDD['Frequencies'] # Res_FSDD['Frequencies']
