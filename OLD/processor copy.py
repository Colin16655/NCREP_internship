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
        self.idx_method2 = 0

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
            if np.isnan(peak):
                mode_shape = np.full(U.shape[1], np.nan)
                mode_shapes.append(mode_shape)
                mode_fqcies.append(np.nan)
            else:
                peak = int(peak)
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

    def identify_peaks_2(self, PP, PSD_sigma, U, band=(8, 24), distance=2, mac_threshold=0.9, n_modes=4, n_mem=4, results_prev=None, p=None, dt=None):
        """
        Detects peaks within a specified frequency band by dividing the given frequency domain into similar mode ranges using
        the MAC (Modal Assurance Criterion) matrix. The function then selects the argmax within the ranges that best match 
        the previous results. If `results_prev` is None, the function selects the `n_modes + 1` highest peaks among all the 
        detected ranges.

        Parameters:
            PSD_sigma (numpy.ndarray): Power spectral density (PSD) matrix.
            U (numpy.ndarray): Array of eigenvectors (mode shapes) corresponding to the frequencies.
            band (tuple): Frequency band within which to detect peaks (default is (8, 24)).
            distance (int): Minimum distance between peaks in the frequency domain (default is 2).
            mac_threshold (float): Threshold for MAC value to consider peaks as belonging to the same mode (default is 0.9).
            n_modes (int): Number of modes to detect (default is 4).
            n_mem (int): Number of memories or previous instances to consider (default is 4).
            results_prev (numpy.ndarray, optional): Array containing previous results for comparison (default is None).
            p (int, optional): Index or identifier for the current time step (default is None).
            dt (float, optional): Time step duration for ploting (default is None).

        Returns:
            selected_peaks (numpy.ndarray): Indices of the final detected peaks.
            results (numpy_ndarray): Updated array of results with the newly detected peaks.
        """

        # Compute the second derivative (curvature) of the PSD arrays.
        PP_curvatures = np.gradient(np.gradient(PP))
        PSD_curvatures = np.gradient(np.gradient(PSD_sigma))
        
        S = PP

        def select_peak_with_curvature(min_idx, max_idx):
            # Extract the relevant portions of the arrays within the specified frequency band.
            band_freqs = freqs[min_idx:max_idx + 1]
            band_PP = PP[min_idx:max_idx + 1]
            band_PSD_sigma = PSD_sigma[min_idx:max_idx + 1]

            # Determine the indices of the maximum values within the band.
            PP_argmax = np.argmax(band_PP)
            PSD_sigma_argmax = np.argmax(band_PSD_sigma)

            # Compare curvatures to decide which peak to select.
            if PP_curvatures[PP_argmax] > PSD_curvatures[PSD_sigma_argmax]:
                return PP_argmax + min_idx
            else:   
                return PSD_sigma_argmax + min_idx

        # Step 1: Access frequency values and define equally spaced frequency domain within the band.
        freqs = self.analyzer.freq_psd
        band_min_idx, band_max_idx = self.extract_indices_within_band(freqs, band[0], band[1])
        f_domain = np.arange(band_min_idx, band_max_idx + 1, distance)

        # Step 2: Compute the MAC matrix for the detected peaks using eigenvectors.
        mode_shapes = U[f_domain, :, 0]
        MAC = np.zeros((len(f_domain), len(f_domain)))
        for i in range(len(f_domain)):
            for j in range(i, len(f_domain)):
                MAC[i, j] = MAC[j, i] = self.compute_mac(mode_shapes[i], mode_shapes[j])

        # Apply the MAC threshold to the matrix.
        MAC_modified = np.where(MAC >= mac_threshold, MAC, 0)
        
        # Step 3: Group points that belong to the same mode based on MAC threshold.
        modes = []
        used_indices = set()
        for i in range(len(f_domain)):
            if i in used_indices:
                continue
            mode_group = [i]
            for j in range(i + 1, len(f_domain)):
                if MAC[i, j] > mac_threshold:
                    mode_group.append(j)
                    used_indices.add(j)
                else:
                    break
            modes.append(mode_group)
        
        # Step 4: Obtain the frequency range for each mode, ignoring single peaks.
        mode_ranges = []
        for mode_group in modes:
            if len(mode_group) > 1:
                min_idx = min(f_domain[mode_group])
                max_idx = max(f_domain[mode_group])
                mode_ranges.append((min_idx, max_idx))

        # Step 5: Evaluate and score frequency ranges based on previous results.
        score_ranges = np.zeros(len(mode_ranges))
        which_mode = np.zeros((len(mode_ranges), n_modes + 1))
        if self.idx_method2 > 0:
            for i in range(n_modes + 1):
                for j in range(self.idx_method2): 
                    f_to_check = results_prev[i, j]
                    if not np.isnan(f_to_check):
                        for k, (min_idx, max_idx) in enumerate(mode_ranges):
                            if freqs[min_idx] <= freqs[int(f_to_check)] <= freqs[max_idx]:
                                score_ranges[k] += 1
                                which_mode[k][i] += 1
                                break

        # If no previous results, initialize by scoring peaks within the mode ranges.
        else:
            for k, (min_idx, max_idx) in enumerate(mode_ranges):
                selected_peak = np.argmax(S[min_idx:max_idx + 1]) + min_idx
                score_ranges[k] = S[selected_peak]

        # Step 6: Select the highest-scoring frequency ranges.
        potential_selected_ranges = np.argsort(score_ranges)[::-1][:n_modes + 1]
        selected_ranges = []
        for i, temp in enumerate(potential_selected_ranges):
            if score_ranges[potential_selected_ranges[i]] != 0:
                selected_ranges.append(temp)
            else:
                break
        selected_ranges = np.array(selected_ranges)
        
        # Step 7: For each selected range, select the peak with the highest value.
        selected_peaks = np.full(n_modes + 1, np.nan)
        for i, idx in enumerate(selected_ranges):
            min_idx, max_idx = mode_ranges[idx]
            selected_peak = np.argmax(S[min_idx:max_idx + 1]) + min_idx
            if self.idx_method2 > 0:
                selected_peaks[np.argmax(which_mode[idx])] = selected_peak
            else:
                selected_peaks[i] = selected_peak

        # Step 8: Update the results array with the newly detected peaks.
        results = np.zeros((n_modes + 1, n_mem))
        if results_prev is not None:
            results[:, :self.idx_method2] = results_prev[:, :self.idx_method2]

        results[:, self.idx_method2] = selected_peaks

        if self.idx_method2 < n_mem - 1:
            self.idx_method2 += 1

        # # Uncomment the following line to enable plotting for debugging.
        # self.plot_debug(freqs, S, f_domain, MAC_modified, mode_ranges, selected_ranges, p, dt)

        return selected_peaks,

    def plot_debug(self, freqs, S, f_domain, MAC_modified, mode_ranges, selected_ranges, p, dt):
        """
        Plots the signal with detected peaks and the MAC matrix for debugging purposes.

        Parameters:
            freqs (array-like): Array of frequency values.
            S (array-like): Signal values corresponding to the frequencies.
            f_domain (array-like): Frequency domain indices within the specified band.
            MAC_modified (array-like): Modified MAC matrix with applied threshold.
            mode_ranges (array-like): List of frequency ranges corresponding to detected modes.
            selected_ranges (array-like): Indices of the selected frequency ranges.
            p (int): Index or identifier for the current time step.
            dt (float): Time step duration for frequency domain analysis.
        """
        # Plotting the signal with detected peaks and mode ranges
        fig0, ax0 = plt.subplots(1, 1, figsize=(14, 6))
        fig1, ax1 = plt.subplots(1, 1, figsize=(20, 20))

        time = p * dt
        
        # Plot S array
        for min_idx, max_idx in np.array(mode_ranges)[selected_ranges]:
            ax0.semilogy(freqs[min_idx:max_idx + 1], S[min_idx:max_idx + 1], label=f'Mode Range: {freqs[min_idx]:.1f} to {freqs[max_idx]:.1f}')
        ax0.semilogy(freqs, S, color='black', linestyle='--', alpha=0.3)
        ax0.set_xlim([band[0], band[1]])
        ax0.set_xlabel('Frequency')
        ax0.set_ylabel('PSD')
        ax0.set_title('Signal with Detected Peaks and Mode Ranges')
        
        # Plot MAC matrix
        freq_indices = np.arange(len(f_domain))
        ax1.imshow(MAC_modified, cmap='viridis', extent=[freq_indices[0], freq_indices[-1], freq_indices[-1], freq_indices[0]])
        ax1.set_title('Modified MAC Matrix')
        ax1.set_xlabel('Frequency Index')
        ax1.set_ylabel('Frequency Index')
        
        plt.show()

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
