import numpy as np
from helper.data_loader import DataLoader
from helper.preprocessor import SignalTransformer
from helper.processor import FrequencyAnalyzer
from helper.visualizer import Visualizer
from scipy.ndimage import gaussian_filter1d
import PyOMA as oma

def get_vibration_frequencies(file_paths, location):
    # Load the data
    loader = DataLoader(file_paths)
    time, data = loader.load_data()
    print("Time shape:", time.shape)
    print("Data shape:", data.shape)

    # Define scaling factors for the sensor data
    scaling_factors = np.array([0.4035*1000, 0.4023*1000, 0.4023*1000, 0.4023*1000, 0.4015*1000, 0.4014*1000, 0.4007*1000, 0.4016*1000])

    # Transform the data
    transformer = SignalTransformer(time, data, scaling_factors)
    detrended_data = transformer.detrend_and_scale()

    # Initialize the Visualizer with time and frequencies
    # Create a unique folder name based on hyperparameters
    time_window_size = 600 * len(file_paths)
    processing_method = "Welch"
    nperseg = 2048 # 256, 512, 1024, 2048, 4096, 8192
    folder_name = f"loc_{location}_wind_{time_window_size}_meth_{processing_method}_nperseg_{nperseg}"

    visualizer = Visualizer(time, output_dir="results")
    labels=["Base X", "Base Y", "Base Z", "Stair 1 X", "Stair 1 Y", "Stair 1 Z", "Stair 2 Z", "Vitral Z"]
    # visualizer.plot_data(data[:,3:7], "Original_sensor_data", folder_name, y_label="[A]", labels=labels[3:7])
    # visualizer.plot_data(detrended_data[:,3:7], "Detrended_scaled_sensor_data", folder_name, y_label="[mG]", labels=labels[3:7])

    # Analyze the frequency components
    analyzer = FrequencyAnalyzer(detrended_data, time)
    frequencies, fft_data = analyzer.compute_fft()
    # visualizer.plot_fft(frequencies, fft_data, folder_name, labels=labels)

    selected_indices = [3, 4, 5, 6]  # Indices of the selected sensors : stair

    # Compute PSD using Welch's method
    fs = 1 / np.mean(np.diff(time))
    # print("Sampling frequency:", fs)
    freqs, psd_matrix = analyzer.compute_psd_matrix(fs, selected_indices, nperseg=nperseg)
    # print(np.mean(np.diff(freqs)), np.mean(np.diff(time)))
    # Visualize the PSDs
    # visualizer.plot_psd(freqs, psd_matrix, folder_name, labels=labels[3:7])

    # Compute the coherence matrix
    coherence_matrix = analyzer.compute_coherence_matrix(psd_matrix)

    # Perform SVD of the PSD matrix and coherence matrix
    U_PSD, S_PSD, V_PSD = analyzer.perform_svd(psd_matrix)
    U_corr, S_corr, V_corr = analyzer.perform_svd(coherence_matrix)

    # Get the PP index
    P1, P2, P3 = analyzer.get_pp_index(coherence_matrix, S_corr)

    # Identify and print the mode frequency
    peaks = analyzer.identify_peaks(freqs, P3, distance=1)
    range_to_check = [(11,12), (16,17), (22,23)]
    peaks, range_to_check = analyzer.identify_peaks_bis(freqs, P3, S_PSD[:, 0], ranges_to_check=range_to_check)

    # Visualize
    if False :
        visualizer.plot_pp_index(freqs, [P1, P2, P3], peaks, folder_name)
        visualizer.plot_sigmas(freqs, S_PSD, peaks, folder_name)

    mode_frequency, mode_shape = analyzer.identify_mode_shapes(freqs, U_PSD, peaks)
    print(f"Identified mode frequencies: {mode_frequency} Hz")

    # Visualize the PCA of the mode shapes
    # visualizer.plot_PCA(mode_shape, folder_name)

    # Visualize the MAC matrix of the mode shapes
    MAC = np.zeros((len(peaks), len(peaks)))
    for i in range(len(peaks)):
        for j in range(len(peaks)):
            MAC[i, j] = analyzer.compute_mac(mode_shape[i], mode_shape[j])
    # visualizer.plot_MAC_matrix(MAC, mode_frequency, peaks, folder_name)
    return mode_frequency

def get_vibration_frequencies_pyoma(file_paths, location):
    """
    Detects vibration frequencies using PyOMA's FDD method.

    Parameters:
        file_paths (list): List of file paths to process.
        fs (float): Sampling frequency of the data.

    Returns:
        list: List of detected frequencies for each file.
    """
    loader = DataLoader(file_paths)
    time, data = loader.load_data()
    print("Time shape:", time.shape)
    print("Data shape:", data.shape)

    # Define scaling factors for the sensor data
    scaling_factors = np.array([0.4035*1000, 0.4023*1000, 0.4023*1000, 0.4023*1000, 0.4015*1000, 0.4014*1000, 0.4007*1000, 0.4016*1000])

    # Transform the data
    transformer = SignalTransformer(time, data, scaling_factors)
    detrended_data = transformer.detrend_and_scale()

    selected_indices = [3, 4, 5, 6]  # Indices of the selected sensors : stair

    # Compute PSD using Welch's method
    fs = 1 / np.mean(np.diff(time))

    all_frequencies = []
    
    data = detrended_data[:, selected_indices]

    # Apply FDD method directly
    FDD = oma.FDDsvp(data, fs)

    # Define approximate peaks identified from the plot
    FreQ = [11.29, 16.05, 22.54]

    # Extract the modal properties
    Res_FDD = oma.FDDmodEX(FreQ, FDD[1])
    Res_EFDD = oma.EFDDmodEX(FreQ, FDD[1], method='EFDD')
    # Res_FSDD = oma.EFDDmodEX(FreQ, FDD[1], method='FSDD', npmax = 35, MAClim=0.95)

    return Res_FDD['Frequencies'], Res_EFDD['Frequencies']#, Res_FSDD['Frequencies']

def get_vibration_frequencies_best(file_paths, location, results_prev=None, p=None, dt=None):
    # Load the data
    loader = DataLoader(file_paths)
    time, data = loader.load_data()
    print("Time shape:", time.shape)
    print("Data shape:", data.shape)

    # Define scaling factors for the sensor data
    scaling_factors = np.array([0.4035*1000, 0.4023*1000, 0.4023*1000, 0.4023*1000, 0.4015*1000, 0.4014*1000, 0.4007*1000, 0.4016*1000])

    # Transform the data
    transformer = SignalTransformer(time, data, scaling_factors)
    detrended_data = transformer.detrend_and_scale()

    # Initialize the Visualizer with time and frequencies
    # Create a unique folder name based on hyperparameters
    time_window_size = 600 * len(file_paths)
    processing_method = "Welch"
    nperseg = 2048 # 256, 512, 1024, 2048, 4096, 8192
    folder_name = f"loc_{location}_wind_{time_window_size}_meth_{processing_method}_nperseg_{nperseg}"

    visualizer = Visualizer(time, output_dir="results")
    labels=["Base X", "Base Y", "Base Z", "Stair 1 X", "Stair 1 Y", "Stair 1 Z", "Stair 2 Z", "Vitral Z"]
    # visualizer.plot_data(data[:,3:7], "Original_sensor_data", folder_name, y_label="[A]", labels=labels[3:7])
    # visualizer.plot_data(detrended_data[:,3:7], "Detrended_scaled_sensor_data", folder_name, y_label="[mG]", labels=labels[3:7])

    # Analyze the frequency components
    analyzer = FrequencyAnalyzer(detrended_data, time)

    selected_indices = [3, 4, 5, 6]  # Indices of the selected sensors : stair

    # Compute PSD using Welch's method
    fs = 1 / np.mean(np.diff(time))
    # print("Sampling frequency:", fs)
    freqs, psd_matrix = analyzer.compute_psd_matrix(fs, selected_indices, nperseg=nperseg)
    # print(np.mean(np.diff(freqs)), np.mean(np.diff(time)))
    # Visualize the PSDs
    # visualizer.plot_psd(freqs, psd_matrix, folder_name, labels=labels[3:7])

    # Compute the coherence matrix
    coherence_matrix = analyzer.compute_coherence_matrix(psd_matrix)

    # Perform SVD of the PSD matrix and coherence matrix
    U_PSD, S_PSD, V_PSD = analyzer.perform_svd(psd_matrix)
    U_corr, S_corr, V_corr = analyzer.perform_svd(coherence_matrix)

    # Get the PP index
    P1, P2, P3 = analyzer.get_pp_index(coherence_matrix, S_corr)

    # Identify and print the mode frequency
    peaks, results = analyzer.identify_peaks_bis_bis(freqs, S_PSD[:, 0], U_PSD, results_prev=results_prev, p=p, dt=dt)

    # Visualize
    if False :
        visualizer.plot_pp_index(freqs, [P1, P2, P3], peaks, folder_name)
        visualizer.plot_sigmas(freqs, S_PSD, peaks, folder_name)

    mode_frequency, mode_shape = analyzer.identify_mode_shapes(freqs, U_PSD, peaks)
    # print(f"Identified mode frequencies: {mode_frequency} Hz")

    # Visualize the PCA of the mode shapes
    # visualizer.plot_PCA(mode_shape, folder_name)

    # Visualize the MAC matrix of the mode shapes
    # MAC = np.zeros((len(peaks), len(peaks)))
    # for i in range(len(peaks)):
    #     for j in range(len(peaks)):
    #         MAC[i, j] = analyzer.compute_mac(mode_shape[i], mode_shape[j])
    # visualizer.plot_MAC_matrix(MAC, mode_frequency, peaks, folder_name)
    return mode_frequency, results