# Process with a single time window

import numpy as np
from helper.data_loader import DataLoader
from helper.preprocessor import Preprocessor
from helper.processor import ModalFrequencyAnalyzer, PeakPicker
from helper.visualizer import Visualizer

# Set rcParams to customize plot appearance
import matplotlib.pyplot as plt
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.grid'] = True
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['legend.loc'] = 'upper right'
plt.rcParams['axes.titlesize'] = 20
plt.rcParams["figure.autolayout"] = True

# Define file paths for the data
file_paths_23 = [
    "data/Lello/Jul23/data_2023_07_10_040019_0.csv",
    "data/Lello/Jul23/data_2023_07_10_041032_0.csv",
    "data/Lello/Jul23/data_2023_07_10_042045_0.csv",
    "data/Lello/Jul23/data_2023_07_10_043058_0.csv",
    "data/Lello/Jul23/data_2023_07_10_044111_0.csv",
    "data/Lello/Jul23/data_2023_07_10_045123_0.csv",
]

# Define file paths for the data
file_paths_24 = [
    "data/Lello/Jul24/data_2024_07_10_041716_0.csv",
    "data/Lello/Jul24/data_2024_07_10_042745_0.csv",
    "data/Lello/Jul24/data_2024_07_10_043815_0.csv",
    "data/Lello/Jul23/data_2023_07_10_043058_0.csv",
    "data/Lello/Jul23/data_2023_07_10_044111_0.csv",
    "data/Lello/Jul23/data_2023_07_10_045123_0.csv",
]

### USER ###
file_paths = file_paths_24[:4]
location = "Lello_Jul24_stairs"
nperseg = 2048 # 256, 512, 1024, 2048, 4096, 8192
selected_indices = [3, 4, 5, 6]  # Indices of the selected sensors : stair
### USER ###

# Load the data
loader = DataLoader(file_paths)
time, data = loader.load_data()
print("Time shape:", time.shape)
print("Data shape:", data.shape)

# Define scaling factors for the sensor data
scaling_factors = np.array([0.4035*1000, 0.4023*1000, 0.4023*1000, 0.4023*1000, 0.4015*1000, 0.4014*1000, 0.4007*1000, 0.4016*1000])

# Transform the data
preprocessor = Preprocessor(time, data, scaling_factors)
detrended_data = preprocessor.detrend_and_scale()

# Initialize the Visualizer with time and frequencies
# Create a unique folder name based on hyperparameters
time_window_size = 600 * len(file_paths)
processing_method = "Welch"
folder_name = f"loc_{location}_wind_{time_window_size}_meth_{processing_method}_nperseg_{nperseg}"

visualizer = Visualizer(time, output_dir="results")
labels=["Base X", "Base Y", "Base Z", "Stair 1 X", "Stair 1 Y", "Stair 1 Z", "Stair 2 Z", "Vitral Z"]
selected_labels = [labels[i] for i in selected_indices]

visualizer.plot_data(np.array([data[:, i] for i in selected_indices]).T, "Original_sensor_data", folder_name, y_label="[A]", labels=selected_labels)

visualizer.plot_data(np.array([detrended_data[:, i] for i in selected_indices]).T, "Detrended_scaled_sensor_data", folder_name, y_label="[mG]", labels=selected_labels)

# Analyze the frequency components
analyzer = ModalFrequencyAnalyzer(detrended_data, time)

frequencies, fft_data = analyzer.compute_fft()
visualizer.plot_fft(frequencies, fft_data, folder_name, labels=labels)

freqs, psd_matrix = analyzer.compute_psd_matrix(selected_indices, nperseg=nperseg)
U_PSD, S_PSD, V_PSD = analyzer.perform_svd_psd()
visualizer.plot_psd(freqs, psd_matrix, folder_name, labels=selected_labels)

coherence_matrix = analyzer.compute_coherence_matrix()
U_corr, S_corr, V_corr = analyzer.perform_svd_coherence()
P1, P2, P3 = analyzer.get_pp_index()

# Peak picking
peak_picker = PeakPicker(analyzer)
peaks = peak_picker.identify_peaks_0(P3, distance=1)
range_to_check = [(11,12), (16,17), (22,23)]
# peaks, range_to_check = analyzer.identify_peaks_bis(freqs, P3, S_PSD[:, 0])
# peaks, _ = analyzer.identify_peaks_bis_bis(freqs, S_PSD[:, 0], U_PSD)

# Visualize the singular values
visualizer.plot_sigmas(freqs, S_PSD, peaks, folder_name)
visualizer.plot_pp_index(freqs, [P1, P2, P3], peaks, folder_name)

mode_frequency, mode_shape = peak_picker.identify_mode_shapes(U_PSD, peaks)
print(f"Identified mode frequencies: {mode_frequency} Hz")

# Visualize the PCA of the mode shapes
# visualizer.plot_PCA(mode_shape, folder_name)

# Visualize the MAC matrix of the mode shapes
MAC = np.zeros((len(peaks), len(peaks)))
for i in range(len(peaks)):
    for j in range(len(peaks)):
        MAC[i, j] = peak_picker.compute_mac(mode_shape[i], mode_shape[j])
visualizer.plot_MAC_matrix(MAC, mode_frequency, peaks, folder_name)
