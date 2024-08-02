import numpy as np
from helper.data_loader import DataLoader
from helper.signal_transformer import SignalTransformer
from helper.frequency_analyzer import FrequencyAnalyzer
from helper.visualizer import Visualizer
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

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
file_paths = [
    "data/Lello/Jul23/data_2023_07_10_040019_0.csv",
    "data/Lello/Jul23/data_2023_07_10_041032_0.csv",
    "data/Lello/Jul23/data_2023_07_10_042045_0.csv",
    "data/Lello/Jul23/data_2023_07_10_043058_0.csv",
    "data/Lello/Jul23/data_2023_07_10_044111_0.csv",
    "data/Lello/Jul23/data_2023_07_10_045123_0.csv",
]

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
location = "Lello_Jul23_stairs"
time_window_size = 600 * len(file_paths)
processing_method = "Welch"
nperseg = 8192*8 # 256, 512, 1024, 2048, 4096, 8192
folder_name = f"loc_{location}_wind_{time_window_size}_meth_{processing_method}_nperseg_{nperseg}"

visualizer = Visualizer(time, output_dir="results")
labels=["Base X", "Base Y", "Base Z", "Stair 1 X", "Stair 1 Y", "Stair 1 Z", "Stair 2 Z", "Vitral Z"]
visualizer.plot_data(data, "Original_sensor_data", folder_name, y_label="[A]", labels=labels)
visualizer.plot_data(detrended_data, "Detrended_scaled_sensor_data", folder_name, y_label="[mG]", labels=labels)

# Analyze the frequency components
analyzer = FrequencyAnalyzer(detrended_data, time)
frequencies, fft_data = analyzer.compute_fft()
visualizer.plot_fft(frequencies, fft_data, folder_name, labels=labels)

selected_indices = [3, 4, 5, 6]  # Indices of the selected sensors : stair

# Compute PSD using Welch's method
fs = 1 / np.mean(np.diff(time))
freqs, psd_matrix = analyzer.compute_psd_matrix(fs, selected_indices, nperseg=nperseg)

# Visualize the PSDs
visualizer.plot_psd(freqs, psd_matrix, folder_name, labels=labels)

# Compute the correlation matrix
correlation_matrix = analyzer.compute_correlation_matrix(psd_matrix)

# Perform SVD of the PSD matrix and correlation matrix
U_PSD, S_PSD, V_PSD = analyzer.perform_svd(psd_matrix)
U_corr, S_corr, V_corr = analyzer.perform_svd(correlation_matrix)

# Get the PP index
P1, P2, P3 = analyzer.get_pp_index(correlation_matrix, S_corr)

# Visualize the singular values
visualizer.plot_sigmas(freqs, S_PSD, S_corr, folder_name)

# Identify and print the mode frequency
peaks = analyzer.identify_peaks(freqs, P3, distance=1)

visualizer.plot_pp_index(freqs, P1, P2, P3, peaks, folder_name)

mode_frequency, mode_shape = analyzer.identify_mode_shapes(freqs, U_PSD, peaks)
print(np.array(mode_shape).shape)
print(f"Identified mode frequencies: {mode_frequency} Hz")
print(mode_shape)
MAC = np.zeros((len(peaks), len(peaks)))
for i in range(len(peaks)):
    for j in range(len(peaks)):
        MAC[i, j] = analyzer.compute_mac(mode_shape[i], mode_shape[j])

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
plt.tight_layout()
visualizer._save_figure(fig, "MAC_matrix", folder_name)
# nperseg_list = [256, 512, 1024, 2048, 4096, 8192]
# fig, ax = plt.subplots(len(nperseg_list), 1, figsize=(5, 20))
# for i, nperseg in enumerate(nperseg_list):
#     # Compute PSD using Welch's method
#     freqs, psd_matrix = analyzer.compute_psd_matrix(fs, selected_indices, nperseg=nperseg)

#     # Compute the correlation matrix
#     correlation_matrix = analyzer.compute_correlation_matrix(psd_matrix)

#     # Perform SVD of the PSD matrix and correlation matrix
#     U_PSD, S_PSD, V_PSD = analyzer.perform_svd(psd_matrix)
#     U_corr, S_corr, V_corr = analyzer.perform_svd(correlation_matrix)

#     # Get the PP index
#     P1, P2, P3 = analyzer.get_pp_index(correlation_matrix, S_corr)
#     visualizer.plot_pp_index(freqs, P1, P2, P3, folder_name)

#     ax[i].semilogy(freqs, P1, label="P1")
#     ax[i].semilogy(freqs, P2, label="P2")
#     ax[i].semilogy(freqs, P3, label="P3")
#     ax[i].set_title(f"nperseg = {nperseg}")
#     ax[i].set_ylabel("PP index")
#     ax[i].set_xlim([8, 24])
#     ax[i].legend()
# ax[-1].set_xlabel("Frequency [Hz]")

# fig.tight_layout()
# visualizer._save_figure(fig, "PP_index_compare_results", folder_name)


data = mode_shape

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

plt.tight_layout()
visualizer._save_figure(fig, "PCA", folder_name)