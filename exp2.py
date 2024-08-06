import numpy as np
from helper.data_loader import DataLoader
from helper.signal_transformer import SignalTransformer
from helper.frequency_analyzer import FrequencyAnalyzer
from helper.visualizer import Visualizer
from scipy.ndimage import gaussian_filter1d


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
file_paths = file_paths_24[2:]
location = "Lello_Jul24_stairs"
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
transformer = SignalTransformer(time, data, scaling_factors)
detrended_data = transformer.detrend_and_scale()

# Initialize the Visualizer with time and frequencies
# Create a unique folder name based on hyperparameters
time_window_size = 600 * len(file_paths)
processing_method = "Welch"
folder_name = f"loc_{location}_wind_{time_window_size}_meth_{processing_method}_nperseg_vary"

visualizer = Visualizer(time, output_dir="results")
labels=["Base X", "Base Y", "Base Z", "Stair 1 X", "Stair 1 Y", "Stair 1 Z", "Stair 2 Z", "Vitral Z"]

analyzer = FrequencyAnalyzer(detrended_data, time)

fs = 1 / np.mean(np.diff(time))

nperseg_list = [256, 512, 1024, 2048, 4096, 8192]
fig, ax = plt.subplots(len(nperseg_list), 1, figsize=(5, 20))
for i, nperseg in enumerate(nperseg_list):
    # Compute PSD using Welch's method
    freqs, psd_matrix = analyzer.compute_psd_matrix(fs, selected_indices, nperseg=nperseg)

    # Compute the correlation matrix
    correlation_matrix = analyzer.compute_correlation_matrix(psd_matrix)

    # Perform SVD of the PSD matrix and correlation matrix
    U_PSD, S_PSD, V_PSD = analyzer.perform_svd(psd_matrix)
    U_corr, S_corr, V_corr = analyzer.perform_svd(correlation_matrix)

    # Get the PP index
    P1, P2, P3 = analyzer.get_pp_index(correlation_matrix, S_corr)

    # Identify and print the mode frequency
    peaks = analyzer.identify_peaks(freqs, P3, distance=1)

    # Apply Gaussian smoothing
    P1_smooth = gaussian_filter1d(P1, sigma=14)
    P2_smooth = gaussian_filter1d(P2, sigma=14)
    P3_smooth = gaussian_filter1d(P3, sigma=14)


    # Plot non-smoothed curves
    ax[i].semilogy(freqs, P1, label="P1 (Raw)", linestyle='dotted', color='orange')
    ax[i].semilogy(freqs, P2, label="P2 (Raw)", linestyle='dotted', color='green')
    ax[i].semilogy(freqs, P3, label="P3 (Raw)", linestyle='dotted', color='blue')

    # Plot smoothed curves
    ax[i].semilogy(freqs, P1_smooth, label="P1 (Smoothed)", linestyle='solid', color='orange')
    ax[i].semilogy(freqs, P2_smooth, label="P2 (Smoothed)", linestyle='solid', color='green')
    ax[i].semilogy(freqs, P3_smooth, label="P3 (Smoothed)", linestyle='solid', color='blue')

    # Scatter plot for P3 at the peak points
    ax[i].scatter(freqs[peaks], P3[peaks], color='red', marker='x', label='Peaks')

    ax[i].set_title(f"nperseg = {nperseg}")
    ax[i].set_ylabel("PP index")
    ax[i].set_xlim([8, 24])
    # ax[i].legend()
ax[-1].set_xlabel("Frequency [Hz]")

fig.tight_layout()
visualizer._save_figure(fig, "PP_index_compare_results", folder_name)