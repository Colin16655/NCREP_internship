import numpy as np
from helper.data_loader import DataLoader
from helper.processor import ModalFrequencyAnalyzer, PeakPicker
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
file_paths = file_paths_23[:1]
location = "Lello_Jul23_stairs"
selected_indices = [3, 4, 5, 6]  # Indices of the selected sensors : stair
distance0 = 1 # for pp method 0
distance1 = 1 # for peak picking method 
sigma = 16 # for pp method
ranges_to_check = [(10.5, 12.5), (15.5, 17.5), (21.5, 23.5)] # for pp method
### USER ###

# Define scaling factors for the sensor data
scaling_factors = np.array([0.4035*1000, 0.4023*1000, 0.4023*1000, 0.4023*1000, 0.4015*1000, 0.4014*1000, 0.4007*1000, 0.4016*1000])[selected_indices]

# Load the data
loader = DataLoader(selected_indices, file_paths=file_paths)
time, data = loader.load_data()
print("Time shape:", time.shape)
print("Data shape:", data.shape)

# Initialize the Visualizer with time and frequencies
# Create a unique folder name based on hyperparameters
time_window_size = 600 * len(file_paths)
processing_method = "Welch"
folder_name = f"loc_{location}_wind_{time_window_size}_meth_{processing_method}_vary"

visualizer = Visualizer(time, output_dir="results")
labels = np.array(["Base X", "Base Y", "Base Z", "Stair 1 X", "Stair 1 Y", "Stair 1 Z", "Stair 2 Z", "Vitral Z"])[selected_indices]

analyzer = ModalFrequencyAnalyzer(data, time)

nperseg_list = [512, 1024, 2048]
fig, ax = plt.subplots(len(nperseg_list), 2, figsize=(8, 10))
for i, nperseg in enumerate(nperseg_list):
    # Compute PSD using Welch's method
    freqs, _ = analyzer.compute_psd_matrix(nperseg=nperseg)

    # Compute the correlation matrix
    analyzer.compute_coherence_matrix()

    # Perform SVD of the PSD matrix and correlation matrix
    U_PSD, S_PSD, V_PSD = analyzer.perform_svd_psd()
    U_coh, S_coh, V_coh = analyzer.perform_svd_coherence()

    # Get the PP index
    P1, P2, P3 = analyzer.compute_pp_index()

    # Identify and print the mode frequency
    peak_picker = PeakPicker(analyzer)
    peaks = peak_picker.identify_peaks_1(P3, S_PSD[:,0], distance=distance1, sigma=sigma, ranges_to_check=ranges_to_check)

    # Apply Gaussian smoothing
    P1_smooth = gaussian_filter1d(P1, sigma=sigma)
    P2_smooth = gaussian_filter1d(P2, sigma=sigma)
    P3_smooth = gaussian_filter1d(P3, sigma=sigma)


    # Plot non-smoothed curves
    ax[i,0].semilogy(freqs, P1, label="P1 (Raw)", linestyle='dotted', color='orange')
    ax[i,0].semilogy(freqs, P2, label="P2 (Raw)", linestyle='dotted', color='green')
    ax[i,0].semilogy(freqs, P3, label="P3 (Raw)", linestyle='dotted', color='blue')

    # Plot smoothed curves
    ax[i,0].semilogy(freqs, P1_smooth, linestyle='solid', color='orange')
    ax[i,0].semilogy(freqs, P2_smooth, linestyle='solid', color='green')
    ax[i,0].semilogy(freqs, P3_smooth, linestyle='solid', color='blue')

    # Scatter plot for P3 at the peak points
    ax[i,0].scatter(freqs[peaks], P3[peaks], color='black', marker='x', label='Peaks')

    ax[i,0].set_title(f"nperseg = {nperseg}")
    ax[i,0].set_ylabel("PP index")
    ax[i,0].set_xlim([8, 24])
    

    # Plot the PSD
    num_singular_values = S_PSD.shape[1]  # Number of singular values
    colors = ['orange', 'green', 'blue', 'black']
    freq_mask = (freqs >= 8) & (freqs <= 24)
    S_PSD_filt = S_PSD[freq_mask, :]
    freqs_filt = freqs[freq_mask]
    for j in range(num_singular_values):
        original_values = S_PSD_filt[:, j]
        filtered_values = gaussian_filter1d(original_values, sigma=sigma)

        # Apply frequency mask to the original and filtered values
        original_values_filtered = original_values
        filtered_values_filtered = filtered_values

        # Plot non-smoothed curves
        ax[i,1].semilogy(freqs_filt, original_values_filtered, label=f'$\sigma_{j+1}$ (Raw)', linestyle='dotted', color=colors[j])
        # Plot smoothed curves
        ax[i,1].semilogy(freqs_filt, filtered_values_filtered, linestyle='solid', color=colors[j])

    # Scatter plot for first sigma at the peak points
    peak_values = S_PSD[peaks, 0]
    ax[i,1].scatter(freqs[peaks], peak_values, color='black', marker='x', label='Peaks')

    ax[i,1].set_ylabel("Singular Value")

    ax[i,1].set_title(f"nperseg = {nperseg}")


ax[-1,0].set_xlabel("Frequency [Hz]")
ax[-1,1].set_xlabel("Frequency [Hz]")
ax[0,0].legend(framealpha=0.5)
ax[0,1].legend(framealpha=0.5)

fig.tight_layout()
visualizer._save_figure(fig, f"PP_index_compare_n_sigma_{sigma}_nperseg", folder_name)