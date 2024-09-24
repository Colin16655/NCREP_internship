import os
import numpy as np

class DataLoader:
    def __init__(self, selected_indices, folder_path=None, batch_size=100, scaling_factors=None):
        """
        Loads and processes sensor data in batches, applying optional detrending and scaling.
        
        Args:
            selected_indices (list): Indices of the sensor channels to select.
            folder_path (str): Path to the folder containing the data files.
            batch_size (int, optional): Number of data points per batch. Defaults to 100.
            scaling_factors (list, optional): Factors to scale the data by. Defaults to None.
        """
        self.selected_indices = selected_indices
        self.scaling_factors = scaling_factors
        self.batch_size = batch_size
        
        if folder_path is None:
            raise ValueError("folder_path must be provided.")

        self.file_paths = self.get_files_list(folder_path)
        self.data_list = self._load_data(self.file_paths) 
        self.current_idx = 0 

    def _load_data(self, file_paths):
        """
        Loads data from file paths, applies a window function, and selects the required channels.
        
        Args:
            file_paths (list): Paths to the data files.
        
        Returns:
            numpy.ndarray: Loaded and processed data.
        """
        # apply Hamming window to the batch
        window = np.ones(60000)
        data = [np.loadtxt(file_path, delimiter=';') * np.array([window]).T for file_path in file_paths]
        data = np.concatenate(data, axis=0) if len(file_paths) > 1 else data[0]
        data = np.array(data[:, 1:])[:, self.selected_indices]  # Exclude the time column

        if self.scaling_factors is not None:
            data = self._detrend_and_scale(data)

        return data

    def __len__(self):
        return len(self.data_list) // self.batch_size

    def __iter__(self):
        """
        Returns the iterator object itself, initializing the current batch index.
        """
        self.current_idx = 0 
        return self

    def __next__(self):
        start_idx = self.current_idx
        end_idx = start_idx + self.batch_size
        if end_idx > len(self.data_list):
            raise StopIteration
        self.current_idx = end_idx

        # Prepare the batch
        batch_data = self.data_list[start_idx:end_idx]
        return batch_data

    @staticmethod
    def get_files_list(folder_path):
        """
        Gets a list of files in the specified folder, sorted alphabetically.
        
        Args:
            folder_path (str): Path to the folder.
        
        Returns:
            list of str: List of file paths.
        
        Raises:
            FileNotFoundError: If the folder does not exist.
        """
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"The folder {folder_path} does not exist.")

        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        files.sort()
        file_paths = [os.path.join(folder_path, file_name) for file_name in files]
        return file_paths

    def _detrend_and_scale(self, data):
        """
        Detrends and scales the data using the provided scaling factors.
        
        Args:
            data (numpy.ndarray): Data to be processed.
        
        Returns:
            numpy.ndarray: Detrended and scaled data.
        """
        detrended_data = data - np.mean(data, axis=0)
        if self.scaling_factors is not None:
            scaled_data = detrended_data * self.scaling_factors
        else:
            scaled_data = detrended_data
        return scaled_data
    