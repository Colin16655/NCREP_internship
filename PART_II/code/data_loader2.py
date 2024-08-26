import os
import numpy as np

class DataLoader:
    def __init__(self, selected_indices, folder_path=None, batch_size=100, scaling_factors=None):
        if folder_path is None:
            raise ValueError("folder_path must be provided.")

        self.file_paths = self.get_files_list(folder_path)
        self.batch_size = batch_size
        self.scaling_factors = None
        self.current_csv = None  # Array to hold data from 2 contiguous CSV files
        self.current_csv_idx = 0  # Index for the current position in current_csv
        self.file_paths_idx = 0  # Index for the next file to be loaded
        self.csv_size = 60000  # Default size for one CSV file's data

        self.selected_indices = selected_indices
        self.scaling_factors = scaling_factors

    def _load_data(self, file_paths):
        try:
            data = [np.loadtxt(file_path, delimiter=';') for file_path in file_paths]
            data = np.concatenate(data, axis=0) if len(file_paths) > 1 else data[0]
            data = np.array(data[:, 1:])[:, self.selected_indices]  # Exclude the time column

            if self.scaling_factors is not None:
                data = self._detrend_and_scale(data)

            return data
        except Exception as e:
            raise ValueError(f"Error loading data from files {file_paths}: {e}")

    def __len__(self):
        total_lines = self.csv_size * len(self.file_paths)
        return total_lines // self.batch_size  # Incomplete batches are ignored

    def _load_next_file(self):
        if self.current_csv is None:
            # Initialize current_csv with the first file's data
            file_path = self.file_paths[self.file_paths_idx]
            data = self._load_data([file_path])
            self.current_csv = np.full((2 * self.csv_size, data.shape[1]), np.nan)
            self.current_csv[:len(data)] = data
            self.current_csv_idx = 0
            self.file_paths_idx += 1
        else:
            # Load the next file and update the current_csv
            if self.file_paths_idx < len(self.file_paths):
                file_path = self.file_paths[self.file_paths_idx]
                next_data = self._load_data([file_path])
                self.file_paths_idx += 1

                # Shift the existing data and append the new data

                # Check if the second file in current_csv has been loaded completely (case if L = 66000, 11')
                if self.current_csv_idx >= self.csv_size*2:
                    self.current_csv[:self.csv_size] = next_data
                    next_next_data = self._load_data([self.file_paths[self.file_paths_idx]])
                    self.current_csv[self.csv_size:] = next_next_data
                    self.current_csv_idx -= self.csv_size
                    self.file_paths_idx += 1
                else:
                    if not np.isnan(self.current_csv[-1][0]) : self.current_csv[:self.csv_size] = self.current_csv[-self.csv_size:]
                    self.current_csv[self.csv_size:] = next_data
            else:
                # if self.current_csv_idx + self.batch_size > 2 * self.csv_size:
                raise StopIteration

    def __iter__(self):
        """
        Returns the iterator object itself, initializing the current batch index.
        """
        self.current_csv = None  # Reset current_csv to reload files
        self.file_paths_idx = 0  # Reset to start from the first file
        self.current_csv_idx = 0  # Reset the current position in current_csv

        # Preload the first two files
        self._load_next_file()
        self._load_next_file()

        return self

    def __next__(self):
        # Check if we need to load more data
        if self.current_csv_idx + self.batch_size > 2 * self.csv_size - 1:
            self._load_next_file()
            self.current_csv_idx -= self.csv_size
        # Prepare the batch
        batch_data = self.current_csv[self.current_csv_idx:self.current_csv_idx + self.batch_size]

        # Update the current index
        self.current_csv_idx += self.batch_size
        # Ignore incomplete batches
        if batch_data.shape[0] < self.batch_size:
            raise StopIteration

        return batch_data

    @staticmethod
    def get_files_list(folder_path):
        """
        Retrieves a list of all files in the specified folder, sorted in alphabetical order based on filenames.

        Returns:
            list of str: A list of file paths in the specified folder.

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
        Detrends and scales the sensor data using the specified scaling factors.

        Args:
            data (numpy.ndarray): The raw sensor data to be processed.

        Returns:
            numpy.ndarray: The detrended and scaled sensor data.
        """
        detrended_data = data - np.mean(data, axis=0)
        if self.scaling_factors is not None:
            scaled_data = detrended_data * self.scaling_factors
        else:
            scaled_data = detrended_data
        return scaled_data
    
    def load_data(self):
        """
        Loads all data from the specified CSV files, concatenates them, and preprocesses the data.
        """
        datas = [np.loadtxt(file_path, delimiter=';') for file_path in self.file_paths]
        data = np.concatenate(datas, axis=0)

        if len(self.file_paths) == 1:
            time = np.array(data[:, 0])
            data = np.array(data[:, 1:])[:, self.selected_indices]
        else:
            time = np.linspace(0, 600 * len(self.file_paths), len(data))
            data = np.array(data[:, 1:])[:, self.selected_indices]
        if self.scaling_factors is not None:
            # Detrend and scale the data
            data = self._detrend_and_scale(data)
        return time, data
