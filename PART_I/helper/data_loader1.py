import numpy as np
import os
import re
from datetime import datetime

class DataLoader:
    """
    A class to load and preprocess data from multiple CSV files in mini-batches.

    This class handles loading data from multiple CSV files, concatenating the data, 
    and preprocessing it to separate time values from sensor data. It supports iterating 
    over the files in mini-batches and accessing specific batches via indexing. The class 
    also includes scaling preprocessing for the sensor data.

    Attributes:
        file_paths (list of str or None): A list of file paths pointing to the CSV files to load.
        folder_path (str or None): The path to a folder containing CSV files to load.
        time (numpy.ndarray or None): An array containing time values extracted from the data.
        data (numpy.ndarray or None): An array containing sensor data with time values removed.
        initialized (bool): A flag indicating whether the data has been successfully loaded and processed.
        batch_size (int): The size of each mini-batch of files.
        scaling_factors (numpy.ndarray or None): The factors used to scale the sensor data.
        total_batches (int): The total number of batches available based on file paths and batch size.
    """

    def __init__(self, selected_indices, file_paths=None, folder_path=None, batch_size=1, scaling_factors=None):
        """
        Initializes the DataLoader with a list of file paths or a folder path.

        You can initialize the DataLoader in one of two ways:
        1. **Direct File Paths:** Provide a list of CSV file paths via the `file_paths` argument.
        2. **Folder Path:** Provide a folder path via the `folder_path` argument. The `get_files_list` method 
           will be automatically called to generate the list of file paths from all files in the folder.

        Args:
            selected_indices (list of int): Indices of the columns to be selected from the sensor data.
            file_paths (list of str, optional): Paths to the CSV files to load.
            folder_path (str, optional): The path to a folder containing CSV files to load.
            batch_size (int, optional): The number of files to include in each mini-batch.
            scaling_factors (numpy.ndarray, optional): The factors used to scale the sensor data.
        """
        self.selected_indices = selected_indices
        self.file_paths = file_paths
        self.folder_path = folder_path
        self.time = None
        self.data = None
        self.initialized = False
        self.batch_size = batch_size
        self.scaling_factors = scaling_factors
        self.time_offset = 0

        # Automatically call get_files_list if a folder path is provided
        if self.folder_path and not self.file_paths:
            self.file_paths = self.get_files_list()

        # Calculate the total number of batches
        self.total_batches = self._calculate_total_batches()

    def __iter__(self):
        """
        Returns the iterator object itself.
        """
        self.current_batch_index = 0  # Reset the index for new iterations
        self.time_offset = 0  # Initialize the time offset
        return self

    def __next__(self):
        """
        Returns the next mini-batch of data.
        """
        if self.current_batch_index > len(self.file_paths) - self.batch_size:
            raise StopIteration

        time, data = self._load_batch(self.current_batch_index)
        if self.batch_size == 1: self.current_batch_index += 1
        self.current_batch_index += self.batch_size // 2
        self.time_offset += len(time)  # Update the time offset
        return time, data

    def __getitem__(self, index):
        """
        Returns a specific mini-batch of data based on the index.

        Args:
            index (int): The index of the batch to retrieve.

        Returns:
            tuple: A tuple containing time values and sensor data.

        Raises:
            IndexError: If the index is out of range.
        """
        if index * self.batch_size >= len(self.file_paths):
            raise IndexError("Index out of range")

        return self._load_batch(index * self.batch_size)

    def __len__(self):
        """
        Returns the total number of batches available.
        """
        return self.total_batches

    def _calculate_total_batches(self):
        """
        Calculate the total number of batches available based on the file paths and batch size.
        """
        if self.batch_size == 1:
            # Special case when batch_size is 1
            return len(self.file_paths)
        elif self.batch_size == 0:
            return 0

        if not self.folder_path:
            return len(self.file_paths) // self.batch_size
        else:
            # Calculating how many steps of half the batch size we can take
            effective_steps = (len(self.file_paths) - self.batch_size) // (self.batch_size // 2)
            return effective_steps + 1
        
    def _load_batch(self, start_index):
        """
        Helper function to load a batch of data starting from the given index.
        """
        batch_file_paths = self.file_paths[start_index:start_index + self.batch_size]
        datas = [np.loadtxt(file_path, delimiter=';') for file_path in batch_file_paths]
        data = np.concatenate(datas, axis=0)

        if len(batch_file_paths) == 1:
            time = np.array(data[:, 0])
            data = np.array((data[:, 1:])[:, self.selected_indices])
        else:
            time = np.linspace(self.time_offset, self.time_offset + 600 * len(batch_file_paths), len(data))
            data = np.array((data[:, 1:])[:, self.selected_indices])

        if self.scaling_factors is not None:
            # Detrend and scale the data
            data = self._detrend_and_scale(data)

        return time, data

    def _detrend_and_scale(self, data):
        """
        Detrends and scales the sensor data using the specified scaling factors.

        Args:
            data (numpy.ndarray): The raw sensor data to be processed.

        Returns:
            numpy.ndarray: The detrended and scaled sensor data.
        """
        # Detrend data
        detrended_data = data - np.mean(data, axis=0)
        # Scale data
        scaled_data = detrended_data * self.scaling_factors
        return scaled_data

    def load_data(self):
        """
        Loads all data from the specified CSV files, concatenates them, and preprocesses the data.
        """
        datas = [np.loadtxt(file_path, delimiter=';') for file_path in self.file_paths]
        self.data = np.concatenate(datas, axis=0)

        if len(self.file_paths) == 1:
            self.time = np.array(self.data[:, 0])
            self.data = np.array(self.data[:, 1:])[:, self.selected_indices]
        else:
            self.time = np.linspace(0, 600 * len(self.file_paths), len(self.data))
            self.data = np.array(self.data[:, 1:])[:, self.selected_indices]

        if self.scaling_factors is not None:
            # Detrend and scale the data
            self.data = self._detrend_and_scale(self.data)

        self.initialized = True
        return self.time, self.data

    def get_files_list(self):
        """
        Retrieve a list of all files in the specified folder, sorted in alphabetical order based on filenames.
        """
        if not os.path.exists(self.folder_path):
            raise FileNotFoundError(f"The folder {self.folder_path} does not exist.")

        # List all files in the folder
        files = [f for f in os.listdir(self.folder_path) if os.path.isfile(os.path.join(self.folder_path, f))]
        files.sort()
        file_paths = [os.path.join(self.folder_path, file_name) for file_name in files]
        return file_paths

# Function to extract date and time from the filename
def extract_date_time(filename, date_time_pattern = r"\d{4}_\d{2}_\d{2}_\d{6}"): # Pattern for date and time in the format YYYY_MM_DD_HHMMSS
    # Search for the date and time pattern in the filename
    match = re.search(date_time_pattern, filename)
    
    if match:
        # Extract the matched date and time string
        date_time_str = match.group(0)
        # Convert the string to a datetime object
        return datetime.strptime(date_time_str, "%Y_%m_%d_%H%M%S")
    else:
        # Return None if no match is found
        return None