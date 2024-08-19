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
    also includes preprocessing methods for scaling and detrending sensor data.

    Attributes:
        file_paths (list of str or None): A list of file paths pointing to the CSV files to load.
        folder_path (str or None): The path to a folder containing CSV files to load.
        time (numpy.ndarray or None): An array containing time values extracted from the data.
        data (numpy.ndarray or None): An array containing sensor data with time values removed.
        initialized (bool): A flag indicating whether the data has been successfully loaded and processed.
        batch_size (int): The size of each mini-batch of files.
        scaling_factors (numpy.ndarray or None): The factors used to scale the sensor data.
        total_batches (int): The total number of batches available based on file paths and batch size.
        time_offset (int): The offset to adjust time values when multiple files are concatenated.
    """

    def __init__(self, selected_indices, folder_path=None, batch_size=1, scaling_factors=None):
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
        self.folder_path = folder_path
        self.batch_size = batch_size
        self.scaling_factors = scaling_factors
        self.time = None
        self.data = None
        self.time_offset = 0
        self.file_paths = self.get_files_list()

        self.csv_files = np.array()

        # Calculate the total number of batches
        self.total_batches = len(self.file_paths) // self.batch_size # incomplete batches are ignored

    def __iter__(self):
        """
        Returns the iterator object itself, initializing the current batch index and time offset.
        """
        self.current_batch_index = 0  # Reset the index for new iterations
        self.time_offset = 0  # Reset the time offset for new iterations
        return self

    def __next__(self):
        """
        Returns the next mini-batch of data.

        Raises:
            StopIteration: When there are no more batches to return.
        """
        if self.current_batch_index >= len(self.file_paths) - self.batch_size + 1:
            raise StopIteration

        time, data = self._load_batch(self.current_batch_index)
        self.current_batch_index += self.batch_size
        self.time_offset = time[-1] + np.mean(np.diff(time))  # Update the time offset
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

    def _load_batch(self, start_index):
        """
        Helper function to load a batch of data starting from the given index.

        Args:
            start_index (int): The starting index of the batch to load.

        Returns:
            tuple: A tuple containing time values and sensor data.
        """
        batch_file_paths = self.file_paths[start_index:start_index + self.batch_size]
        datas = [np.loadtxt(file_path, delimiter=';') for file_path in batch_file_paths]
        data = np.concatenate(datas, axis=0)

        if len(batch_file_paths) == 1:
            time = np.array(data[:, 0])
        else:
            time = np.linspace(self.time_offset, self.time_offset + 600 * len(batch_file_paths), len(data))
        data = np.array((data[:, 1:])[:, self.selected_indices])

        if self.scaling_factors is not None:
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

    def get_files_list(self):
        """
        Retrieves a list of all files in the specified folder, sorted in alphabetical order based on filenames.

        Returns:
            list of str: A list of file paths in the specified folder.

        Raises:
            FileNotFoundError: If the folder does not exist.
        """
        if not os.path.exists(self.folder_path):
            raise FileNotFoundError(f"The folder {self.folder_path} does not exist.")

        # List all files in the folder
        files = [f for f in os.listdir(self.folder_path) if os.path.isfile(os.path.join(self.folder_path, f))]
        files.sort()
        file_paths = [os.path.join(self.folder_path, file_name) for file_name in files]
        return file_paths

# Function to extract date and time from the filename
def extract_date_time(filename, date_time_pattern=r"\d{4}_\d{2}_\d{2}_\d{6}"): 
    """
    Extracts the date and time from the filename based on a specified pattern.

    Args:
        filename (str): The filename from which to extract the date and time.
        date_time_pattern (str, optional): A regular expression pattern to match the date and time in the filename.
            Defaults to a pattern matching the format YYYY_MM_DD_HHMMSS.

    Returns:
        datetime.datetime or None: The extracted date and time as a datetime object, or None if no match is found.
    """
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
