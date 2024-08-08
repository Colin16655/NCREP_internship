import numpy as np
import os
import re
from datetime import datetime

class DataLoader:
    """
    A class to load and preprocess data from multiple CSV files.

    This class is designed to handle the loading of data from multiple CSV files,
    concatenate the data, and preprocess it to separate time values from sensor data.

    Attributes:
        file_paths (list of str): A list of file paths pointing to the CSV files to load.
        time (numpy.ndarray or None): An array containing time values extracted from the data. 
                                      This attribute is set to None if not yet initialized.
        data (numpy.ndarray or None): An array containing sensor data, with time values removed.
                                       This attribute is set to None if not yet initialized.
        initialized (bool): A flag indicating whether the data has been successfully loaded and processed.
    """

    def __init__(self, file_paths):
        """
        Initializes the DataLoader with a list of file paths.

        Args:
            file_paths (list of str): Paths to the CSV files to load. Each file should be a CSV with 
                                      the first column representing time and the subsequent columns 
                                      representing sensor data.
        """
        self.file_paths = file_paths
        self.time = None
        self.data = None
        self.initialized = False

    def load_data(self):
        """
        Loads data from the specified CSV files, concatenates them, and preprocesses the data.

        This method reads the CSV files specified in `file_paths`, concatenates the data from all files,
        and separates the time values from the sensor data. If only one file is provided, it extracts
        time values directly from the first column of the data. If multiple files are provided, it generates
        a time array spanning the total duration covered by all files.

        Returns:
            tuple: A tuple containing two numpy.ndarray objects:
                - The first array contains time values.
                - The second array contains sensor data with the time values removed.

        Raises:
            FileNotFoundError: If any of the specified CSV files do not exist.
            ValueError: If the CSV files have incompatible dimensions or formats.
        """
        datas = [np.loadtxt(file_path, delimiter=';') for file_path in self.file_paths]
        self.data = np.concatenate(datas, axis=0)
        
        if len(self.file_paths) == 1:
            self.time = self.data[:, 0]
            self.data = self.data[:, 1:]
        else:
            self.time = np.linspace(0, 600 * len(self.file_paths), len(self.data))
            self.data = self.data[:, 1:]
        
        self.initialized = True
        return self.time, self.data


def get_files_list(folder_path):
    """
    Retrieve a list of all files in a specified folder, sorted in alphabetical order based on filenames.

    This function lists all files in the given folder and sorts them lexicographically by filename. 
    It then returns the full file paths for each file in the sorted order.

    Parameters:
        folder_path (str): The path to the folder containing the files. This should be a valid directory path.

    Returns:
        list of str: A list of full file paths for the files in the folder, sorted alphabetically by filename.
    """
    # List all files in the folder
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    # Sort files by filename (assumes filenames can be sorted lexicographically)
    files.sort()
    # Create full file paths
    sorted_file_paths = [os.path.join(folder_path, file_name) for file_name in files]
    return sorted_file_paths

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