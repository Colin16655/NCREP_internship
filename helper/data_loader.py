import numpy as np

class DataLoader:
    """
    A class to load data from multiple CSV files.

    Attributes:
        file_paths (list of str): Paths to the CSV files to load.
        data (numpy.ndarray): The data loaded from the CSV files.
    """

    def __init__(self, file_paths):
        """
        Initializes the DataLoader with a list of file paths.

        Parameters:
            file_paths (list of str): Paths to the CSV files to load.
        """
        self.file_paths = file_paths
        self.time = None
        self.data = None
        self.initialized = False

    def load_data(self):
        """
        Loads data from the specified CSV files and concatenates them.

        Returns:
            numpy.ndarray: An array containing the time values.
            numpy.ndarray: An array containing the sensor data, with time values removed.
    """
        datas = [np.loadtxt(file_path, delimiter=";") for file_path in self.file_paths]
        self.data = np.concatenate(datas, axis=0)
        if len(self.file_paths) == 1:
            self.time = self.data[:, 0]
            self.data = self.data[:, 1:]
        else:
            self.time = np.linspace(0, 600*len(self.file_paths), len(self.data))
            self.data = self.data[:, 1:]
        self.initialized = True
        return self.time, self.data
