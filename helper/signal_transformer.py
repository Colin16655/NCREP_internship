import numpy as np

class SignalTransformer:
    """
    A class to transform and preprocess signal data.

    Attributes:
        data (numpy.ndarray): The raw data loaded from CSV files.
        scaling_factors (numpy.ndarray): The factors used to scale the sensor data.
        sensor_data (numpy.ndarray): The extracted sensor data from the raw data.
        time (numpy.ndarray): The time vector corresponding to the sensor data.
        detrended_data (numpy.ndarray): The detrended and scaled sensor data.
    """

    def __init__(self, time, data, scaling_factors):
        """
        Initializes the SignalTransformer with raw data and scaling factors.

        Parameters:
            data (numpy.ndarray): The raw data loaded from CSV files.
            scaling_factors (numpy.ndarray): The factors used to scale the sensor data.
        """
        self.scaling_factors = scaling_factors
        self.sensor_data = data
        self.time = time
        self.detrended_data = None
        self.initialized = False

    def detrend_and_scale(self):
        """
        Detrends and scales the sensor data.

        Returns:
            numpy.ndarray: The detrended and scaled sensor data.
        """
        self.detrended_data = self.sensor_data - np.mean(self.sensor_data, axis=0)
        self.detrended_data *= self.scaling_factors
        self.initialized = True
        return self.detrended_data
