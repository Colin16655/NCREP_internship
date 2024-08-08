import numpy as np

class Preprocessor:
    """
    A class to transform and preprocess signal data, including detrending and scaling.

    This class handles the preprocessing of sensor data by detrending and scaling it using specified factors.

    Attributes:
        time (numpy.ndarray): The time vector corresponding to the sensor data.
        data (numpy.ndarray): The raw sensor data.
        scaling_factors (numpy.ndarray): The factors used to scale the sensor data.
        sensor_data (numpy.ndarray): The raw sensor data, as initialized.
        detrended_data (numpy.ndarray or None): The detrended and scaled sensor data. Initialized to None before processing.
        initialized (bool): A flag indicating whether the data has been processed.
    """

    def __init__(self, time, data, scaling_factors):
        """
        Initializes the SignalTransformer with time, raw data, and scaling factors.

        Parameters:
            time (numpy.ndarray): The time vector corresponding to the sensor data.
            data (numpy.ndarray): The raw sensor data.
            scaling_factors (numpy.ndarray): The factors used to scale the sensor data.
        """
        self.time = time
        self.sensor_data = data
        self.scaling_factors = scaling_factors
        self.detrended_data = None
        self.initialized = False

    def detrend_and_scale(self):
        """
        Detrends and scales the sensor data using the specified scaling factors.

        The method removes the mean from the sensor data (detrending) and then scales it according to the scaling factors.
        The processed data is stored in the `detrended_data` attribute and is returned by the method.

        Returns:
            numpy.ndarray: The detrended and scaled sensor data.
        """
        self.detrended_data = self.sensor_data - np.mean(self.sensor_data, axis=0)
        self.detrended_data *= self.scaling_factors
        self.initialized = True
        return self.detrended_data

