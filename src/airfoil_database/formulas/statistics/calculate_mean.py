import numpy as np
import pandas as pd


def calculate_mean(data):
    """
    Manually calculates the mean and sample variance of a NumPy array or Pandas Series.

    Parameters:
        data (np.ndarray or pd.Series): Input data.

    Returns:
        tuple: mean
    """
    if not isinstance(data, (np.ndarray, pd.Series)):
        raise TypeError("Input must be a NumPy array or Pandas Series")

    n = len(data)
    if n < 2:
        raise ValueError("Sample variance requires at least two data points")
    
    return sum(data) / n