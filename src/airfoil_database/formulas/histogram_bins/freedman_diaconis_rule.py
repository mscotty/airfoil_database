import numpy as np

def freedman_diaconis_rule(data):
    """!
    @brief Calculate bin width and count using Freedman-Diaconis Rule.

    @param[in] data Array-like, numeric data for histogram binning.
    
    @return A tuple containing bin width and bin count.
    """
    q25, q75 = np.percentile(data, [25, 75])  # Interquartile range (IQR)
    iqr = q75 - q25
    bin_width = 2 * iqr / (len(data) ** (1/3))  # Freedman-Diaconis formula
    bin_count = max(1, int((data.max() - data.min()) / bin_width))  # Avoid 0 bins
    return bin_width, bin_count