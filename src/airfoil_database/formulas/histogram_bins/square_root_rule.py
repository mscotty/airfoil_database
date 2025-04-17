import numpy as np

def square_root_rule(data):
    """
    @brief Compute bin width and bin count using the Square Root Rule.
    
    @param[in] data Array-like, numeric data for histogram binning.
    
    @return A tuple containing bin width and bin count.
    """
    bin_width = (max(data) - min(data)) / np.sqrt(len(data))
    bin_count = int(np.sqrt(len(data)))
    return bin_width, bin_count