import numpy as np

def sturges_rule(data):
    """!
    @brief Compute bin width and bin count using Sturges' Rule.

    @param[in] data Array-like, numeric data for histogram binning.
    
    @return A tuple containing bin width and bin count.
    """
    
    bin_count = int(np.ceil(np.log2(len(data))) + 1)
    bin_width = (np.max(data) - np.min(data)) / bin_count
    return bin_width, bin_count