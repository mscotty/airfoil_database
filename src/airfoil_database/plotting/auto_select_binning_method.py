import numpy as np

def auto_select_binning_method(data):
    """!
    @brief Automatically select a binning method based on data characteristics.
    
    @param[in] data Array-like, numeric data for determining the binning method.
    
    @return The name of the binning method as a string ("Freedman-Diaconis", "Square Root", or "Sturges").    
    """

    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    data_range = data.max() - data.min()
    if iqr > 0.1 * data_range:
        return "Freedman-Diaconis"
    elif len(data) < 100:
        return "Square Root"
    else:
        return "Sturges"