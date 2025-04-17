def print_series_mode(mode_in):
    """!
    @brief Format and print the modes of a pandas Series.

    @param[in] mode_in Pandas Series containing the mode(s) of a dataset.
    
    @return A formatted string listing all modes.
    """
    
    vals = mode_in.values
    return f"{len(vals)} Mode(s) found: {', '.join(map(str, vals))}"
