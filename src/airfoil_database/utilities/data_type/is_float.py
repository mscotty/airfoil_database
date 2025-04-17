def is_float(value):
    """!
    @brief Use a simple try except loop to attempt conversion of input value to a float
    """
    try:
        float(value)
        return True
    except ValueError:
        return False