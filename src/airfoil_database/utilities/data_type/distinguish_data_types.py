import os

import pandas as pd

from DASC500.utilities.data_type.is_float import is_float

def distinguish_data_types(input):
    """!
    @brief Check a pandas dataframe for the types of data it stores. Currently only supports numeric or string data.
    """
    if isinstance(input, pd.DataFrame):
        df = input
    elif isinstance(input, str):
        if os.path.exists(str):
            df = pd.read_csv(input)
        else:
            raise FileNotFoundError(f'The provided file {input} does not exist.')
    
    # Initialize a dictionary to store the column types
    column_types = {}
    
    for column in df.columns:
        # Check if all values in the column can be numeric using pandas built-in function
        if pd.api.types.is_numeric_dtype(df[column]):
            column_types[column] = 'Numeric'
        else:
            # If that built-in function fails, rely on checking each individual value (remove any empties)
            if all(is_float(value) for value in df[column].dropna()):
                column_types[column] = 'Numeric'
            else:
                column_types[column] = 'String'
    
    return column_types