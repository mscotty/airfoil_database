import os

def get_files_starting_with(directory, prefix, include_extension=True):
    """
    Gets a list of files in the specified directory that start with the given prefix.
    
    Args:
        directory (str): The directory path to search in.
        prefix (str): The prefix to match file names against.
        include_extension (bool, optional): If True, returns full filenames with extensions.
                                           If False, returns filenames without extensions.
                                           Defaults to True.
    
    Returns:
        list: A list of filenames (with or without extensions) that start with the given prefix.
    """
    if not os.path.isdir(directory):
        raise ValueError(f"Directory '{directory}' does not exist")
    
    matching_files = []
    
    # Get all files in the directory
    for filename in os.listdir(directory):
        # Check if it's a file (not a directory) and starts with the prefix
        if os.path.isfile(os.path.join(directory, filename)) and filename.startswith(prefix):
            if include_extension:
                matching_files.append(filename)
            else:
                # Remove the extension if include_extension is False
                name_without_ext, _ = os.path.splitext(filename)
                matching_files.append(name_without_ext)
    
    return matching_files