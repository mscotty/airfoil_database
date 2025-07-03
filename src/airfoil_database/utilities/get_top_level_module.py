import os
import inspect
import sys
import importlib.util

def get_package_root(package_name=None):
    """
    Returns the filepath to the root directory of a specified package,
    or the current package if none is specified.

    Args:
        package_name (str, optional): The name of the package to find.
            If None, attempts to find the package containing the calling module.

    Returns:
        str: The filepath to the package's root directory, or None if not found.
    """
    if package_name:
        # Try to find the specified package
        try:
            package = importlib.import_module(package_name)
            if hasattr(package, '__path__'):
                return os.path.abspath(package.__path__[0])
            elif hasattr(package, '__file__'):
                return os.path.dirname(os.path.abspath(package.__file__))
        except (ImportError, AttributeError):
            return None
    else:
        # Try to find the package containing the calling module
        frame = inspect.currentframe().f_back
        try:
            module_file = frame.f_globals.get('__file__')
            if not module_file:
                return None
                
            module_path = os.path.dirname(os.path.abspath(module_file))
            
            # Walk up the directory tree looking for a __init__.py file
            current_path = module_path
            while current_path and current_path != os.path.dirname(current_path):
                if os.path.isfile(os.path.join(current_path, '__init__.py')):
                    parent_path = os.path.dirname(current_path)
                    # Check if the parent also has an __init__.py (is part of the package)
                    if os.path.isfile(os.path.join(parent_path, '__init__.py')):
                        current_path = parent_path
                        continue
                    return current_path
                current_path = os.path.dirname(current_path)
            
            # If we couldn't find a package structure, return the module directory
            return module_path
        finally:
            del frame

def get_project_root():
    """
    Attempts to find the project root directory by looking for common project files.
    
    Returns:
        str: The filepath to the project root directory, or None if not found.
    """
    # Common files/directories that might indicate a project root
    root_indicators = [
        'setup.py', 'pyproject.toml', '.git', '.gitignore', 
        'README.md', 'requirements.txt'
    ]
    
    # Start from the directory of the calling module
    frame = inspect.currentframe().f_back
    try:
        module_file = frame.f_globals.get('__file__')
        if not module_file:
            return None
            
        current_path = os.path.dirname(os.path.abspath(module_file))
        
        # Walk up the directory tree looking for root indicators
        while current_path and current_path != os.path.dirname(current_path):
            for indicator in root_indicators:
                if os.path.exists(os.path.join(current_path, indicator)):
                    return current_path
            current_path = os.path.dirname(current_path)
        
        return None
    finally:
        del frame

def get_project_root_jupyter():
    """Alternative version for Jupyter notebooks"""
    import os
    
    current_path = os.getcwd()  # Start from current working directory
    print(f"Starting from: {current_path}")
    
    root_indicators = ['setup.py', 'pyproject.toml', '.git', '.gitignore', 
                      'README.md', 'requirements.txt']
    
    while current_path and current_path != os.path.dirname(current_path):
        for indicator in root_indicators:
            indicator_path = os.path.join(current_path, indicator)
            if os.path.exists(indicator_path):
                print(f"Found indicator: {indicator} at {current_path}")
                return current_path
        current_path = os.path.dirname(current_path)
    
    print("No project root found")
    return None


def get_module_path(module_name=None):
    """
    Returns the directory path of a specified module or the calling module.
    
    Args:
        module_name (str, optional): The name of the module to find.
            If None, returns the directory of the calling module.
            
    Returns:
        str: The filepath to the module's directory, or None if not found.
    """
    if module_name:
        # Try to find the specified module
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, '__file__'):
                return os.path.dirname(os.path.abspath(module.__file__))
        except ImportError:
            return None
    else:
        # Get the directory of the calling module
        frame = inspect.currentframe().f_back
        try:
            module_file = frame.f_globals.get('__file__')
            if module_file:
                return os.path.dirname(os.path.abspath(module_file))
            return None
        finally:
            del frame

def add_to_python_path(directory):
    """
    Adds a directory to the Python path if it's not already there.
    
    Args:
        directory (str): The directory to add to the Python path.
        
    Returns:
        bool: True if the directory was added, False if it was already in the path.
    """
    directory = os.path.abspath(directory)
    if directory not in sys.path:
        sys.path.insert(0, directory)
        return True
    return False

# Example usage:
if __name__ == "__main__":
    # Get the path of the current module
    current_module_path = get_module_path()
    print(f"Current module path: {current_module_path}")
    
    # Get the root of the package containing this module
    package_root = get_package_root()
    print(f"Package root: {package_root}")
    
    # Get the project root
    project_root = get_project_root()
    print(f"Project root: {project_root}")
    
    # Example of adding the project root to the Python path
    if project_root:
        added = add_to_python_path(project_root)
        print(f"Added project root to Python path: {added}")
    
    # Example of finding a specific package
    numpy_path = get_package_root('numpy')
    print(f"NumPy package path: {numpy_path}")
