import numpy as np
import logging

def pointcloud_string_to_array(pointcloud_str, min_points=3):
    """
    Convert a pointcloud string with x,y coordinates to a numpy array.
    
    Args:
        pointcloud_str (str): String containing x,y coordinates, one pair per line
                             Format expected: "x1 y1\nx2 y2\n..."
        min_points (int): Minimum number of points required for a valid pointcloud
        
    Returns:
        np.ndarray: 2D array of shape (n,2) containing x,y coordinates,
                   or None if conversion failed
    """
    if not pointcloud_str or not isinstance(pointcloud_str, str):
        logging.error("Invalid pointcloud string provided")
        return None
    
    try:
        # Split the string into lines and process each line
        lines = pointcloud_str.strip().split('\n')
        points = []
        
        for line in lines:
            line = line.strip()
            # Skip empty lines or comment lines
            if not line or line.startswith('#') or line.startswith('!'):
                continue
            
            # Split the line into components (expecting x y format)
            components = line.split()
            if len(components) >= 2:  # Need at least x and y
                try:
                    x = float(components[0])
                    y = float(components[1])
                    points.append([x, y])
                except ValueError:
                    # Skip lines that can't be converted to floats
                    continue
        
        # Check if we have enough points
        if len(points) < min_points:
            logging.warning(f"Not enough valid points found: {len(points)}, minimum required: {min_points}")
            return None
        
        # Convert to numpy array
        return np.array(points)
        
    except Exception as e:
        logging.error(f"Error converting pointcloud string to array: {e}")
        return None
    
def pointcloud_string_to_array_optimized(pointcloud_str):
    """Optimized version of pointcloud string to numpy array conversion."""
    if not pointcloud_str:
        return np.array([])
    
    rows = [row for row in pointcloud_str.split('\n') if row.strip()]
    
    # Pre-allocate the array
    points = np.zeros((len(rows), 2))
    
    for i, row in enumerate(rows):
        points[i] = np.fromstring(row, sep=' ')
    
    return points


def pointcloud_array_to_string(points_array, precision=10):
    """
    Convert a numpy array of points to a formatted string.
    
    Args:
        points_array (np.ndarray): Array of shape (n,2) containing x,y coordinates
        precision (int): Number of decimal places to include in the output
        
    Returns:
        str: Formatted string with one x,y coordinate pair per line,
             or None if conversion failed
    """
    try:
        if points_array is None or not isinstance(points_array, np.ndarray):
            logging.error("Invalid points array provided")
            return None
        
        if points_array.size == 0 or points_array.shape[1] != 2:
            logging.error(f"Invalid points array shape: {points_array.shape}")
            return None
        
        # Format each point to a string with specified precision
        formatted_lines = []
        for point in points_array:
            x, y = point
            formatted_lines.append(f"{x:.{precision}f} {y:.{precision}f}")
        
        # Join lines with newline characters
        return '\n'.join(formatted_lines)
        
    except Exception as e:
        logging.error(f"Error converting points array to string: {e}")
        return None

# Example usage
if __name__ == "__main__":
    # Example pointcloud string
    example_str = """
    0.0 0.0
    0.25 0.1
    0.5 0.15
    0.75 0.1
    1.0 0.0
    0.75 -0.1
    0.5 -0.15
    0.25 -0.1
    0.0 0.0
    """
    
    # Convert string to array
    points_array = pointcloud_string_to_array(example_str)
    print("Converted to array:")
    print(points_array)
    
    # Convert array back to string
    if points_array is not None:
        points_str = pointcloud_array_to_string(points_array, precision=6)
        print("\nConverted back to string:")
        print(points_str)
