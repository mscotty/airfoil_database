# utils/helpers.py
import numpy as np
import logging
from typing import List, Optional, Union, Dict, Any

def _pointcloud_to_numpy(pointcloud_str):
    """Converts a pointcloud string to a NumPy array."""
    if not pointcloud_str:
        return np.array([])
    rows = pointcloud_str.strip().split('\n')
    rows = [x.strip() for x in rows if x.strip()]
    try:
        return np.array([np.fromstring(row, dtype=float, sep=' ') for row in rows])
    except ValueError:
        return np.array([])

def parse_pointcloud_string(pointcloud_str):
    """Parse a pointcloud string into a numpy array."""
    if not pointcloud_str or not isinstance(pointcloud_str, str):
        return None
        
    try:
        rows = pointcloud_str.strip().split('\n')
        rows = [x.strip() for x in rows if x.strip()]
        return np.array([np.fromstring(row, dtype=float, sep=' ') for row in rows])
    except Exception as e:
        logging.error(f"Error parsing pointcloud string: {e}")
        return None

def format_pointcloud_array(points_array, precision=10):
    """Format a numpy array of points back to a string with specified precision."""
    if points_array is None or len(points_array) == 0:
        return ""
        
    format_str = f"{{:.{precision}f}} {{:.{precision}f}}"
    return "\n".join(format_str.format(x, y) for x, y in points_array)

# Default configuration for the pointcloud fixer
DEFAULT_FIXER_CONFIG = {
    "precision": 10,
    "normalize": True,
    "remove_duplicates": True,
    "sort_points": True,
    "interpolate_points": False,
    "num_points": 100
}

def fix_pointcloud(points_array, config=None):
    """Fix issues with a pointcloud array."""
    if points_array is None or len(points_array) == 0:
        return None
        
    # Use default config if none provided
    if config is None:
        config = DEFAULT_FIXER_CONFIG
        
    try:
        # Apply fixes based on configuration
        result = points_array.copy()
        
        # Remove duplicates if configured
        if config.get("remove_duplicates", True):
            result = np.unique(result, axis=0)
            
        # Normalize if configured
        if config.get("normalize", True):
            from airfoil_database.xfoil.fix_airfoil_data import normalize_pointcloud
            result = normalize_pointcloud(result)
            
        # Sort points if configured
        if config.get("sort_points", True):
            # Sort by x-coordinate
            result = result[result[:, 0].argsort()]
            
        # Interpolate if configured
        if config.get("interpolate_points", False):
            from airfoil_database.xfoil.interpolate_points import interpolate_points
            num_points = config.get("num_points", 100)
            result = interpolate_points(result, num_points=num_points)
            
        return result
    except Exception as e:
        logging.error(f"Error fixing pointcloud: {e}")
        return None
