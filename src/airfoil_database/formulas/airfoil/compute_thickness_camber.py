import numpy as np

# Compute Thickness and Camber
def compute_thickness_camber(points):
    """
    Compute airfoil thickness and camber distributions.
    
    @param points: Numpy array of (x, y) coordinates.
    @return: x coordinates, thickness, and camber distributions.
    """
    x_coords = np.unique(points[:, 0])
    upper_surface = []
    lower_surface = []
    
    for x in x_coords:
        y_vals = points[points[:, 0] == x][:, 1]
        upper_surface.append(max(y_vals))
        lower_surface.append(min(y_vals))
    
    thickness = np.array(upper_surface) - np.array(lower_surface)
    camber = (np.array(upper_surface) + np.array(lower_surface)) / 2
    
    return x_coords, thickness, camber