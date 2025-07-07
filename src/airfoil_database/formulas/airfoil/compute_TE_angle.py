import numpy as np
from sklearn.linear_model import RANSACRegressor, HuberRegressor

def trailing_edge_angle(points, num_te_points=5):
    """
    Compute the trailing edge angle of the airfoil using a more robust approach.

    @param points: Numpy array of (x, y) coordinates.
    @param num_te_points: The number of points to use for trailing edge fitting.
    @return: Trailing edge angle in degrees.
    """
    if len(points) < 2:
        return np.nan

    # Find trailing edge (maximum x-coordinate)
    te_idx = np.argmax(points[:, 0])
    te_x = points[te_idx, 0]
    
    # Get upper and lower surface points near trailing edge
    upper_points = []
    lower_points = []
    
    # Sort by distance from trailing edge along x-axis
    distances = np.abs(points[:, 0] - te_x)
    sorted_indices = np.argsort(distances)
    
    # Classify points as upper or lower surface
    for idx in sorted_indices[:min(num_te_points*3, len(points))]:
        if points[idx, 1] >= points[te_idx, 1]:
            upper_points.append(points[idx])
        else:
            lower_points.append(points[idx])
    
    upper_points = np.array(upper_points)
    lower_points = np.array(lower_points)
    
    # Ensure we have enough points
    if len(upper_points) < 2 or len(lower_points) < 2:
        # Fall back to simple approach
        sorted_points = points[np.argsort(points[:, 0])[::-1]]
        te_points = sorted_points[:min(num_te_points, len(sorted_points))]
        
        try:
            # Use robust regression
            huber = HuberRegressor(epsilon=1.35)
            huber.fit(te_points[:, 0].reshape(-1, 1), te_points[:, 1])
            angle = np.arctan(huber.coef_[0]) * (180 / np.pi)
            return angle
        except:
            # Fallback to standard polyfit
            poly = np.polyfit(te_points[:, 0], te_points[:, 1], 1)
            angle = np.arctan(poly[0]) * (180 / np.pi)
            return angle
    
    # Fit lines to upper and lower surfaces
    try:
        # Use robust regression
        upper_model = HuberRegressor(epsilon=1.35)
        upper_model.fit(upper_points[:, 0].reshape(-1, 1), upper_points[:, 1])
        upper_angle = np.arctan(upper_model.coef_[0]) * (180 / np.pi)
        
        lower_model = HuberRegressor(epsilon=1.35)
        lower_model.fit(lower_points[:, 0].reshape(-1, 1), lower_points[:, 1])
        lower_angle = np.arctan(lower_model.coef_[0]) * (180 / np.pi)
        
        # Trailing edge angle is the difference between upper and lower angles
        te_angle = upper_angle - lower_angle
        
        # Ensure positive angle
        return abs(te_angle)
    except:
        # Fallback to standard polyfit
        try:
            upper_poly = np.polyfit(upper_points[:, 0], upper_points[:, 1], 1)
            lower_poly = np.polyfit(lower_points[:, 0], lower_points[:, 1], 1)
            
            upper_angle = np.arctan(upper_poly[0]) * (180 / np.pi)
            lower_angle = np.arctan(lower_poly[0]) * (180 / np.pi)
            
            te_angle = upper_angle - lower_angle
            return abs(te_angle)
        except:
            return np.nan
