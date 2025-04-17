import numpy as np

def trailing_edge_angle(points, num_te_points=5):
    """
    Compute the trailing edge angle of the airfoil using a more robust approach.

    @param points: Numpy array of (x, y) coordinates.
    @param num_te_points: The number of points to use for trailing edge fitting.
    @return: Trailing edge angle in degrees.
    """
    if len(points) < 2: #Changed to 2. A line requires 2 points.
        raise ValueError("Not enough points to compute trailing edge angle.")

    # Sort points by x-coordinate in descending order
    sorted_points = points[np.argsort(points[:, 0])[::-1]]

    # Select the specified number of trailing edge points, or as many as available
    te_points = sorted_points[:min(num_te_points, len(sorted_points))]

    # Robust linear fit using RANSAC
    try:
        from sklearn.linear_model import RANSACRegressor
        ransac = RANSACRegressor()
        ransac.fit(te_points[:, 0].reshape(-1, 1), te_points[:, 1])
        angle = np.arctan(ransac.estimator_.coef_[0]) * (180 / np.pi)
    except ImportError:
        # Fallback to standard polyfit if scikit-learn is not available
        poly = np.polyfit(te_points[:, 0], te_points[:, 1], 1)
        angle = np.arctan(poly[0]) * (180 / np.pi)

    return angle