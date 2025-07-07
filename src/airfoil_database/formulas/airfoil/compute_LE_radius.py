import numpy as np
from scipy.spatial.distance import pdist
from scipy.linalg import lstsq

def fit_circle(points):
    """
    Fit a circle to the given points using a least-squares method.
    Returns (xc, yc, r) where (xc, yc) is the circle center and r is the radius.
    """
    x = points[:, 0]
    y = points[:, 1]
    
    # Improved matrix construction
    A = np.column_stack([2 * x, 2 * y, np.ones(len(x))])
    b = x**2 + y**2
    
    try:
        # Use scipy's lstsq for better numerical stability
        xc, yc, C = lstsq(A, b)[0]
        r = np.sqrt(C + xc**2 + yc**2)
        return xc, yc, r
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan

def is_collinear(points, tol=1e-3):
    """
    Check if the points are nearly collinear using SVD for better numerical stability.
    """
    if len(points) < 3:
        return True
        
    # Center the points
    centered = points - np.mean(points, axis=0)
    
    # Compute SVD
    _, s, _ = np.linalg.svd(centered)
    
    # If the smallest singular value is very small compared to the largest,
    # the points are nearly collinear
    return s[-1] / s[0] < tol

def remove_outliers(points, threshold=2.0):
    """
    Remove outliers based on the median absolute deviation (MAD).
    Returns a filtered set of points.
    """
    if len(points) < 4:  # Need at least 4 points to identify outliers
        return points
        
    # Calculate distances from each point to the median point
    median_point = np.median(points, axis=0)
    distances = np.sqrt(np.sum((points - median_point)**2, axis=1))
    
    # Calculate MAD
    mad = np.median(np.abs(distances - np.median(distances)))
    
    # Use modified z-score with MAD
    if mad > 1e-10:  # Avoid division by zero
        modified_z = 0.6745 * (distances - np.median(distances)) / mad
        filtered_points = points[np.abs(modified_z) < threshold]
        return filtered_points if len(filtered_points) >= 3 else points
    else:
        return points

def leading_edge_radius(points, num_le_points=8):
    """
    Compute the leading edge radius with improved robustness.
    """
    if len(points) < 3:
        return np.nan

    # Find leading edge index (minimum x-coordinate)
    le_idx = np.argmin(points[:, 0])
    le_x = points[le_idx, 0]
    le_y = points[le_idx, 1]

    # Select points around leading edge based on distance
    distances = np.sqrt((points[:, 0] - le_x)**2 + (points[:, 1] - le_y)**2)
    sorted_indices = np.argsort(distances)
    
    # Take more points initially, then filter
    initial_points = points[sorted_indices[:min(num_le_points*2, len(points))]]
    
    # Remove duplicate points
    le_points = np.unique(initial_points, axis=0)
    
    # Remove outliers
    le_points = remove_outliers(le_points)
    
    # Ensure we have enough points after filtering
    if len(le_points) < 3:
        # Fall back to using original points
        le_points = points[sorted_indices[:min(num_le_points, len(points))]]
    
    # Check if points are collinear
    if is_collinear(le_points):
        return np.inf  # Return large radius for nearly flat leading edges

    # Fit a circle to the selected points
    xc, yc, radius = fit_circle(le_points)

    # Check for unrealistic values
    if not np.isfinite(radius) or radius < 0 or radius > 0.5:  # Assuming normalized airfoil (0-1)
        # Try with different number of points
        for n in [6, 10, 4]:
            alt_points = points[sorted_indices[:min(n, len(points))]]
            if not is_collinear(alt_points):
                xc, yc, radius = fit_circle(alt_points)
                if np.isfinite(radius) and 0 < radius < 0.5:
                    return radius
        
        return np.nan

    return radius
