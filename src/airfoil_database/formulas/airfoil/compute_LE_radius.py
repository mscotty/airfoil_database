import numpy as np
from scipy.spatial.distance import pdist

def fit_circle(points):
    """
    Fit a circle to the given points using a least-squares method.
    Returns (xc, yc, r) where (xc, yc) is the circle center and r is the radius.
    """
    x = points[:, 0]
    y = points[:, 1]
    
    A = np.c_[2 * x, 2 * y, np.ones(len(x))]
    b = x ** 2 + y ** 2
    
    try:
        xc, yc, C = np.linalg.lstsq(A, b, rcond=None)[0]
        r = np.sqrt(C + xc**2 + yc**2)
        return xc, yc, r
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan

def is_collinear(points, tol=1e-3):
    """Check if the points are nearly collinear using pairwise distances."""
    dists = pdist(points)  # Compute pairwise distances
    min_dist, max_dist = np.min(dists), np.max(dists)
    return (max_dist - min_dist) < tol

def remove_outliers(points, threshold=1.5):
    """
    Remove outliers based on the median absolute deviation (MAD).
    Returns a filtered set of points.
    """
    y_vals = points[:, 1]
    median = np.median(y_vals)
    mad = np.median(np.abs(y_vals - median))
    
    if mad == 0:
        return points  # No filtering needed
    
    filtered_points = points[np.abs(y_vals - median) / mad < threshold]
    return filtered_points if len(filtered_points) >= 3 else points  # Ensure at least 3 points

def leading_edge_radius(points, num_le_points=6):
    """
    Compute the leading edge radius with improved robustness.
    """
    if len(points) < 3:
        return np.nan

    # Find leading edge index (minimum x-coordinate)
    le_idx = np.argmin(points[:, 0])
    le_x = points[le_idx, 0]

    # Select points symmetrically around leading edge
    distances = np.abs(points[:, 0] - le_x)
    sorted_indices = np.argsort(distances)
    le_points = points[sorted_indices[:min(num_le_points, len(points))]]

    # Remove duplicate points
    le_points = np.unique(le_points, axis=0)

    # Remove outliers
    le_points = remove_outliers(le_points)

    # Check if points are collinear
    if is_collinear(le_points):
        return np.inf  # Return large radius for nearly flat leading edges

    # Fit a circle to the selected points
    xc, yc, radius = fit_circle(le_points)

    # Check for unrealistic values
    if radius < 0 or radius > 10_000:
        return np.nan

    return radius
