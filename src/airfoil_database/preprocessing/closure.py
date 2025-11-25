import numpy as np


def ensure_pointcloud_closed(pointcloud, tolerance=1e-6):
    """
    Ensures that a pointcloud is closed by checking if the first and last points are the same.
    If not, copies the first point to be the last point.

    Parameters:
    -----------
    pointcloud : numpy.ndarray or list
        Array of (x, y) coordinate pairs with shape (n, 2)
    tolerance : float, optional
        Tolerance for considering points as "the same" (default: 1e-6)

    Returns:
    --------
    numpy.ndarray
        Closed pointcloud array

    Example:
    --------
    >>> points = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 0.0], [0.5, -0.5]])
    >>> closed_points = ensure_pointcloud_closed(points)
    >>> print(closed_points)
    [[1.0, 0.0], [0.5, 0.5], [0.0, 0.0], [0.5, -0.5], [1.0, 0.0]]
    """

    # Convert to numpy array if it's not already
    if not isinstance(pointcloud, np.ndarray):
        pointcloud = np.array(pointcloud)

    # Check if we have at least 2 points
    if len(pointcloud) < 2:
        return pointcloud

    # Check if pointcloud has the right shape
    if pointcloud.shape[1] != 2:
        raise ValueError(f"Pointcloud must have shape (n, 2), got {pointcloud.shape}")

    # Get first and last points
    first_point = pointcloud[0]
    last_point = pointcloud[-1]

    # Calculate distance between first and last points
    distance = np.linalg.norm(first_point - last_point)

    # If points are not close enough, append the first point at the end
    if distance > tolerance:
        # Create new array with first point appended at the end
        closed_pointcloud = np.vstack([pointcloud, first_point.reshape(1, -1)])
        return closed_pointcloud
    else:
        # Already closed
        return pointcloud


# Alternative version that modifies the array in place (if you prefer)
def ensure_pointcloud_closed_inplace(pointcloud, tolerance=1e-6):
    """
    In-place version that modifies the original list/array.
    Note: This will convert lists to numpy arrays.
    """

    # Convert to numpy array if it's not already
    if not isinstance(pointcloud, np.ndarray):
        pointcloud = np.array(pointcloud)

    if len(pointcloud) < 2 or pointcloud.shape[1] != 2:
        return pointcloud

    # Check if first and last points are the same
    distance = np.linalg.norm(pointcloud[0] - pointcloud[-1])

    if distance > tolerance:
        # Append first point to the end
        pointcloud = np.vstack([pointcloud, pointcloud[0].reshape(1, -1)])

    return pointcloud


# Simple version for basic use cases
def close_pointcloud(points):
    """
    Simple version - just checks if first == last, if not, appends first to end.
    """
    if len(points) < 2:
        return points

    # Convert to numpy array for easy comparison
    points = np.array(points)

    # Check if first and last points are exactly the same
    if not np.array_equal(points[0], points[-1]):
        # Append first point to end
        points = np.vstack([points, points[0]])

    return points
