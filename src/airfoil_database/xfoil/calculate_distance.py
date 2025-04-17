import numpy as np

def calculate_pointcloud_distance(pc1, pc2):
    """Calculates the average Euclidean distance between two point clouds."""
    if len(pc1) != len(pc2):
        return float('inf')  # Return infinity for point clouds of different sizes

    distances = np.linalg.norm(pc1 - pc2, axis=1)
    return np.mean(distances)

import numpy as np

import numpy as np

def calculate_min_distance_sum(pc1, pc2):
    """Calculates the sum of minimum distances from each point in pc1 to pc2."""
    if len(pc2) == 0:
        return float('inf')  # Handle empty pc2 (even if pc1 is also empty)

    total_min_distance = 0
    for point1 in pc1:
        distances = np.linalg.norm(pc2 - point1, axis=1)
        total_min_distance += np.min(distances)
    return total_min_distance