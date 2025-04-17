import numpy as np
import logging
from shapely.geometry import LineString, Point

def detect_airfoil_format(points, start_tol=1e-5):
    """Detects airfoil format and key indices."""
    le_idx = np.argmax(points[:, 0])
    te_idx = np.argmin(points[:, 0])
    
    # Do easiest check first, if the first and last points are the same, then nothing needs to be done
    if np.allclose(points[0], points[-1], atol=1e-4):
        return "closed", le_idx, te_idx
    
    # Complete check to see if the data is split
    tol = start_tol
    while tol < 1e-2:
        le_indices = np.where(np.abs(points[:, 0] - 1) < tol)[0]
        te_indices = np.where(np.abs(points[:, 0] - 0) < tol)[0]
        if len(le_indices) >= 2 and len(te_indices) >= 2:
            return "split", le_indices, te_indices
        elif len(le_indices) >= 2 or len(te_indices) >= 2:
            return "partial_split", le_indices, te_indices
        tol = tol*10
    
    return "unordered", le_idx, te_idx

def detect_direction(le_idx, te_idx):
    """Detects if the airfoil points are in LE to TE or TE to LE order."""
    if le_idx < te_idx:
        return "LE_to_TE"
    else:
        return "TE_to_LE"

def reorder_partial_split_airfoil(points, le_indices, te_indices):
    """Reorders a partial split airfoil."""

    if len(le_indices) >= 2:
        indices = le_indices
        is_le = True
    else:
        indices = te_indices
        is_le = False

    i1, i2 = sorted(indices[:2])
    if is_le:
        i1 += 1
        i2 += 1
    side1 = points[i1:i2]
    
    # Handle wrapping
    if i2 < i1:
        side2 = np.vstack([points[i2:], points[:i1]])
    else:
        side2 = np.vstack([points[i2:], points[:i1]]) if i2 == len(points)-1 else np.vstack([points[i2:], points[:i1]])

    # Determine direction of side1
    if side1[0][0] > side1[-1][0]:
        side1_dir = "LE_to_TE" if is_le else "TE_to_LE"
    else:
        side1_dir = "TE_to_LE" if is_le else "LE_to_TE"
    
    if side1_dir == "TE_to_LE":
        side1 = np.flip(side1, axis=0)

    # Determine direction of side2
    if side2[0][0] > side2[-1][0]:
        side2_dir = "LE_to_TE" if is_le else "TE_to_LE"
    else:
        side2_dir = "TE_to_LE" if is_le else "LE_to_TE"
    
    if side2_dir == "LE_to_TE":
        side2 = np.flip(side2[1:], axis=0)

    # Assemble the airfoil
    reordered_points = np.vstack([side1, side2])
    
    return reordered_points

def reorder_split_airfoil(points, le_indices, te_indices):
    """Reorders split airfoil data to LE -> SS -> TE -> PS -> LE."""
    le1_idx, le2_idx = sorted(le_indices[:2])
    te1_idx, te2_idx = sorted(te_indices[:2])

    if le1_idx < te1_idx:
        side1 = points[le1_idx:te1_idx+1]
    else:
        side1 = points[te1_idx:le1_idx+1]
        side1 = np.flip(side1, axis=0)
    
    if le2_idx < te2_idx:
        side2 = points[le2_idx:te2_idx+1]
    else:
        side2 = points[te2_idx:le2_idx+1]
        side2 = np.flip(side2, axis=0)
    
    if np.mean(side1, axis=0)[1] < np.mean(side2, axis=0)[1]:
        ps_points = np.flip(side1, axis=0)
        ss_points = side2
    else:
        ps_points = np.flip(side2, axis=0)
        ss_points = side1
    
    reordered_points = np.vstack([ss_points, ps_points])

    return reordered_points

def reorder_airfoil(points, le_idx, te_idx):
    """Reorders airfoil points to LE -> SS -> TE -> PS -> LE."""
    le_point = points[le_idx]
    te_point = points[te_idx]
    remaining_points = [p for i, p in enumerate(points) if i not in [le_idx, te_idx]]

    ss_points = []
    ps_points = []
    if remaining_points:
        mean_y = np.mean([p[1] for p in remaining_points])
        for p in remaining_points:
            if p[1] >= mean_y:
                ss_points.append(p)
            else:
                ps_points.append(p)

    ss_points.sort(key=lambda p: p[0], reverse=True)
    ps_points.sort(key=lambda p: p[0])

    direction = detect_direction(le_idx, te_idx)

    if direction == "LE_to_TE":
        reordered_points = [le_point] + ss_points + [te_point] + ps_points + [le_point]
    else:
        reordered_points = [le_point] + ps_points[::-1] + [te_point] + ss_points[::-1] + [le_point]

    return np.array(reordered_points)

def check_closure(points):
    """Ensures airfoil closure."""
    if not np.allclose(points[0], points[-1], atol=1e-6):
        points = np.vstack([points, [points[0]]])
    return points

def check_self_intersection(points, airfoil_name):
    """Checks for self-intersections."""
    line = LineString(points[:-1])
    if not line.is_simple:
        logging.warning(f"Self-intersection detected in {airfoil_name}")
        return True
    return False

def simplify_points(points, tolerance=1e-5):
    """Simplifies points by removing near-collinear points."""
    simplified_points = [points[0]]
    for i in range(1, len(points) - 1):
        v1 = points[i] - points[i - 1]
        v2 = points[i + 1] - points[i]
        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        if angle > tolerance:
            simplified_points.append(points[i])
    simplified_points.append(points[-1])
    return np.array(simplified_points)

def resolve_self_intersection(points):
    """Resolves self-intersections by reordering points."""
    line = LineString(points[:-1])
    if line.is_simple:
        return points

    intersections = []
    for i in range(len(points) - 2):
        segment1 = LineString([points[i], points[i + 1]])
        for j in range(i + 2, len(points) - 2):
            segment2 = LineString([points[j], points[j + 1]])
            if segment1.intersects(segment2):
                intersection = segment1.intersection(segment2)
                if isinstance(intersection, Point):
                    intersections.append((i, j))

    if not intersections:
        return points

    te_idx = np.argmax(points[:, 0])
    closest_intersection = None
    closest_distance = float('inf')

    for i, j in intersections:
        dist_to_te = min(abs(te_idx - i), abs(te_idx - j))
        if dist_to_te < closest_distance:
            closest_distance = dist_to_te
            closest_intersection = (i, j)

    if closest_intersection:
        i, j = closest_intersection
        points[i + 1:j + 1] = points[j:i:-1]

    line = LineString(points[:-1])
    if not line.is_simple:
        te_idx = np.argmax(points[:, 0])
        points[te_idx + 1:len(points)-1] = points[len(points)-2:te_idx:-1]

    return points

def remove_duplicate_points(points, tolerance=1e-8):
    """Removes duplicate points from the point cloud using a tolerance. Note it does count the last point"""
    if points is None or len(points) == 0:
        return np.array([])  # Handle empty or None input

    unique_points = [points[0]]
    for point in points[1:-1]:
        is_duplicate = False
        for unique_point in unique_points:
            if np.linalg.norm(point - unique_point) < tolerance:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_points.append(point)
    unique_points.append(points[-1])
    return np.array(unique_points)

def process_airfoil(airfoil_name, points):
    """Main function to process airfoil point cloud."""
    logging.info(f"Processing airfoil: {airfoil_name}, initial points: {len(points)}")

    format_type, le_info, te_info = detect_airfoil_format(points)

    if format_type == "split":
        logging.warning(f"Reordering split airfoil {airfoil_name}.")
        points = reorder_split_airfoil(points, le_info, te_info)
    elif format_type == "partial_split":
        logging.warning(f"Reordering partial split airfoil {airfoil_name}.")
        points = reorder_partial_split_airfoil(points, le_info, te_info)
    elif format_type in ["unordered"]:
        logging.warning(f"Reordering airfoil {airfoil_name} due to {format_type} format.")
        le_idx = np.argmax(points[:, 0])
        te_idx = np.argmin(points[:, 0])
        points = reorder_airfoil(points, le_idx, te_idx)

    points = check_closure(points)
    points = remove_duplicate_points(points)
    points = resolve_self_intersection(points)

    if check_self_intersection(points, airfoil_name):
        logging.warning(f"Self-intersection issues in {airfoil_name}, manual review recommended.")

    logging.info(f"Processed airfoil: {airfoil_name}, final points: {len(points)}")
    return points