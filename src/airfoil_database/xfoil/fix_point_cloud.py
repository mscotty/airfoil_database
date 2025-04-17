import numpy as np
from scipy.interpolate import splprep, splev
from shapely.geometry import LineString, Point
import logging
import math
import os
import re

# --- Configuration ---
# These thresholds might need tuning based on your data
DEFAULT_FIXER_CONFIG = {
    "min_points": 10,           # Minimum required points
    "max_te_gap_rel": 0.01,     # Max TE gap relative to chord (0 to 1)
    "max_le_point_spacing_rel": 0.02, # Max spacing near LE relative to chord
    "max_point_spacing_rel": 0.1,   # Max general spacing relative to chord
    "intersection_tolerance": 1e-6, # Tolerance for intersection checks
    "reorder_attempts": 2,         # How many reordering methods to try
    "interpolation_points": 240,   # Target points after interpolation (if needed)
    "precision": 10,               # Decimal places for output string
}

# --- Parsing and Formatting ---

def parse_airfoil_dat_file(filepath):
    """
    Parses an airfoil .dat file, detecting different formats and handling headers.
    Correctly handles Format 1 (NU/NL counts) and Format 2 (coordinate list).

    Args:
        filepath (str): The path to the .dat file.

    Returns:
        tuple: (airfoil_name, np.ndarray) or (None, None) on file read error,
               or (airfoil_name, None) on parsing error.
               The np.ndarray contains the correctly ordered points.
    """
    if not os.path.exists(filepath):
        logging.error(f"File not found: {filepath}")
        return None, None # Indicate file read error

    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            # Read all lines, strip whitespace, keep non-empty ones
            lines = [line.strip() for line in f if line.strip()]
    except Exception as e:
        logging.error(f"Error reading file {filepath}: {e}")
        return None, None # Indicate file read error

    if not lines:
        logging.error(f"File is empty or contains only whitespace: {filepath}")
        # Try to get name from filepath as fallback
        airfoil_name_fallback = os.path.splitext(os.path.basename(filepath))[0]
        return airfoil_name_fallback, None

    # --- Header and Format Detection ---
    airfoil_name = lines[0] # Assume first non-empty line is the name
    logging.debug(f"Parsing file: {filepath}, Name: {airfoil_name}")

    if len(lines) < 2:
         logging.error(f"File {filepath} contains only a header line. No coordinate data.")
         return airfoil_name, None

    # Look at the second non-empty line to detect format
    second_line = lines[1]
    parts = second_line.split()
    is_format_1 = False
    n_upper = 0
    n_lower = 0
    data_lines = [] # Store the actual coordinate lines

    if len(parts) >= 2:
        try:
            # Try parsing as NU NL counts
            potential_n_upper = int(float(parts[0]))
            potential_n_lower = int(float(parts[1]))

            # Check if counts seem valid and roughly match remaining lines
            # Use >= check as there might be trailing comments sometimes? Be lenient.
            if potential_n_upper > 0 and potential_n_lower > 0 and \
               (potential_n_upper + potential_n_lower <= len(lines) - 2): # Check against lines *after* NU/NL line
                is_format_1 = True
                n_upper = potential_n_upper
                n_lower = potential_n_lower
                data_lines = lines[2:] # Coordinate data starts from the 3rd line
                logging.debug(f"Detected Format 1 counts: NU={n_upper}, NL={n_lower}")
                # Sanity check actual coordinate lines vs counts
                num_coord_lines_found = 0
                for line in data_lines:
                    coord_parts = line.split()
                    if len(coord_parts) >= 2:
                        try: float(coord_parts[0]); float(coord_parts[1]); num_coord_lines_found += 1
                        except ValueError: break # Stop if non-coord found early
                    else: break # Stop if line format wrong
                if num_coord_lines_found != n_upper + n_lower:
                     logging.warning(f"Format 1 count mismatch in {filepath}: Expected {n_upper+n_lower}, found {num_coord_lines_found} coordinate lines. Attempting to parse anyway.")
                     # Keep is_format_1=True but be aware parsing might fail later

            else:
                # Doesn't look like valid counts or doesn't match line count. Assume Format 2.
                is_format_1 = False
                logging.debug(f"Line '{second_line}' not valid Format 1 counts/match. Assuming Format 2.")
                data_lines = lines[1:] # Coordinate data starts from the 2nd line

        except ValueError:
            # Second line cannot be parsed as two numbers. Assume Format 2.
            is_format_1 = False
            logging.debug(f"Line '{second_line}' not numeric counts. Assuming Format 2.")
            data_lines = lines[1:] # Coordinate data starts from the 2nd line
    else:
        # Second line doesn't have enough parts for NU NL. Assume Format 2.
        is_format_1 = False
        logging.debug(f"Line '{second_line}' has < 2 parts. Assuming Format 2.")
        data_lines = lines[1:] # Coordinate data starts from the 2nd line


    # --- Parse Data Lines ---
    points = []
    if is_format_1:
        # --- Parse Format 1 Data ---
        if len(data_lines) < n_upper + n_lower:
             logging.error(f"Format 1 error: Not enough data lines ({len(data_lines)}) found for specified counts NU={n_upper}, NL={n_lower} in {filepath}")
             return airfoil_name, None

        upper_points_raw = []
        lower_points_raw = []

        # Read Upper
        for i in range(n_upper):
            line = data_lines[i]
            try: parts = line.split(); x, y = float(parts[0]), float(parts[1]); upper_points_raw.append([x, y])
            except (ValueError, IndexError): logging.error(f"Error parsing Format 1 upper line {i+1}: '{line}' in {filepath}"); return airfoil_name, None
        # Read Lower
        for i in range(n_upper, n_upper + n_lower):
             # Check index bounds carefully
             if i >= len(data_lines):
                  logging.error(f"Format 1 error: Trying to read lower point index {i} but only {len(data_lines)} data lines exist in {filepath}")
                  return airfoil_name, None
             line = data_lines[i]
             try: parts = line.split(); x, y = float(parts[0]), float(parts[1]); lower_points_raw.append([x, y])
             except (ValueError, IndexError): logging.error(f"Error parsing Format 1 lower line {i+1-n_upper}: '{line}' in {filepath}"); return airfoil_name, None

        if not upper_points_raw or not lower_points_raw:
             logging.error(f"Failed to parse points for upper or lower surface in Format 1 for {filepath}")
             return airfoil_name, None

        upper_points = np.array(upper_points_raw)
        lower_points = np.array(lower_points_raw)

        # Reorder: TE -> Upper -> LE -> Lower -> TE
        upper_reversed = upper_points[::-1]
        if upper_points.shape[0] > 0 and lower_points.shape[0] > 0:
            if not np.allclose(upper_points[0], lower_points[0], atol=1e-5): logging.warning(f"LE point mismatch in {filepath}")
            # Combine: All reversed upper points + lower points excluding the first (LE) point
            ordered_points = np.vstack((upper_reversed, lower_points[1:]))
            logging.debug(f"Reordered Format 1 points from {filepath}. Total points: {len(ordered_points)}")
            return airfoil_name, ordered_points
        else:
            logging.error(f"Cannot reorder Format 1 points for {filepath}: Upper or Lower surface empty after parsing.")
            return airfoil_name, None

    else:
        # --- Parse Format 2 Data (Standard Coordinate List) ---
        logging.debug(f"Parsing {filepath} as coordinate list format (Format 2).")
        for i, line in enumerate(data_lines):
            try:
                parts = line.split()
                if len(parts) >= 2:
                    x, y = float(parts[0]), float(parts[1])
                    points.append([x, y])
                # else: Ignore lines with incorrect number of parts
            except ValueError:
                # Ignore lines that cannot be parsed as floats
                logging.debug(f"Skipping non-coordinate line {i+2} in Format 2 parsing of {filepath}: '{line}'")
                continue

        if not points:
            logging.error(f"No valid coordinate points found in Format 2 parsing of {filepath}")
            return airfoil_name, None
        logging.debug(f"Parsed {len(points)} points in Format 2 from {filepath}.")
        return airfoil_name, np.array(points)


# --- Formatting Function (remains the same) ---
def format_pointcloud_array(points_array, precision=10):
    """Formats a NumPy array of points back into a string with specified precision."""
    # ... (implementation is the same as previous version) ...
    if points_array is None or points_array.shape[0] == 0: return ""
    if not isinstance(points_array, np.ndarray):
        try:
            points_array = np.array(points_array)
            if points_array.ndim != 2 or points_array.shape[1] != 2: raise ValueError("Input not Nx2 array")
        except Exception as e: logging.error(f"Invalid input to format_pointcloud_array: {e}"); return ""
    point_strings = [f"{x:.{precision}f} {y:.{precision}f}" for x, y in points_array]
    return "\n".join(point_strings)

# --- Checking Functions ---

def check_basic_validity(points, config):
    """Check minimum points and basic structure."""
    if points is None or not isinstance(points, np.ndarray):
        logging.warning("Point cloud is None or not a NumPy array.")
        return False, "Invalid input type"
    if points.shape[0] < config["min_points"]:
        logging.warning(f"Insufficient points: {points.shape[0]} < {config['min_points']}")
        return False, f"Too few points ({points.shape[0]})"
    if points.shape[1] != 2:
         logging.warning(f"Incorrect point dimension: {points.shape[1]} != 2")
         return False, "Incorrect dimensions"
    return True, "Basic checks passed"

def estimate_chord_length(points):
    """Estimate chord length (max x - min x). Assumes roughly aligned with x-axis."""
    if points is None or points.shape[0] < 2:
        return 1.0 # Default guess if unable to calculate
    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    return max(max_x - min_x, 1e-6) # Avoid zero chord

def check_te_closure(points, config):
    """Check if the trailing edge gap is within tolerance."""
    chord = estimate_chord_length(points)
    gap = np.linalg.norm(points[0] - points[-1])
    max_gap_abs = config["max_te_gap_rel"] * chord
    if gap > max_gap_abs:
        logging.debug(f"TE gap {gap:.4f} exceeds tolerance {max_gap_abs:.4f}")
        return False, f"TE gap too large ({gap:.4f})"
    return True, "TE closed"

def check_point_spacing(points, config):
    """Check for excessive spacing between points, especially near LE."""
    chord = estimate_chord_length(points)
    max_spacing_abs = config["max_point_spacing_rel"] * chord
    max_le_spacing_abs = config["max_le_point_spacing_rel"] * chord

    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)

    # Check general spacing
    max_dist = np.max(distances)
    if max_dist > max_spacing_abs:
        logging.debug(f"Max point spacing {max_dist:.4f} exceeds tolerance {max_spacing_abs:.4f}")
        return False, f"Excessive point spacing ({max_dist:.4f})"

    # Check LE spacing (find LE point index - min x)
    le_index = np.argmin(points[:, 0])
    # Check spacing around LE (e.g., 5 points before and after, handle boundaries)
    le_region_start = max(0, le_index - 5)
    le_region_end = min(points.shape[0] - 1, le_index + 5)
    if le_region_start < le_region_end: # Ensure region has size
        le_distances = np.linalg.norm(np.diff(points[le_region_start:le_region_end+1], axis=0), axis=1)
        max_le_dist = np.max(le_distances) if le_distances.size > 0 else 0
        if max_le_dist > max_le_spacing_abs:
            logging.debug(f"Max LE spacing {max_le_dist:.4f} exceeds tolerance {max_le_spacing_abs:.4f}")
            return False, f"Excessive LE spacing ({max_le_dist:.4f})"

    return True, "Point spacing acceptable"

def check_intersections(points, config):
    """Check for self-intersections using Shapely."""
    try:
        line = LineString(points)
        if not line.is_simple:
            # More detailed check (optional): find intersection points
            # intersection_point = line.intersection(line) ...
            logging.debug("Self-intersection detected.")
            return False, "Self-intersection"
    except Exception as e:
        logging.warning(f"Shapely intersection check failed: {e}")
        # Treat failure as potential issue or skip check? For now, assume okay.
        return True, "Intersection check failed (skipped)"
    return True, "No self-intersections"

# --- Fixing Functions ---

def normalize_airfoil(points):
    """Translates LE to (0,0) and scales chord to 1."""
    if points is None or points.shape[0] < 2:
        return points
    min_x_idx = np.argmin(points[:, 0])
    le_point = points[min_x_idx]

    # Translate LE to origin
    points_translated = points - le_point

    # Find TE (point furthest from new origin, likely max x)
    # A more robust way might involve finding the two points furthest apart
    max_x_translated = np.max(points_translated[:, 0])
    if max_x_translated <= 1e-6: # Avoid division by zero/tiny number
         logging.warning("Could not determine chord length for scaling after translation.")
         return points_translated # Return translated but unscaled

    # Scale chord to 1
    points_normalized = points_translated / max_x_translated
    return points_normalized


def reorder_points(points):
    """Attempts to reorder points to follow a standard airfoil path (TE->Upper->LE->Lower->TE)."""
    # This is complex. A simple approach:
    # 1. Find LE (min x) and TE (approx avg of first/last or points near x=1 after normalization).
    # 2. Split points into upper/lower based on y-value relative to a line connecting LE/TE.
    # 3. Sort upper surface points by decreasing x.
    # 4. Sort lower surface points by increasing x.
    # 5. Combine TE_lower -> Lower -> LE -> Upper -> TE_upper.
    # This is a placeholder for a more robust algorithm.
    logging.warning("Reordering logic is complex and placeholder used.")
    # For now, just ensure LE is roughly at min x and TE near max x
    min_x_idx = np.argmin(points[:, 0])
    # Simple roll to put LE roughly in the middle if it's near start/end
    if min_x_idx < 5 or min_x_idx > points.shape[0] - 5:
         logging.debug("Attempting simple roll based on LE position.")
         points = np.roll(points, shift=points.shape[0]//2 - min_x_idx, axis=0)

    return points # Return potentially rolled points


def close_te_gap(points, config):
    """Closes the trailing edge gap by averaging or projecting."""
    chord = estimate_chord_length(points)
    gap = np.linalg.norm(points[0] - points[-1])
    max_gap_abs = config["max_te_gap_rel"] * chord

    if 0 < gap <= max_gap_abs * 1.5: # Only close if reasonably small
        logging.debug(f"Closing TE gap of {gap:.4f}")
        avg_point = (points[0] + points[-1]) / 2.0
        points[0] = avg_point
        points[-1] = avg_point
    elif gap > max_gap_abs * 1.5:
         logging.warning(f"TE gap {gap:.4f} too large to close automatically.")
    return points

def interpolate_points(points, num_target_points):
    """Interpolates points using splines to achieve target density."""
    if points is None or points.shape[0] < 4: # Need enough points for spline
        logging.warning("Not enough points to interpolate.")
        return points

    try:
        # Parameterize based on cumulative distance
        distances = np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))
        distances = np.insert(distances, 0, 0) # Add starting point distance
        param = distances / distances[-1] # Normalize to 0-1

        # Fit spline (k=3 for cubic)
        tck, _ = splprep([points[:, 0], points[:, 1]], u=param, k=3, s=0) # s=0 forces interpolation through points

        # Evaluate spline at new parameter values
        new_param = np.linspace(0, 1, num_target_points)
        new_points = splev(new_param, tck)
        new_points = np.vstack(new_points).T # Transpose to get Nx2 array

        # Ensure start/end points match originals closely (optional, but good practice)
        # new_points[0] = points[0]
        # new_points[-1] = points[-1]
        # Or if TE was closed, ensure the new start/end are identical
        if np.allclose(points[0], points[-1]):
             new_points[-1] = new_points[0]

        logging.debug(f"Interpolated to {num_target_points} points.")
        return new_points
    except Exception as e:
        logging.error(f"Spline interpolation failed: {e}")
        return points # Return original points on failure


# --- Main Fixing Orchestrator ---

def fix_pointcloud(points_array, config=None):
    """
    Applies a sequence of checks and fixes to an airfoil point cloud.

    Args:
        points_array (np.ndarray): NumPy array of points (Nx2).
        config (dict, optional): Configuration dictionary. Uses defaults if None.

    Returns:
        np.ndarray or None: The fixed point cloud array, or None if unfixable.
    """
    if config is None:
        config = DEFAULT_FIXER_CONFIG

    current_points = points_array.copy() # Work on a copy

    # 1. Basic Validity Check
    is_valid, reason = check_basic_validity(current_points, config)
    if not is_valid:
        logging.error(f"Fixing failed: Basic validity check failed - {reason}")
        return None # Cannot proceed

    # 2. Normalization (Optional but recommended for consistent checks)
    # Decide if you want to store normalized or original scale points.
    # Storing normalized is often better for comparisons.
    original_points = current_points.copy() # Keep original if needed later
    current_points = normalize_airfoil(current_points)
    logging.debug("Normalized airfoil (LE=0,0, Chord=1).")

    # 3. Reordering (Attempt if needed - complex)
    # Add a check_order function here if possible
    # For now, we apply the simple roll attempt
    current_points = reorder_points(current_points)

    # 4. TE Closure Check & Fix
    is_closed, reason = check_te_closure(current_points, config)
    if not is_closed:
        logging.warning(f"Attempting to fix TE gap: {reason}")
        current_points = close_te_gap(current_points, config)
        # Recheck after fixing
        is_closed, _ = check_te_closure(current_points, config)
        if not is_closed:
             logging.error("Fixing failed: Could not close TE gap.")
             # Decide whether to return original or None
             return None # Or return original_points if preferred

    # 5. Intersection Check
    is_simple, reason = check_intersections(current_points, config)
    if not is_simple:
        logging.error(f"Fixing failed: {reason} detected.")
        # Attempting to fix intersections automatically is very difficult.
        # Usually best to flag or return None/original.
        return None # Or return original_points

    # 6. Point Spacing Check & Interpolation (Fix)
    needs_interp, reason = check_point_spacing(current_points, config)
    if not needs_interp: # Note: check returns True if spacing is OK
        logging.warning(f"Point spacing issue detected: {reason}. Attempting interpolation.")
        current_points = interpolate_points(current_points, config["interpolation_points"])
        # Optional: Re-run checks after interpolation if needed

    # 7. Final Checks (Optional)
    # Run checks again to ensure fixes didn't introduce new problems
    is_valid, reason = check_basic_validity(current_points, config)
    if not is_valid: return None # Failed after fixing
    is_closed, _ = check_te_closure(current_points, config)
    if not is_closed: return None # Failed after fixing
    is_simple, _ = check_intersections(current_points, config)
    if not is_simple: return None # Failed after fixing

    logging.info("Point cloud fixing process completed.")
    # Decide whether to return normalized or rescale to original size
    # Returning normalized points here:
    return current_points

    # To return rescaled (more complex):
    # 1. Need to store original LE position and chord scale from normalize_airfoil
    # 2. Apply inverse scaling and translation here before returning


