# formulas/airfoil/compute_thickness_camber.py
"""
Robust thickness and camber calculation from airfoil x,y coordinates.
Fixes the zero-thickness bug by properly separating upper/lower surfaces.
"""

import numpy as np
from scipy.interpolate import interp1d


def compute_thickness_camber(points, n_samples=100):
    """
    Compute thickness and camber distributions from airfoil coordinates.

    Args:
        points: Nx2 array of (x, y) coordinates
        n_samples: Number of chord positions to sample

    Returns:
        x_coords: Chord positions (0 to 1)
        thickness: Thickness distribution
        camber: Camber distribution
    """

    if points.shape[0] < 4:
        raise ValueError("Need at least 4 points to define an airfoil")

    x = points[:, 0]
    y = points[:, 1]

    # Step 1: Normalize chord to 0-1
    x_min, x_max = x.min(), x.max()
    chord_length = x_max - x_min

    if chord_length < 1e-6:
        raise ValueError("Airfoil has zero chord length")

    x_norm = (x - x_min) / chord_length

    # Step 2: Find leading edge (minimum x) and split into upper/lower surfaces
    le_idx = np.argmin(x_norm)

    # Method 1: Split by leading edge index
    # Upper surface: from LE to trailing edge (typically counter-clockwise)
    # Lower surface: from trailing edge to LE (continuing around)

    # Check if data goes LE->TE on top, then TE->LE on bottom (typical format)
    if le_idx < len(x_norm) / 2:
        # Leading edge is near start
        upper_indices = np.arange(le_idx, len(x_norm))
        lower_indices = np.arange(0, le_idx + 1)[::-1]  # Reverse to go LE->TE
    else:
        # Leading edge is near end
        upper_indices = np.arange(0, le_idx + 1)
        lower_indices = np.arange(le_idx, len(x_norm))[::-1]

    x_upper = x_norm[upper_indices]
    y_upper = y[upper_indices]
    x_lower = x_norm[lower_indices]
    y_lower = y[lower_indices]

    # Alternative Method 2: Split by y-coordinate sign relative to mean
    # This is more robust for arbitrary orderings
    y_mean = np.mean(y)
    upper_mask = y >= y_mean
    lower_mask = y < y_mean

    # If Method 1 gives weird results, use Method 2
    if len(x_upper) < 3 or len(x_lower) < 3:
        x_upper = x_norm[upper_mask]
        y_upper = y[upper_mask]
        x_lower = x_norm[lower_mask]
        y_lower = y[lower_mask]

        # Sort by x
        upper_sort = np.argsort(x_upper)
        lower_sort = np.argsort(x_lower)
        x_upper = x_upper[upper_sort]
        y_upper = y_upper[upper_sort]
        x_lower = x_lower[lower_sort]
        y_lower = y_lower[lower_sort]

    # Step 3: Create interpolation functions for upper and lower surfaces
    # Remove duplicate x values (keep last occurrence)
    x_upper_unique, unique_indices = np.unique(x_upper, return_index=True)
    y_upper_unique = y_upper[unique_indices]

    x_lower_unique, unique_indices = np.unique(x_lower, return_index=True)
    y_lower_unique = y_lower[unique_indices]

    if len(x_upper_unique) < 2 or len(x_lower_unique) < 2:
        raise ValueError("Insufficient unique points on upper or lower surface")

    # Interpolate to common x positions
    x_common = np.linspace(0, 1, n_samples)

    try:
        # Linear interpolation is most robust
        f_upper = interp1d(
            x_upper_unique,
            y_upper_unique,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        f_lower = interp1d(
            x_lower_unique,
            y_lower_unique,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )

        y_upper_interp = f_upper(x_common)
        y_lower_interp = f_lower(x_common)

    except Exception as e:
        raise ValueError(f"Interpolation failed: {e}")

    # Step 4: Calculate thickness and camber
    # Thickness: vertical distance between upper and lower surfaces
    thickness = np.abs(y_upper_interp - y_lower_interp)

    # Camber: mean line (average of upper and lower)
    camber = (y_upper_interp + y_lower_interp) / 2.0

    # Validate results
    if np.all(thickness < 1e-10):
        raise ValueError(
            "Computed thickness is zero everywhere - check airfoil orientation"
        )

    if np.any(np.isnan(thickness)) or np.any(np.isnan(camber)):
        raise ValueError("NaN values in thickness or camber calculation")

    return x_common, thickness, camber


def compute_thickness_camber_simple(points):
    """
    Simplified version that returns only max values for quick checks.

    Args:
        points: Nx2 array of (x, y) coordinates

    Returns:
        max_thickness: Maximum thickness
        max_camber: Maximum camber (signed)
        max_t_position: Chord position of max thickness (0-1)
        max_c_position: Chord position of max camber (0-1)
    """

    try:
        x_coords, thickness, camber = compute_thickness_camber(points)

        max_thickness = np.max(thickness)
        max_camber_idx = np.argmax(np.abs(camber))
        max_camber = camber[max_camber_idx]

        max_t_position = x_coords[np.argmax(thickness)]
        max_c_position = x_coords[max_camber_idx]

        return max_thickness, max_camber, max_t_position, max_c_position

    except Exception as e:
        print(f"Error in thickness/camber calculation: {e}")
        return 0.0, 0.0, 0.0, 0.0


def validate_airfoil_coordinates(points, name="airfoil"):
    """
    Validate airfoil coordinate data and provide diagnostics.

    Args:
        points: Nx2 array of (x, y) coordinates
        name: Airfoil name for reporting

    Returns:
        bool: True if valid, False otherwise
        str: Diagnostic message
    """

    if points.shape[0] < 4:
        return False, f"{name}: Too few points ({points.shape[0]})"

    x = points[:, 0]
    y = points[:, 1]

    # Check for constant x (vertical line)
    if np.ptp(x) < 1e-6:
        return False, f"{name}: Zero chord length (x range: {np.ptp(x)})"

    # Check for NaN/Inf
    if np.any(~np.isfinite(points)):
        return False, f"{name}: Contains NaN or Inf values"

    # Check if upper and lower surfaces exist
    y_mean = np.mean(y)
    n_above = np.sum(y > y_mean)
    n_below = np.sum(y < y_mean)

    if n_above < 2 or n_below < 2:
        return (
            False,
            f"{name}: Missing upper or lower surface (above={n_above}, below={n_below})",
        )

    # Try to compute thickness
    try:
        x_coords, thickness, camber = compute_thickness_camber(points)
        max_t = np.max(thickness)

        if max_t < 1e-10:
            return False, f"{name}: Zero thickness computed"

        return True, f"{name}: Valid (max_t={max_t:.4f})"

    except Exception as e:
        return False, f"{name}: Computation failed - {str(e)}"


# Example usage and testing
if __name__ == "__main__":
    """Test the thickness/camber calculation with sample data."""

    print("Testing thickness/camber calculation...")
    print("=" * 60)

    # Test 1: Symmetric airfoil (NACA 0012 approximation)
    print("\nTest 1: Symmetric Airfoil")
    x_test = np.linspace(0, 1, 50)
    y_upper = 0.06 * (
        0.2969 * np.sqrt(x_test)
        - 0.126 * x_test
        - 0.3516 * x_test**2
        + 0.2843 * x_test**3
        - 0.1015 * x_test**4
    )
    y_lower = -y_upper

    # Combine into single airfoil (LE to TE on top, TE to LE on bottom)
    x_airfoil = np.concatenate([x_test, x_test[::-1]])
    y_airfoil = np.concatenate([y_upper, y_lower[::-1]])
    points = np.column_stack([x_airfoil, y_airfoil])

    valid, msg = validate_airfoil_coordinates(points, "NACA0012")
    print(f"Validation: {msg}")

    if valid:
        x_coords, thickness, camber = compute_thickness_camber(points)
        print(f"Max thickness: {np.max(thickness):.4f}")
        print(f"Max camber: {np.max(np.abs(camber)):.4f}")
        print(f"Thickness at 30% chord: {thickness[15]:.4f}")

    # Test 2: Cambered airfoil
    print("\nTest 2: Cambered Airfoil")
    camber_line = 0.02 * (2 * 0.4 * x_test - x_test**2) / 0.4**2
    y_upper_camber = camber_line + y_upper
    y_lower_camber = camber_line + y_lower

    x_airfoil2 = np.concatenate([x_test, x_test[::-1]])
    y_airfoil2 = np.concatenate([y_upper_camber, y_lower_camber[::-1]])
    points2 = np.column_stack([x_airfoil2, y_airfoil2])

    valid, msg = validate_airfoil_coordinates(points2, "NACA2412")
    print(f"Validation: {msg}")

    if valid:
        x_coords, thickness, camber = compute_thickness_camber(points2)
        print(f"Max thickness: {np.max(thickness):.4f}")
        print(f"Max camber: {np.max(camber):.4f}")
        print(f"Camber position: {x_coords[np.argmax(camber)]:.2f}")

    print("\n" + "=" * 60)
    print("Tests complete!")
