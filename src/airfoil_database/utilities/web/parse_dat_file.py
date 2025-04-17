import numpy as np

def parse_airfoil_dat(filepath):
    """
    Parses an airfoil .dat file supporting both classic and split formats.
    Returns a clockwise, non-redundant Nx2 NumPy array of (x, y) coordinates.
    """
    try:
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        # Remove header if the first line is not coordinates
        if not _is_point_line(lines[0]):
            lines = lines[1:]

        # Try full parse (Format 1)
        full = _parse_points(lines)
        if full is not None and _is_closed_loop(full):
            return full

        # Try split format (Format 2)
        split_idx = _detect_split_index(lines)
        if split_idx is None:
            return None

        upper_lines = lines[:split_idx]
        lower_lines = lines[split_idx:]

        upper = _parse_points(upper_lines)
        lower = _parse_points(lower_lines)

        if upper is None or lower is None:
            return None

        # Remove duplicate leading edge point if needed
        if np.allclose(upper[-1], lower[0]):
            lower = lower[1:]

        # Combine upper (TE → LE) + lower (LE → TE)
        points = np.vstack([upper, lower])

        return points

    except Exception:
        return None

def _is_point_line(line):
    try:
        float(line.split()[0])
        float(line.split()[1])
        return True
    except (ValueError, IndexError):
        return False

def _parse_points(lines):
    coords = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 2:
            x, y = float(parts[0]), float(parts[1])
            coords.append((x, y))
    return np.array(coords) if coords else None

def _detect_split_index(lines):
    # Check for empty line
    for i, line in enumerate(lines):
        if not line.strip():
            return i

    # Heuristic: Find x increasing after decreasing (classic LE)
    try:
        x_vals = [float(line.split()[0]) for line in lines if _is_point_line(line)]
        for i in range(1, len(x_vals)):
            if x_vals[i] > x_vals[i - 1]:  # switch from decreasing to increasing
                return i
    except Exception:
        pass
    return None

def _is_closed_loop(pts):
    return np.allclose(pts[0], pts[-1])
