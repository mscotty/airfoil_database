# core/data_preparation.py
import numpy as np
import json
import logging
from typing import List, Tuple, Optional
from sqlmodel import Session, select
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d
from .database import AirfoilDatabase
from .models import Airfoil


class AirfoilDataPreparator:
    def __init__(self, database: AirfoilDatabase, target_points: int = 1000):
        self.db = database
        self.target_points = target_points
        self.original_points = None  # Store for validation

    def parse_pointcloud_from_string(self, pointcloud_str: str) -> np.ndarray:
        """Parse point cloud string from database into numpy array."""
        if not pointcloud_str:
            return np.array([])

        lines = pointcloud_str.strip().split("\n")
        points = []

        for line in lines:
            line = line.strip()
            if line and not line.startswith("#"):  # Skip comments
                try:
                    coords = line.split()
                    if len(coords) >= 2:
                        x, y = float(coords[0]), float(coords[1])
                        points.append([x, y])
                except ValueError:
                    continue

        return np.array(points)

    def normalize_airfoil_coordinates(self, points: np.ndarray) -> np.ndarray:
        """Normalize airfoil to unit chord and standard position."""
        if len(points) == 0:
            return points

        # Ensure points are in correct order (leading edge at x=0, trailing edge at x=1)
        x_coords = points[:, 0]

        # Normalize to unit chord
        x_min, x_max = x_coords.min(), x_coords.max()
        chord_length = x_max - x_min

        if chord_length == 0:
            return points

        normalized_points = points.copy()
        normalized_points[:, 0] = (points[:, 0] - x_min) / chord_length
        normalized_points[:, 1] = points[:, 1] / chord_length

        return normalized_points

    def detect_point_ordering(self, points: np.ndarray) -> str:
        """
        Detect point ordering using aerodynamic surface identification.
        Uses the principle that suction surface (upper) has higher y-values
        and pressure surface (lower) has lower y-values between LE and TE.
        """
        if len(points) < 4:
            return "unknown"

        # Find LE and TE using robust detection
        te_idx, le_idx = self._find_le_te_robust(points)

        if te_idx == le_idx:
            return "unknown"

        # Split points into two paths between LE and TE
        path1_indices, path2_indices = self._get_le_te_paths(points, le_idx, te_idx)

        if len(path1_indices) < 2 or len(path2_indices) < 2:
            return "unknown"

        # Extract the two surface paths
        path1_points = points[path1_indices]
        path2_points = points[path2_indices]

        # Calculate average y-coordinates for each path
        path1_avg_y = np.mean(path1_points[:, 1])
        path2_avg_y = np.mean(path2_points[:, 1])

        # Identify which path is upper (suction) vs lower (pressure) surface
        if path1_avg_y > path2_avg_y:
            upper_surface_indices = path1_indices
            lower_surface_indices = path2_indices
        else:
            upper_surface_indices = path2_indices
            lower_surface_indices = path1_indices

        # Determine ordering based on surface sequence
        ordering = self._determine_ordering_from_surfaces(
            points, upper_surface_indices, lower_surface_indices, te_idx, le_idx
        )

        return ordering

    def _find_le_te_robust(self, points: np.ndarray) -> Tuple[int, int]:
        """Robust LE/TE detection for airfoil ordering analysis."""

        # Primary method: x-coordinate extremes
        te_idx = np.argmax(points[:, 0])  # Rightmost point
        le_idx = np.argmin(points[:, 0])  # Leftmost point

        # Validation: ensure they make aerodynamic sense
        if points[te_idx, 0] <= points[le_idx, 0]:
            # Fallback: find points with maximum separation
            max_distance = 0
            te_candidate, le_candidate = 0, 0

            for i in range(len(points)):
                for j in range(i + 1, len(points)):
                    dist = np.sqrt(np.sum((points[i] - points[j]) ** 2))
                    if dist > max_distance:
                        max_distance = dist
                        # Assign based on x-coordinate
                        if points[i, 0] > points[j, 0]:
                            te_candidate, le_candidate = i, j
                        else:
                            te_candidate, le_candidate = j, i

            te_idx, le_idx = te_candidate, le_candidate

        return te_idx, le_idx

    def _get_le_te_paths(
        self, points: np.ndarray, le_idx: int, te_idx: int
    ) -> Tuple[List[int], List[int]]:
        """Get the two possible paths between LE and TE."""

        n_points = len(points)

        # Handle different cases based on LE/TE positions
        if te_idx == 0 or te_idx == n_points - 1:
            # TE at start or end
            if te_idx == 0:
                # Path 1: TE to LE (forward)
                path1 = list(range(te_idx, le_idx + 1))
                # Path 2: LE to TE (wrap around)
                path2 = list(range(le_idx, n_points))
            else:  # te_idx == n_points - 1
                # Path 1: start to LE
                path1 = list(range(0, le_idx + 1))
                # Path 2: LE to TE
                path2 = list(range(le_idx, te_idx + 1))
        else:
            # TE somewhere in middle
            if te_idx < le_idx:
                # Path 1: TE to LE
                path1 = list(range(te_idx, le_idx + 1))
                # Path 2: LE around to TE
                path2 = list(range(le_idx, n_points)) + list(range(0, te_idx + 1))
            else:
                # Path 1: TE around to LE
                path1 = list(range(te_idx, n_points)) + list(range(0, le_idx + 1))
                # Path 2: LE to TE
                path2 = list(range(le_idx, te_idx + 1))

        return path1, path2

    def _determine_ordering_from_surfaces(
        self,
        points: np.ndarray,
        upper_indices: List[int],
        lower_indices: List[int],
        te_idx: int,
        le_idx: int,
    ) -> str:
        """
        Determine if points are ordered clockwise or counterclockwise based on
        identified upper and lower surfaces.
        """

        # Check which surface comes first in the point sequence
        first_upper_idx = min(upper_indices) if upper_indices else float("inf")
        first_lower_idx = min(lower_indices) if lower_indices else float("inf")

        # For a counterclockwise airfoil starting from TE:
        # Should go TE -> Upper Surface -> LE -> Lower Surface -> TE

        # Check if upper surface comes before lower surface in sequence
        if te_idx in upper_indices:
            # TE is on upper surface path
            if first_upper_idx < first_lower_idx:
                return "counterclockwise"
            else:
                return "clockwise"
        elif te_idx in lower_indices:
            # TE is on lower surface path
            if first_lower_idx < first_upper_idx:
                return "clockwise"
            else:
                return "counterclockwise"
        else:
            # More complex analysis needed
            return self._analyze_surface_progression(
                points, upper_indices, lower_indices, te_idx, le_idx
            )

    def _analyze_surface_progression(
        self,
        points: np.ndarray,
        upper_indices: List[int],
        lower_indices: List[int],
        te_idx: int,
        le_idx: int,
    ) -> str:
        """Analyze the progression of surfaces to determine ordering."""

        # Find the sequence of surface transitions
        surface_sequence = []

        for i in range(len(points)):
            if i in upper_indices:
                surface_sequence.append("upper")
            elif i in lower_indices:
                surface_sequence.append("lower")
            else:
                surface_sequence.append("unknown")

        # Look for the pattern: should be continuous upper, then continuous lower
        # for counterclockwise starting from TE

        # Find transitions
        transitions = []
        current_surface = surface_sequence[0]

        for i, surface in enumerate(surface_sequence[1:], 1):
            if surface != current_surface and surface != "unknown":
                transitions.append((i, current_surface, surface))
                current_surface = surface

        # Analyze transition pattern
        if len(transitions) >= 1:
            first_transition = transitions[0]
            if first_transition[1] == "upper" and first_transition[2] == "lower":
                return "counterclockwise"
            elif first_transition[1] == "lower" and first_transition[2] == "upper":
                return "clockwise"

        # Fallback to geometric method if surface analysis is inconclusive
        return self._geometric_ordering_fallback(points)

    def _geometric_ordering_fallback(self, points: np.ndarray) -> str:
        """Fallback geometric ordering detection."""
        if len(points) < 3:
            return "unknown"

        # Use shoelace formula as fallback
        signed_area = 0
        n = len(points)

        for i in range(n):
            j = (i + 1) % n
            signed_area += (points[j, 0] - points[i, 0]) * (points[j, 1] + points[i, 1])

        if abs(signed_area) < 1e-10:
            return "unknown"

        return "clockwise" if signed_area > 0 else "counterclockwise"

    def standardize_point_ordering(self, points: np.ndarray) -> np.ndarray:
        """Ensure consistent counterclockwise ordering starting from trailing edge."""
        if len(points) == 0:
            return points

        # Remove last point if it's a duplicate of first (existing closure)
        working_points = (
            points[:-1]
            if len(points) > 1 and np.allclose(points[0], points[-1])
            else points
        )

        if len(working_points) < 3:
            return points

        # Step 1: Find trailing edge (rightmost) and leading edge (leftmost)
        te_idx = np.argmax(working_points[:, 0])
        le_idx = np.argmin(working_points[:, 0])

        if te_idx == le_idx:
            logging.warning("TE and LE are the same point")
            return points

        # Step 2: Reorder to start from trailing edge
        n_points = len(working_points)

        if te_idx == 0:
            # Already starts at TE
            te_start_points = working_points
        elif te_idx == n_points - 1:
            # TE is at end, rotate to start
            te_start_points = np.vstack(
                [working_points[te_idx:], working_points[:te_idx]]
            )
        else:
            # TE is in middle, split and reorder
            te_start_points = np.vstack(
                [working_points[te_idx:], working_points[:te_idx]]
            )

        # Step 3: INTEGRATED REORDERING LOGIC
        # Detect current ordering using aerodynamic surface analysis
        current_ordering = self.detect_point_ordering(te_start_points)

        # Apply reordering logic based on detected ordering
        if current_ordering == "clockwise":
            # Flip array to make counterclockwise while maintaining TE start
            logging.info("Detected clockwise ordering, flipping to counterclockwise")
            # Keep first point (TE) in place, reverse the rest
            reordered_points = np.vstack(
                [
                    te_start_points[0:1],  # Keep TE at start
                    te_start_points[1:][::-1],  # Reverse remaining points
                ]
            )
        elif current_ordering == "counterclockwise":
            # Already correct ordering
            logging.info(
                "Detected counterclockwise ordering, maintaining current order"
            )
            reordered_points = te_start_points
        else:
            # Unknown ordering - attempt to fix using aerodynamic analysis
            logging.warning(
                "Unknown ordering detected, attempting aerodynamic correction"
            )
            reordered_points = self._fix_unknown_ordering(
                te_start_points, te_idx, le_idx
            )

        # Step 4: Validate the reordering was successful
        final_ordering = self.detect_point_ordering(reordered_points)
        if final_ordering != "counterclockwise":
            logging.warning(
                f"Reordering validation failed: final ordering is {final_ordering}"
            )

        # Step 5: Add closure point (duplicate first point at end)
        closed_points = np.vstack([reordered_points, reordered_points[0:1]])

        return closed_points

    def _fix_unknown_ordering(
        self, points: np.ndarray, te_idx: int, le_idx: int
    ) -> np.ndarray:
        """
        Attempt to fix unknown ordering using aerodynamic principles.
        Forces counterclockwise ordering: TE -> upper surface -> LE -> lower surface -> TE
        """

        # Find the original TE and LE in the reordered array
        te_x = np.max(points[:, 0])
        le_x = np.min(points[:, 0])

        # Split points into two paths between LE and TE
        upper_surface = []
        lower_surface = []

        # Simple approach: split at LE, analyze y-coordinates
        le_new_idx = np.argmin(points[:, 0])

        # Path 1: from TE (index 0) to LE
        path1 = points[: le_new_idx + 1]
        # Path 2: from LE to end (should wrap back to TE)
        path2 = points[le_new_idx:]

        # Determine which path is upper vs lower based on average y-coordinate
        if len(path1) > 1 and len(path2) > 1:
            path1_avg_y = np.mean(path1[1:-1, 1]) if len(path1) > 2 else path1[0, 1]
            path2_avg_y = np.mean(path2[1:-1, 1]) if len(path2) > 2 else path2[0, 1]

            if path1_avg_y > path2_avg_y:
                # Path 1 is upper surface - correct counterclockwise order
                return points
            else:
                # Path 1 is lower surface - need to reverse
                return np.vstack(
                    [
                        points[0:1],  # Keep TE at start
                        points[1:][::-1],  # Reverse to get upper surface first
                    ]
                )

        # Fallback: return original if analysis fails
        return points

    def resample_airfoil(self, points: np.ndarray, target_points: int) -> np.ndarray:
        """Resample airfoil to target number of points with proper closure."""
        if len(points) < 4:
            logging.warning(f"Too few points for resampling: {len(points)}")
            return points

        # Remove duplicate points but preserve closure intention
        points_clean = self._remove_duplicate_points(points)

        if len(points_clean) < 3:
            return points

        # Calculate robust distances
        distances = np.sqrt(np.sum(np.diff(points_clean, axis=0) ** 2, axis=1))

        # Replace zero distances with small positive value
        min_distance = np.finfo(float).eps * 100
        distances = np.maximum(distances, min_distance)

        # Create cumulative distance
        cumulative_distance = np.concatenate([[0], np.cumsum(distances)])
        total_length = cumulative_distance[-1]

        # Resample to target_points - 1 (reserve space for closure point)
        new_params = np.linspace(0, total_length, target_points - 1)

        try:
            # Safe interpolation
            interp_x = interp1d(
                cumulative_distance,
                points_clean[:, 0],
                kind="linear",
                bounds_error=True,
                assume_sorted=True,
            )
            interp_y = interp1d(
                cumulative_distance,
                points_clean[:, 1],
                kind="linear",
                bounds_error=True,
                assume_sorted=True,
            )

            new_x = interp_x(new_params)
            new_y = interp_y(new_params)

            # Create resampled points
            resampled_points = np.column_stack([new_x, new_y])

            # Add closure point (duplicate of first point)
            closed_points = np.vstack([resampled_points, resampled_points[0:1]])

            # Validate for NaN
            if np.any(~np.isfinite(closed_points)):
                raise ValueError("NaN detected in resampled points")

            return closed_points

        except Exception as e:
            logging.warning(f"Interpolation failed: {e}, using fallback")
            return self._fallback_resampling_with_closure(points_clean, target_points)

    def _fallback_resampling_with_closure(
        self, points: np.ndarray, target_points: int
    ) -> np.ndarray:
        """Fallback resampling that maintains closure."""
        if len(points) >= target_points - 1:
            # Subsample
            indices = np.linspace(0, len(points) - 1, target_points - 1, dtype=int)
            resampled = points[indices]
        else:
            # Simple repetition for upsampling
            repeat_factor = (target_points - 1) // len(points)
            remainder = (target_points - 1) % len(points)

            resampled = np.tile(points, (repeat_factor, 1))
            if remainder > 0:
                extra_indices = np.linspace(0, len(points) - 1, remainder, dtype=int)
                resampled = np.vstack([resampled, points[extra_indices]])

        # Add closure point
        return np.vstack([resampled, resampled[0:1]])

    def _remove_duplicate_points(
        self, points: np.ndarray, tolerance: float = 1e-10
    ) -> np.ndarray:
        """Remove duplicate consecutive points that cause zero distances."""
        if len(points) <= 1:
            return points

        # Calculate distances between consecutive points
        point_distances = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))

        # Keep first point and points that are sufficiently far from previous
        keep_indices = [0]  # Always keep first point
        for i in range(1, len(points)):
            if point_distances[i - 1] > tolerance:
                keep_indices.append(i)

        # Always keep last point if it's not already kept
        if keep_indices[-1] != len(points) - 1:
            keep_indices.append(len(points) - 1)

        return points[keep_indices]

    def process_single_airfoil(self, airfoil_name: str) -> Optional[np.ndarray]:
        """Process a single airfoil through the complete preparation pipeline."""
        # Get airfoil data from database
        airfoil_data = self.db.get_airfoil_data(airfoil_name)
        if not airfoil_data:
            logging.error(f"Airfoil {airfoil_name} not found in database")
            return None

        description, pointcloud_str, series, source = airfoil_data

        # Parse point cloud
        original_points = self.parse_pointcloud_from_string(pointcloud_str)
        if len(original_points) == 0:
            logging.error(f"No valid points found for {airfoil_name}")
            return None

        logging.info(
            f"Processing {airfoil_name}: {len(original_points)} original points"
        )

        # Normalize coordinates
        normalized_points = self.normalize_airfoil_coordinates(original_points)

        # Standardize ordering
        ordered_points = self.standardize_point_ordering(normalized_points)

        # Resample to target points
        resampled_points = self.resample_airfoil(ordered_points, self.target_points)

        logging.info(f"Resampled {airfoil_name}: {len(resampled_points)} points")

        return resampled_points

    def process_all_airfoils(self) -> dict:
        """Process all airfoils in the database."""
        processed_data = {}

        with Session(self.db.engine) as session:
            statement = select(Airfoil.name)
            airfoil_names = [row for row in session.exec(statement).all()]

        logging.info(f"Processing {len(airfoil_names)} airfoils...")

        successful_count = 0
        for airfoil_name in airfoil_names:
            try:
                processed_points = self.process_single_airfoil(airfoil_name)
                if processed_points is not None:
                    processed_data[airfoil_name] = processed_points
                    successful_count += 1
            except Exception as e:
                logging.error(f"Error processing {airfoil_name}: {e}")
                continue

        logging.info(
            f"Successfully processed {successful_count}/{len(airfoil_names)} airfoils"
        )
        return processed_data

    def save_processed_data(self, processed_data: dict, output_file: str):
        """Save processed data to file."""
        # Convert numpy arrays to lists for JSON serialization
        serializable_data = {}
        for name, points in processed_data.items():
            serializable_data[name] = points.tolist()

        with open(output_file, "w") as f:
            json.dump(serializable_data, f, indent=2)

        logging.info(f"Saved processed data to {output_file}")

    def validate_point_ordering(self, points: np.ndarray, airfoil_name: str) -> dict:
        """
        Comprehensive validation of point ordering standardization.
        Returns detailed diagnostics for debugging failed cases.
        """
        validation_results = {
            "airfoil_name": airfoil_name,
            "success": True,
            "issues": [],
            "diagnostics": {},
        }

        if len(points) == 0:
            validation_results["success"] = False
            validation_results["issues"].append("Empty point array")
            return validation_results

        # Check 1: Verify trailing edge detection
        te_idx = np.argmax(points[:, 0])
        le_idx = np.argmin(points[:, 0])

        validation_results["diagnostics"]["te_idx"] = te_idx
        validation_results["diagnostics"]["le_idx"] = le_idx
        validation_results["diagnostics"]["te_coordinate"] = points[te_idx].tolist()
        validation_results["diagnostics"]["le_coordinate"] = points[le_idx].tolist()

        # Check 2: Verify airfoil closure (should start and end near same x-coordinate)
        x_start = points[0, 0]
        x_end = points[-1, 0]
        x_closure_diff = abs(x_start - x_end)

        validation_results["diagnostics"]["x_closure_diff"] = x_closure_diff
        if x_closure_diff > 0.1:  # Threshold for acceptable closure
            validation_results["success"] = False
            validation_results["issues"].append(
                f"Poor airfoil closure: start_x={x_start:.3f}, end_x={x_end:.3f}"
            )

        # Check 3: Verify consistent ordering direction
        ordering = self.detect_point_ordering(points)
        validation_results["diagnostics"]["detected_ordering"] = ordering

        if ordering == "unknown":
            validation_results["success"] = False
            validation_results["issues"].append(
                "Could not determine point ordering direction"
            )

        # Check 4: Verify no duplicate points
        if len(points) > 1:
            distances = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
            min_distance = np.min(distances)
            duplicate_count = np.sum(distances < 1e-6)

            validation_results["diagnostics"]["min_distance"] = min_distance
            validation_results["diagnostics"]["duplicate_points"] = int(duplicate_count)

            if duplicate_count > 0:
                validation_results["issues"].append(
                    f"{duplicate_count} duplicate/nearly duplicate points found"
                )

        # Check 5: Verify reasonable geometric bounds
        x_range = np.max(points[:, 0]) - np.min(points[:, 0])
        y_range = np.max(points[:, 1]) - np.min(points[:, 1])

        validation_results["diagnostics"]["x_range"] = x_range
        validation_results["diagnostics"]["y_range"] = y_range

        if x_range < 0.5:  # Airfoil should span reasonable x range
            validation_results["success"] = False
            validation_results["issues"].append(
                f"Unusually small x-range: {x_range:.3f}"
            )

        # Check 6: Verify TE is rightmost and LE is leftmost
        if te_idx == le_idx:
            validation_results["success"] = False
            validation_results["issues"].append(
                "Trailing edge and leading edge are the same point"
            )

        return validation_results

    def _validate_final_ordering(
        self, standardized_points: np.ndarray, airfoil_name: str
    ) -> bool:
        """Final validation of standardized ordering."""

        if len(standardized_points) == 0:
            return False

        # Check 1: Verify counterclockwise ordering
        ordering = self.detect_point_ordering(standardized_points)
        if ordering != "counterclockwise":
            logging.warning(
                f"Final ordering not counterclockwise for {airfoil_name}: {ordering}"
            )
            return False

        # Check 2: Verify airfoil shape preservation
        original_bounds = self._get_bounding_box(self.original_points)
        final_bounds = self._get_bounding_box(standardized_points)

        if not self._bounds_similar(original_bounds, final_bounds, tolerance=0.1):
            logging.warning(f"Shape significantly changed for {airfoil_name}")
            return False

        # Check 3: Verify smooth traversal (no major jumps)
        distances = np.sqrt(np.sum(np.diff(standardized_points, axis=0) ** 2, axis=1))
        max_jump = np.max(distances)
        median_distance = np.median(distances)

        if max_jump > 10 * median_distance:
            logging.warning(f"Large traversal jump detected for {airfoil_name}")
            return False

        return True

    def process_single_airfoil_with_validation(
        self, airfoil_name: str
    ) -> Tuple[Optional[np.ndarray], dict]:
        """Enhanced processing with comprehensive validation and debugging."""
        # Get airfoil data from database
        airfoil_data = self.db.get_airfoil_data(airfoil_name)
        if not airfoil_data:
            return None, {"error": "Airfoil not found in database"}

        description, pointcloud_str, series, source = airfoil_data

        # Parse point cloud
        original_points = self.parse_pointcloud_from_string(pointcloud_str)
        self.original_points = original_points
        if len(original_points) == 0:
            return None, {"error": "No valid points found"}

        # Validate original points
        original_validation = self.validate_point_ordering(
            original_points, airfoil_name
        )

        # Process through pipeline
        normalized_points = self.normalize_airfoil_coordinates(original_points)
        ordered_points = self.standardize_point_ordering(normalized_points)
        resampled_points = self.resample_airfoil(ordered_points, self.target_points)

        # Validate final result
        final_validation = self.validate_point_ordering(resampled_points, airfoil_name)

        # Compile comprehensive diagnostics
        diagnostics = {
            "original_validation": original_validation,
            "final_validation": final_validation,
            "original_point_count": len(original_points),
            "final_point_count": len(resampled_points),
            "processing_successful": final_validation["success"],
        }

        if not final_validation["success"]:
            logging.warning(
                f"Point ordering validation failed for {airfoil_name}: {final_validation['issues']}"
            )

        return resampled_points, diagnostics

    def process_all_airfoils_with_diagnostics(self) -> Tuple[dict, dict]:
        """Process all airfoils with comprehensive failure tracking and diagnostics."""
        processed_data = {}
        diagnostics_data = {}

        with Session(self.db.engine) as session:
            statement = select(Airfoil.name)
            airfoil_names = [row for row in session.exec(statement).all()]

        logging.info(f"Processing {len(airfoil_names)} airfoils with validation...")

        successful_count = 0
        failed_airfoils = []

        for airfoil_name in airfoil_names:
            try:
                processed_points, diagnostics = (
                    self.process_single_airfoil_with_validation(airfoil_name)
                )

                diagnostics_data[airfoil_name] = diagnostics

                if (
                    processed_points is not None
                    and diagnostics["processing_successful"]
                ):
                    processed_data[airfoil_name] = processed_points
                    successful_count += 1
                else:
                    failed_airfoils.append(airfoil_name)
                    logging.error(
                        f"Failed to process {airfoil_name}: {diagnostics.get('error', 'Validation failed')}"
                    )

            except Exception as e:
                logging.error(f"Exception processing {airfoil_name}: {e}")
                failed_airfoils.append(airfoil_name)
                diagnostics_data[airfoil_name] = {
                    "error": str(e),
                    "processing_successful": False,
                }

        # Generate summary report
        failure_summary = self._generate_failure_summary(
            diagnostics_data, failed_airfoils
        )

        logging.info(
            f"Processing complete: {successful_count}/{len(airfoil_names)} successful"
        )
        logging.info(f"Failed airfoils: {len(failed_airfoils)}")

        return processed_data, diagnostics_data

    def _generate_failure_summary(
        self, diagnostics_data: dict, failed_airfoils: List[str]
    ) -> dict:
        """Generate summary of failure patterns to help debug issues."""
        failure_summary = {
            "total_failures": len(failed_airfoils),
            "failure_types": {},
            "problematic_airfoils": failed_airfoils,
            "common_issues": [],
        }

        # Analyze failure patterns
        issue_counts = {}
        for airfoil_name, diagnostics in diagnostics_data.items():
            if not diagnostics.get("processing_successful", True):
                final_validation = diagnostics.get("final_validation", {})
                issues = final_validation.get("issues", [])

                for issue in issues:
                    issue_type = issue.split(":")[0]  # Get general issue type
                    issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1

        failure_summary["failure_types"] = issue_counts
        failure_summary["common_issues"] = sorted(
            issue_counts.items(), key=lambda x: x[1], reverse=True
        )

        return failure_summary

    def create_failed_airfoil_debug_plots(
        self, failed_airfoils: List[str], output_dir: str = "debug_figures"
    ):
        """Create detailed plots for failed airfoils to help debug issues."""
        Path(output_dir).mkdir(exist_ok=True)

        for airfoil_name in failed_airfoils[:5]:  # Debug first 5 failures
            try:
                airfoil_data = self.db.get_airfoil_data(airfoil_name)
                if not airfoil_data:
                    continue

                original_points = self.parse_pointcloud_from_string(airfoil_data[1])
                if len(original_points) == 0:
                    continue

                # Create debug figure
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                fig.suptitle(
                    f"Debug Analysis: {airfoil_name}", fontsize=16, fontweight="bold"
                )

                # Plot 1: Original points with indices
                ax1 = axes[0, 0]
                ax1.plot(
                    original_points[:, 0], original_points[:, 1], "b-o", markersize=3
                )
                ax1.plot(
                    original_points[0, 0],
                    original_points[0, 1],
                    "go",
                    markersize=8,
                    label="Start",
                )
                ax1.plot(
                    original_points[-1, 0],
                    original_points[-1, 1],
                    "ro",
                    markersize=8,
                    label="End",
                )

                # Mark TE and LE
                te_idx = np.argmax(original_points[:, 0])
                le_idx = np.argmin(original_points[:, 0])
                ax1.plot(
                    original_points[te_idx, 0],
                    original_points[te_idx, 1],
                    "s",
                    color="red",
                    markersize=10,
                    label="Detected TE",
                )
                ax1.plot(
                    original_points[le_idx, 0],
                    original_points[le_idx, 1],
                    "s",
                    color="blue",
                    markersize=10,
                    label="Detected LE",
                )

                ax1.set_title("Original Points with Key Indices")
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                ax1.axis("equal")

                # Plot 2: Normalized points
                ax2 = axes[0, 1]
                normalized_points = self.normalize_airfoil_coordinates(original_points)
                ax2.plot(
                    normalized_points[:, 0],
                    normalized_points[:, 1],
                    "g-o",
                    markersize=3,
                )
                ax2.set_title("After Normalization")
                ax2.grid(True, alpha=0.3)
                ax2.axis("equal")

                # Plot 3: Attempted reordering
                ax3 = axes[1, 0]
                try:
                    ordered_points = self.standardize_point_ordering(normalized_points)
                    ax3.plot(
                        ordered_points[:, 0], ordered_points[:, 1], "m-o", markersize=3
                    )
                    ax3.plot(
                        ordered_points[0, 0],
                        ordered_points[0, 1],
                        "go",
                        markersize=8,
                        label="Start",
                    )
                    ax3.set_title("After Attempted Reordering")
                except Exception as e:
                    ax3.text(
                        0.5,
                        0.5,
                        f"Reordering failed:\n{str(e)}",
                        transform=ax3.transAxes,
                        ha="center",
                        va="center",
                    )
                    ax3.set_title("Reordering Failed")
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                ax3.axis("equal")

                # Plot 4: Diagnostics text
                ax4 = axes[1, 1]
                diagnostics = self.validate_point_ordering(
                    original_points, airfoil_name
                )

                diagnostics_text = f"Validation Results:\n"
                diagnostics_text += f"Success: {diagnostics['success']}\n\n"
                diagnostics_text += "Issues:\n"
                for issue in diagnostics["issues"]:
                    diagnostics_text += f"• {issue}\n"

                diagnostics_text += f"\nDiagnostics:\n"
                for key, value in diagnostics["diagnostics"].items():
                    diagnostics_text += f"• {key}: {value}\n"

                ax4.text(
                    0.05,
                    0.95,
                    diagnostics_text,
                    transform=ax4.transAxes,
                    fontsize=10,
                    verticalalignment="top",
                    fontfamily="monospace",
                )
                ax4.set_xlim(0, 1)
                ax4.set_ylim(0, 1)
                ax4.axis("off")

                plt.tight_layout()
                plt.savefig(
                    f"{output_dir}/debug_{airfoil_name.replace('/', '_')}.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()

            except Exception as e:
                logging.error(f"Error creating debug plot for {airfoil_name}: {e}")
