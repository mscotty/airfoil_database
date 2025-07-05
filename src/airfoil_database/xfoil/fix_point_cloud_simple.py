import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AirfoilProcessor:
    """
    A simplified and robust airfoil point cloud processor.
    Handles parsing, validation, reordering, and formatting of airfoil coordinates.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with configuration parameters."""
        self.config = {
            "min_points": 10,
            "max_te_gap_ratio": 0.01,  # Max TE gap as ratio of chord length
            "min_point_distance": 1e-6,  # Minimum distance between points
            "precision": 6,  # Output precision
            "normalize": True,  # Whether to normalize coordinates
        }
        if config:
            self.config.update(config)
    
    def parse_pointcloud(self, data: str) -> np.ndarray:
        """
        Parse airfoil point cloud data from string format.
        
        Args:
            data: String containing x,y coordinates (space or comma separated)
            
        Returns:
            numpy array of shape (n, 2) containing coordinates
        """
        try:
            lines = [line.strip() for line in data.strip().split('\n') if line.strip()]
            points = []
            
            for line in lines:
                # Handle both space and comma separated values
                parts = line.replace(',', ' ').split()
                if len(parts) >= 2:
                    x, y = float(parts[0]), float(parts[1])
                    points.append([x, y])
            
            if not points:
                raise ValueError("No valid points found")
                
            return np.array(points)
            
        except Exception as e:
            logger.error(f"Failed to parse point cloud: {e}")
            raise ValueError(f"Invalid point cloud format: {e}")
    
    def validate_pointcloud(self, points: np.ndarray) -> Tuple[bool, str]:
        """
        Validate basic properties of the point cloud.
        
        Args:
            points: numpy array of coordinates
            
        Returns:
            Tuple of (is_valid, message)
        """
        if points is None or not isinstance(points, np.ndarray):
            return False, "Invalid input: not a numpy array"
        
        if points.shape[0] < self.config["min_points"]:
            return False, f"Too few points: {points.shape[0]} < {self.config['min_points']}"
        
        if points.shape[1] != 2:
            return False, f"Invalid dimensions: expected 2D points, got {points.shape[1]}D"
        
        # Check for NaN or infinite values
        if not np.all(np.isfinite(points)):
            return False, "Contains NaN or infinite values"
        
        return True, "Valid"
    
    def remove_duplicates(self, points: np.ndarray) -> np.ndarray:
        """
        Remove duplicate or very close points.
        
        Args:
            points: array of coordinates
            
        Returns:
            Array with duplicates removed
        """
        if len(points) <= 1:
            return points
        
        result = [points[0]]
        min_dist = self.config["min_point_distance"]
        
        for i in range(1, len(points)):
            # Calculate distance to last kept point
            if np.linalg.norm(points[i] - result[-1]) >= min_dist:
                result.append(points[i])
        
        return np.array(result)
    
    def separate_surfaces(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Separate airfoil points into suction side (SS) and pressure side (PS) surfaces.
        
        Uses the specified logic:
        - LE: most negative x value (should be at 0,0)
        - TE: most positive x value (should be around x=1)
        - SS: continuous points with positive y values on average
        - PS: continuous points with negative/smaller y values on average
        
        Args:
            points: combined point cloud
            
        Returns:
            Tuple of (le_point, te_point, ss_points, ps_points)
        """
        # Find leading edge (most negative x) and trailing edge (most positive x)
        le_idx = np.argmin(points[:, 0])
        te_idx = np.argmax(points[:, 0])
        
        le_point = points[le_idx]
        te_point = points[te_idx]
        
        # Get all points except LE and TE
        remaining_points = []
        remaining_indices = []
        
        for i, point in enumerate(points):
            if i != le_idx and i != te_idx:
                remaining_points.append(point)
                remaining_indices.append(i)
        
        remaining_points = np.array(remaining_points)
        
        if len(remaining_points) == 0:
            logger.warning("No points found between LE and TE")
            return le_point, te_point, np.array([]), np.array([])
        
        # Separate based on y-values relative to a reference line
        # For airfoils, we can use the chord line as reference
        chord_vector = te_point - le_point
        chord_length = np.linalg.norm(chord_vector)
        
        if chord_length < 1e-10:
            logger.warning("Chord length too small, separating by y-coordinate only")
            # Fallback: separate by y-coordinate relative to LE
            ss_points = remaining_points[remaining_points[:, 1] >= le_point[1]]
            ps_points = remaining_points[remaining_points[:, 1] < le_point[1]]
        else:
            # Use cross product to determine which side of chord line
            ss_points = []
            ps_points = []
            
            for point in remaining_points:
                # Vector from LE to current point
                point_vector = point - le_point
                # Cross product gives signed area (positive = above chord line)
                cross_product = np.cross(chord_vector, point_vector)
                
                if cross_product >= 0:  # Above or on chord line = suction side
                    ss_points.append(point)
                else:  # Below chord line = pressure side
                    ps_points.append(point)
            
            ss_points = np.array(ss_points) if ss_points else np.empty((0, 2))
            ps_points = np.array(ps_points) if ps_points else np.empty((0, 2))
        
        # Verify separation makes sense
        if len(ss_points) > 0 and len(ps_points) > 0:
            ss_mean_y = np.mean(ss_points[:, 1])
            ps_mean_y = np.mean(ps_points[:, 1])
            
            if ss_mean_y <= ps_mean_y:
                logger.warning("SS points have lower average y than PS points - this may indicate incorrect separation")
        
        return le_point, te_point, ss_points, ps_points
    
    def sort_surface_points(self, points: np.ndarray, reverse: bool = False) -> np.ndarray:
        """
        Sort surface points by x-coordinate.
        
        Args:
            points: surface points
            reverse: if True, sort in descending order
            
        Returns:
            Sorted points
        """
        if len(points) == 0:
            return points
        
        indices = np.argsort(points[:, 0])
        if reverse:
            indices = indices[::-1]
        
        return points[indices]
    
    def reorder_points(self, points: np.ndarray) -> np.ndarray:
        """
        Reorder points to follow LE -> SS -> TE -> PS -> LE format.
        
        Logic:
        - LE: point with most negative x (should be at 0,0)
        - SS: suction side points (positive y on average)
        - TE: point with most positive x (should be around x=1)
        - PS: pressure side points (negative/smaller y on average)
        
        Args:
            points: input point cloud
            
        Returns:
            Reordered points
        """
        le_point, te_point, ss_points, ps_points = self.separate_surfaces(points)
        
        # Sort SS points by increasing x (LE to TE direction)
        if len(ss_points) > 0:
            ss_indices = np.argsort(ss_points[:, 0])
            ss_sorted = ss_points[ss_indices]
        else:
            ss_sorted = np.array([])
        
        # Sort PS points by decreasing x (TE to LE direction)
        if len(ps_points) > 0:
            ps_indices = np.argsort(-ps_points[:, 0])  # Negative for descending order
            ps_sorted = ps_points[ps_indices]
        else:
            ps_sorted = np.array([])
        
        # Assemble final order: LE -> SS -> TE -> PS -> LE
        reordered = [le_point]
        
        # Add suction side points
        if len(ss_sorted) > 0:
            reordered.extend(ss_sorted)
        
        # Add trailing edge
        reordered.append(te_point)
        
        # Add pressure side points
        if len(ps_sorted) > 0:
            reordered.extend(ps_sorted)
        
        # Close the loop by returning to LE
        reordered.append(le_point)
        
        return np.array(reordered)
    
    def check_closure(self, points: np.ndarray) -> Tuple[bool, float]:
        """
        Check if the airfoil is properly closed (TE gap).
        
        Args:
            points: airfoil coordinates
            
        Returns:
            Tuple of (is_closed, gap_distance)
        """
        if len(points) < 3:
            return False, float('inf')
        
        # Calculate chord length for relative gap assessment
        chord_length = np.max(points[:, 0]) - np.min(points[:, 0])
        
        # Gap between first and last points
        gap = np.linalg.norm(points[0] - points[-1])
        max_allowed_gap = self.config["max_te_gap_ratio"] * chord_length
        
        is_closed = gap <= max_allowed_gap
        
        return is_closed, gap
    
    def close_trailing_edge(self, points: np.ndarray) -> np.ndarray:
        """
        Force closure by averaging first and last points.
        
        Args:
            points: airfoil coordinates
            
        Returns:
            Points with forced closure
        """
        if len(points) < 2:
            return points
        
        # Average first and last points
        avg_point = (points[0] + points[-1]) / 2.0
        
        # Update both first and last points
        points = points.copy()
        points[0] = avg_point
        points[-1] = avg_point
        
        return points
    
    def normalize_airfoil(self, points: np.ndarray) -> np.ndarray:
        """
        Normalize airfoil coordinates so LE is at (0,0) and TE is at (1,y).
        
        Args:
            points: airfoil coordinates
            
        Returns:
            Normalized coordinates with LE at origin
        """
        if len(points) < 2:
            return points
        
        # Find LE point (most negative x)
        le_idx = np.argmin(points[:, 0])
        le_point = points[le_idx]
        
        # Translate so LE is at origin
        points_translated = points - le_point
        
        # Find TE point after translation (most positive x)
        te_idx = np.argmax(points_translated[:, 0])
        chord_length = points_translated[te_idx, 0]
        
        # Scale so chord length = 1 (TE x-coordinate = 1)
        if chord_length > 1e-10:
            points_normalized = points_translated / chord_length
        else:
            logger.warning("Chord length too small for normalization")
            points_normalized = points_translated
        
        return points_normalized
    
    def format_output(self, points: np.ndarray) -> str:
        """
        Format points as string with specified precision.
        
        Args:
            points: airfoil coordinates
            
        Returns:
            Formatted string
        """
        if len(points) == 0:
            return ""
        
        precision = self.config["precision"]
        lines = []
        
        for x, y in points:
            lines.append(f"{x:.{precision}f} {y:.{precision}f}")
        
        return "\n".join(lines)
    
    def process(self, data: str) -> Tuple[str, Dict[str, Any]]:
        """
        Main processing function that handles the complete workflow.
        
        Args:
            data: input point cloud string
            
        Returns:
            Tuple of (formatted_output, processing_info)
        """
        info = {"status": "success", "messages": []}
        
        try:
            # Parse input
            points = self.parse_pointcloud(data)
            info["original_points"] = len(points)
            
            # Validate
            is_valid, message = self.validate_pointcloud(points)
            if not is_valid:
                info["status"] = "error"
                info["messages"].append(f"Validation failed: {message}")
                return "", info
            
            # Remove duplicates
            points = self.remove_duplicates(points)
            if len(points) < info["original_points"]:
                info["messages"].append(f"Removed {info['original_points'] - len(points)} duplicate points")
            
            # Reorder points
            points = self.reorder_points(points)
            info["messages"].append("Points reordered to LE->SS->TE->PS->LE format")
            
            # Validate ordering
            le_x = points[0, 0]
            te_x = np.max(points[:, 0])
            info["messages"].append(f"LE x-coordinate: {le_x:.6f}")
            info["messages"].append(f"TE x-coordinate: {te_x:.6f}")
            
            # Check surface separation
            le_point, te_point, ss_points, ps_points = self.separate_surfaces(points)
            if len(ss_points) > 0 and len(ps_points) > 0:
                ss_mean_y = np.mean(ss_points[:, 1])
                ps_mean_y = np.mean(ps_points[:, 1])
                info["messages"].append(f"SS mean y: {ss_mean_y:.6f}, PS mean y: {ps_mean_y:.6f}")
            info["messages"].append(f"SS points: {len(ss_points)}, PS points: {len(ps_points)}")
            
            # Check closure
            is_closed, gap = self.check_closure(points)
            if not is_closed:
                info["messages"].append(f"Trailing edge gap: {gap:.6f}")
                points = self.close_trailing_edge(points)
                info["messages"].append("Forced trailing edge closure")
            
            # Normalize if requested
            if self.config["normalize"]:
                points = self.normalize_airfoil(points)
                info["messages"].append("Normalized: LE at (0,0), TE at (1,y)")
                
                # Report final LE and TE positions
                le_final = points[0]
                te_idx = np.argmax(points[:, 0])
                te_final = points[te_idx]
                info["messages"].append(f"Final LE: ({le_final[0]:.6f}, {le_final[1]:.6f})")
                info["messages"].append(f"Final TE: ({te_final[0]:.6f}, {te_final[1]:.6f})")
            
            info["final_points"] = len(points)
            
            # Format output
            output = self.format_output(points)
            
            return output, info
            
        except Exception as e:
            info["status"] = "error"
            info["messages"].append(f"Processing failed: {str(e)}")
            logger.error(f"Processing error: {e}")
            return "", info

# Example usage
def main():
    # Your sample data
    sample_data = """0.0000000 0.0000000
0.0024080 0.0108390
0.0096070 0.0192660
0.0215300 0.0271400
0.0380600 0.0344180
0.0590400 0.0411140
0.0842660 0.0472260
0.1134950 0.0527710
0.1464470 0.0577620
0.1828040 0.0621980
0.2222160 0.0660400
0.2643030 0.0692740
0.3086600 0.0718150
0.3548590 0.0735910
0.4024560 0.0744990
0.4509930 0.0744640
0.5000000 0.0734500
0.5490110 0.0714650
0.5975470 0.0685510
0.6451440 0.0647890
0.6913440 0.0602770
0.7357000 0.0551750
0.7777870 0.0491260
0.8171990 0.0422990
0.8535550 0.0351740
0.8865070 0.0282830
0.9157360 0.0216810
0.9409620 0.0156440
0.9619410 0.0104390
0.9784710 0.0062140
0.9903930 0.0030990
0.9975930 0.0011910
1.0000000 0.0005480

0.0000000 0.0000000
0.0024080 -.0076950
0.0096070 -.0155630
0.0215300 -.0227350
0.0380600 -.0289970
0.0590400 -.0344570
0.0842660 -.0391860
0.1134950 -.0432590
0.1464470 -.0467340
0.1828040 -.0496460
0.2222160 -.0520120
0.2643030 -.0538220
0.3086600 -.0550370
0.3548590 -.0556020
0.4024560 -.0554370
0.4509930 -.0544790
0.5000000 -.0527110
0.5490110 -.0501320
0.5975470 -.0468570
0.6451440 -.0428950
0.6913440 -.0384170
0.7357000 -.0336200
0.7777870 -.0290480
0.8171990 -.0247450
0.8535550 -.0204690
0.8865070 -.0164210
0.9157360 -.0127150
0.9409620 -.0092580
0.9619410 -.0062300
0.9784710 -.0037810
0.9903930 -.0019720
0.9975930 -.0009020
1.0000000 -.0005480"""

    # Process the airfoil
    processor = AirfoilProcessor()
    result, info = processor.process(sample_data)
    
    print("Processing Info:")
    print(f"Status: {info['status']}")
    for message in info['messages']:
        print(f"  - {message}")
    
    print(f"\nOriginal points: {info.get('original_points', 0)}")
    print(f"Final points: {info.get('final_points', 0)}")
    
    print("\nProcessed Airfoil:")
    #import matplotlib.pyplot as plt
    #from airfoil_database.utilities.helpers import pointcloud_string_to_array
    #points = pointcloud_string_to_array(result)
    #plt.plot(points[:,0], points[:,1])
    #plt.show()
    print(result)

if __name__ == "__main__":
    main()