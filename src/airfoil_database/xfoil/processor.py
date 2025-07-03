# xfoil/processor.py
import os
import logging
import numpy as np
import tempfile
import subprocess
from typing import List, Optional, Dict, Any, Tuple

class PointcloudProcessor:
    @staticmethod
    def fix_airfoil_pointcloud(name, pointcloud_str, config=None):
        """
        Fixes issues with airfoil pointcloud data.
        
        Args:
            name (str): The name of the airfoil
            pointcloud_str (str): The pointcloud data as a string
            config (dict, optional): Configuration for the fixing process
            
        Returns:
            np.ndarray or None: The fixed pointcloud as a numpy array, or None if fixing failed
        """
        from airfoil_database.utils.helpers import parse_pointcloud_string, fix_pointcloud
        
        logging.info(f"Attempting to fix pointcloud for airfoil: {name}")
        
        # Parse the pointcloud string
        points_array = parse_pointcloud_string(pointcloud_str)
        if points_array is None:
            logging.error(f"Could not parse original pointcloud string for {name}.")
            return None  # Parsing failed

        # Get the default configuration if none provided
        from airfoil_database.utils.helpers import DEFAULT_FIXER_CONFIG
        fixer_config = config if config is not None else DEFAULT_FIXER_CONFIG

        # Call the fixing function
        fixed_points_array = fix_pointcloud(points_array, fixer_config)

        if fixed_points_array is None:
            logging.error(f"Pointcloud fixing failed for {name}.")
            return None
        else:
            logging.info(f"Pointcloud fixing successful for {name}.")
            return fixed_points_array
    
    @staticmethod
    def output_pointcloud_to_file(pointcloud_str, file_path):
        """
        Outputs the point cloud string to a text file.

        Args:
            pointcloud_str (str): The pointcloud data as a string
            file_path (str): The path to the output text file.
        """
        try:
            with open(file_path, 'w') as file:
                file.write(pointcloud_str)
            return True
        except Exception as e:
            logging.error(f"Error outputting point cloud to file: {e}")
            return False
    
    @staticmethod
    def find_best_matching_airfoils(input_pointcloud_str, database, num_matches=3):
        """
        Compares an input point cloud to the airfoils in the database and returns the best matches.
        
        Args:
            input_pointcloud_str (str): The input pointcloud to compare
            database: The database instance to query
            num_matches (int): Number of matches to return
            
        Returns:
            list: List of tuples (name, distance) of the best matching airfoils
        """
        from airfoil_database.xfoil.fix_airfoil_data import normalize_pointcloud
        from airfoil_database.xfoil.interpolate_points import interpolate_points
        from airfoil_database.xfoil.calculate_distance import calculate_min_distance_sum
        from airfoil_database.utils.helpers import parse_pointcloud_string
        
        input_points = parse_pointcloud_string(input_pointcloud_str)
        if input_points is None or len(input_points) == 0:
            return []

        normalized_input_points = normalize_pointcloud(input_points)
        if len(normalized_input_points) == 0:
            return []

        interpolated_input_points = interpolate_points(normalized_input_points)

        matches = []
        with Session(database.engine) as session:
            statement = select(Airfoil.name, Airfoil.pointcloud)
            airfoils = session.exec(statement).all()

            for name, db_pointcloud_str in airfoils:
                db_points = parse_pointcloud_string(db_pointcloud_str)
                if db_points is None or len(db_points) == 0:
                    continue
                    
                normalized_db_points = normalize_pointcloud(db_points)
                if len(normalized_db_points) == 0:
                    continue
                    
                interpolated_db_points = interpolate_points(normalized_db_points)

                if np.shape(interpolated_input_points)[0] == np.shape(interpolated_db_points)[0]:
                    distance = calculate_min_distance_sum(interpolated_input_points, interpolated_db_points)
                    matches.append((name, distance))

        matches.sort(key=lambda x: x[1])  # Sort by distance
        return matches[:num_matches]  # Return the top matches
