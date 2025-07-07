# analysis/geometry_analyzer.py
import time
import numpy as np
import pandas as pd
import logging
from sqlmodel import Session, select
from typing import List, Optional, Dict, Any, Tuple
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

from airfoil_database.core.models import Airfoil, AirfoilGeometry
from airfoil_database.utilities.helpers import pointcloud_string_to_array_optimized
from airfoil_database.utilities.parallel_processing import parallel_map
from airfoil_database.formulas.airfoil.compute_aspect_ratio import calculate_aspect_ratio
from airfoil_database.formulas.airfoil.compute_LE_radius import leading_edge_radius
from airfoil_database.formulas.airfoil.compute_TE_angle import trailing_edge_angle
from airfoil_database.formulas.airfoil.compute_thickness_camber import compute_thickness_camber
from airfoil_database.formulas.airfoil.compute_thickness_to_chord_ratio import thickness_to_chord_ratio


def process_airfoil(airfoil_data):
    """Process a single airfoil in a separate process."""
    name, pointcloud = airfoil_data
    
    if not pointcloud:
        return None
    
    try:
        # Convert pointcloud string to numpy array using optimized function
        points = pointcloud_string_to_array_optimized(pointcloud)
        
        if points.size == 0:
            return None
        
        # Compute metrics
        x_coords, thickness, camber = compute_thickness_camber(points)
        LE_radius = leading_edge_radius(points)
        TE_angle = trailing_edge_angle(points)
        chord_length = max(x_coords) - min(x_coords)
        t_to_c = thickness_to_chord_ratio(thickness, chord_length)
        aspect_ratio = calculate_aspect_ratio(1, chord_length)
        max_thickness = max(thickness)
        max_camber = max(camber)
        
        # Calculate normalized chord
        normalized_chord = np.linspace(0, 1, len(thickness))

        # Store distribution data as comma-separated strings
        thickness_dist_str = ",".join(map(str, thickness))
        camber_dist_str = ",".join(map(str, camber))
        normalized_chord_str = ",".join(map(str, normalized_chord))
        
        return {
            'name': name,
            'max_thickness': max_thickness,
            'max_camber': max_camber,
            'leading_edge_radius': LE_radius,
            'trailing_edge_angle': TE_angle,
            'chord_length': chord_length,
            'thickness_to_chord_ratio': t_to_c,
            'aspect_ratio': aspect_ratio,
            'thickness_distribution': thickness_dist_str,
            'camber_distribution': camber_dist_str,
            'normalized_chord': normalized_chord_str
        }
    except Exception as e:
        print(f"Error processing airfoil {name}: {str(e)}")
        return None


class GeometryAnalyzer:
    def __init__(self, database):
        self.db = database
        self.engine = database.engine
    
    def check_pointcloud_outliers(self, name, threshold=3.0):
        """Checks for outliers in the pointcloud of a given airfoil using vectorized operations."""
        with Session(self.engine) as session:
            statement = select(Airfoil.pointcloud).where(Airfoil.name == name)
            result = session.exec(statement).first()

            if not result:
                return False, "Airfoil not found"
                
            pointcloud_np = pointcloud_string_to_array_optimized(result)
            if pointcloud_np.size == 0:
                return False, "Empty pointcloud"
                
            # Calculate z-scores in one go
            mean = np.mean(pointcloud_np, axis=0)
            std = np.std(pointcloud_np, axis=0)
            
            # Handle zero standard deviation case
            std = np.where(std == 0, 1e-10, std)
            
            z_scores = np.abs((pointcloud_np - mean) / std)
            
            # Find outliers (points where either x or y is an outlier)
            outlier_mask = np.any(z_scores > threshold, axis=1)
            outlier_indices = np.where(outlier_mask)[0]
            
            if len(outlier_indices) > 0:
                outlier_points = pointcloud_np[outlier_indices]
                return True, outlier_points
            else:
                return False, None

    def check_all_pointcloud_outliers(self, threshold=3.0):
        """Checks all airfoils in the database for outliers with optimized database access."""
        outliers_found = {}

        with Session(self.engine) as session:
            # Get all airfoils in one query
            statement = select(Airfoil.name, Airfoil.pointcloud)
            airfoils = session.exec(statement).all()

            for name, pointcloud in airfoils:
                if not pointcloud:
                    continue
                    
                pointcloud_np = pointcloud_string_to_array_optimized(pointcloud)
                if pointcloud_np.size == 0:
                    continue
                    
                # Calculate z-scores in one go
                mean = np.mean(pointcloud_np, axis=0)
                std = np.std(pointcloud_np, axis=0)
                
                # Handle zero standard deviation case
                std = np.where(std == 0, 1e-10, std)
                
                z_scores = np.abs((pointcloud_np - mean) / std)
                
                # Find outliers (points where either x or y is an outlier)
                outlier_mask = np.any(z_scores > threshold, axis=1)
                outlier_indices = np.where(outlier_mask)[0]
                
                if len(outlier_indices) > 0:
                    outlier_points = pointcloud_np[outlier_indices]
                    outliers_found[name] = outlier_points

        # Print results
        if outliers_found:
            print(f"Found outliers in {len(outliers_found)} airfoils")
            for name, outliers in outliers_found.items():
                print(f"- {name}: {len(outliers)} outlier points")
        else:
            print("No outliers found in any airfoils.")
        return outliers_found

    def compute_geometry_metrics(self, batch_size=100):
        """Computes and stores geometry metrics for all airfoils in the database in batches."""
        with Session(self.engine) as session:
            # First, get all existing geometry records to avoid individual lookups
            existing_geometries = session.exec(select(AirfoilGeometry.name)).all()
            existing_geometry_names = set(existing_geometries)
            
            # Get all airfoils that need processing
            statement = select(Airfoil.name, Airfoil.pointcloud)
            airfoils = session.exec(statement).all()
            
            total_airfoils = len(airfoils)
            print(f"Processing {total_airfoils} airfoils in batches of {batch_size}")
            
            # Process in batches
            for i in range(0, total_airfoils, batch_size):
                batch = airfoils[i:i+batch_size]
                geometries_to_update = []
                geometries_to_add = []
                
                for name, pointcloud in batch:
                    if not pointcloud:
                        continue
                        
                    rows = pointcloud.split('\n')
                    rows = [x for x in rows if x.strip()]
                    
                    if not rows:
                        continue
                        
                    points = np.array([np.fromstring(row, dtype=float, sep=' ') for row in rows])
                    
                    # Compute metrics
                    x_coords, thickness, camber = compute_thickness_camber(points)
                    LE_radius = leading_edge_radius(points)
                    TE_angle = trailing_edge_angle(points)
                    chord_length = max(x_coords) - min(x_coords)
                    t_to_c = thickness_to_chord_ratio(thickness, chord_length)
                    aspect_ratio = calculate_aspect_ratio(1, chord_length)
                    max_thickness = max(thickness)
                    max_camber = max(camber)
                    
                    # Calculate normalized chord
                    normalized_chord = np.linspace(0, 1, len(thickness))

                    # Store distribution data as comma-separated strings
                    thickness_dist_str = ",".join(map(str, thickness))
                    camber_dist_str = ",".join(map(str, camber))
                    normalized_chord_str = ",".join(map(str, normalized_chord))
                    
                    # Check if we need to update or add
                    if name in existing_geometry_names:
                        # Fetch the existing record
                        statement = select(AirfoilGeometry).where(AirfoilGeometry.name == name)
                        existing_geometry = session.exec(statement).first()
                        # Update fields
                        existing_geometry.max_thickness = max_thickness
                        existing_geometry.max_camber = max_camber
                        existing_geometry.leading_edge_radius = LE_radius
                        existing_geometry.trailing_edge_angle = TE_angle
                        existing_geometry.chord_length = chord_length
                        existing_geometry.thickness_to_chord_ratio = t_to_c
                        existing_geometry.aspect_ratio = aspect_ratio
                        existing_geometry.thickness_distribution = thickness_dist_str
                        existing_geometry.camber_distribution = camber_dist_str
                        existing_geometry.normalized_chord = normalized_chord_str
                        geometries_to_update.append(existing_geometry)
                    else:
                        # Create new geometry
                        geometry = AirfoilGeometry(
                            name=name,
                            max_thickness=max_thickness,
                            max_camber=max_camber,
                            leading_edge_radius=LE_radius,
                            trailing_edge_angle=TE_angle,
                            chord_length=chord_length,
                            thickness_to_chord_ratio=t_to_c,
                            aspect_ratio=aspect_ratio,
                            thickness_distribution=thickness_dist_str,
                            camber_distribution=camber_dist_str,
                            normalized_chord=normalized_chord_str
                        )
                        geometries_to_add.append(geometry)
                
                # Bulk update and add
                if geometries_to_add:
                    session.add_all(geometries_to_add)
                session.commit()
                
                print(f"Processed batch {i//batch_size + 1}/{(total_airfoils + batch_size - 1)//batch_size}: "
                      f"{len(geometries_to_add)} added, {len(geometries_to_update)} updated")

    def compute_geometry_metrics_parallel(self, num_processes=None, batch_size=100):
        """
        Computes geometry metrics using parallel processing with improved error handling
        and optimized database operations.
        
        Args:
            num_processes (int, optional): Number of processes to use. Defaults to CPU count - 1.
            batch_size (int, optional): Size of batches for database operations. Defaults to 100.
        """
        if num_processes is None:
            num_processes = max(1, multiprocessing.cpu_count() - 1)
            
        with Session(self.engine) as session:
            # Get all airfoils and existing geometries
            print("Fetching airfoils and existing geometry data...")
            
            # Get existing geometries
            existing_geometries = session.exec(select(AirfoilGeometry.name)).all()
            existing_geometry_names = set(existing_geometries)
            
            # Get all airfoils that need processing
            statement = select(Airfoil.name, Airfoil.pointcloud)
            airfoils = session.exec(statement).all()
            
            # Filter out airfoils with no pointcloud
            valid_airfoils = [(name, pc) for name, pc in airfoils if pc]
            
            total_airfoils = len(valid_airfoils)
            print(f"Processing {total_airfoils} airfoils using {num_processes} processes")
            
            # Process airfoils in parallel
            start_time = time.time()
            results = parallel_map(process_airfoil, valid_airfoils, max_workers=num_processes)
            results = [r for r in results if r]  # Filter out None results
            
            processing_time = time.time() - start_time
            print(f"Parallel processing completed in {processing_time:.2f} seconds")
            print(f"Successfully processed {len(results)} out of {total_airfoils} airfoils")
            
            # Update database in batches
            print(f"Updating database in batches of {batch_size}...")
            start_time = time.time()
            
            for i in range(0, len(results), batch_size):
                batch = results[i:i+batch_size]
                geometries_to_update = []
                geometries_to_add = []
                
                for result in batch:
                    name = result.pop('name')
                    
                    if name in existing_geometry_names:
                        # Update existing
                        statement = select(AirfoilGeometry).where(AirfoilGeometry.name == name)
                        existing_geometry = session.exec(statement).first()
                        for key, value in result.items():
                            setattr(existing_geometry, key, value)
                        geometries_to_update.append(existing_geometry)
                    else:
                        # Create new
                        geometry = AirfoilGeometry(name=name, **result)
                        geometries_to_add.append(geometry)
                
                # Bulk update and add
                if geometries_to_add:
                    session.add_all(geometries_to_add)
                session.commit()
                
                print(f"Batch {i//batch_size + 1}/{(len(results) + batch_size - 1)//batch_size}: "
                    f"{len(geometries_to_add)} added, {len(geometries_to_update)} updated")
            
            db_update_time = time.time() - start_time
            print(f"Database update completed in {db_update_time:.2f} seconds")
            print(f"Total execution time: {processing_time + db_update_time:.2f} seconds")
            
            return len(results)

    def find_airfoils_by_geometry(self, parameter, target_value, tolerance, tolerance_type="absolute"):
        """
        Finds airfoils based on a specified geometric parameter, target value, and tolerance.

        Args:
            parameter (str): The geometric parameter to search for (e.g., "max_thickness", "chord_length").
            target_value (float): The target value for the parameter.
            tolerance (float): The tolerance for the search.
            tolerance_type (str): "absolute" or "percentage".
        """
        valid_parameters = ["max_thickness", "max_camber", "leading_edge_radius",
                            "trailing_edge_angle", "chord_length", "thickness_to_chord_ratio", 
                            "aspect_ratio"]

        if parameter not in valid_parameters:
            print(f"Invalid parameter. Choose from: {', '.join(valid_parameters)}")
            return []

        with Session(self.engine) as session:
            if tolerance_type == "absolute":
                lower_bound = target_value - tolerance
                upper_bound = target_value + tolerance
            elif tolerance_type == "percentage":
                lower_bound = target_value * (1 - tolerance / 100.0)
                upper_bound = target_value * (1 + tolerance / 100.0)
            else:
                print("Invalid tolerance_type. Choose 'absolute' or 'percentage'.")
                return []

            # Create a dynamic query using SQLAlchemy expressions
            column = getattr(AirfoilGeometry, parameter)
            statement = select(AirfoilGeometry.name).where(
                (column >= lower_bound) & (column <= upper_bound)
            )
            
            results = session.exec(statement).all()

            airfoil_names = results
            if airfoil_names:
                print(f"Airfoils matching {parameter} = {target_value} ({tolerance} {tolerance_type}):")
                for name in airfoil_names:
                    print(f"- {name}")
                return airfoil_names
            else:
                print(f"No airfoils found matching {parameter} = {target_value} ({tolerance} {tolerance_type}).")
                return []
                
    def find_airfoils_by_multiple_criteria(self, criteria, match_all=True):
        """
        Finds airfoils based on multiple geometric parameters.
        
        Args:
            criteria (list): List of criteria dictionaries, each containing:
                - parameter: The geometric parameter
                - target_value: The target value
                - tolerance: The tolerance
                - tolerance_type: "absolute" or "percentage" (default: "absolute")
            match_all (bool): If True, airfoils must match all criteria. If False, match any criteria.
        
        Returns:
            list: Names of airfoils matching the criteria
        """
        valid_parameters = ["max_thickness", "max_camber", "leading_edge_radius",
                            "trailing_edge_angle", "chord_length", "thickness_to_chord_ratio", 
                            "aspect_ratio"]
        
        # Validate criteria
        for criterion in criteria:
            if "parameter" not in criterion or criterion["parameter"] not in valid_parameters:
                print(f"Invalid parameter in criterion. Choose from: {', '.join(valid_parameters)}")
                return []
            if "target_value" not in criterion or "tolerance" not in criterion:
                print("Each criterion must include 'parameter', 'target_value', and 'tolerance'")
                return []
            if "tolerance_type" not in criterion:
                criterion["tolerance_type"] = "absolute"
        
        with Session(self.engine) as session:
            # Build the query
            statement = select(AirfoilGeometry.name)
            
            # Add conditions based on criteria
            conditions = []
            for criterion in criteria:
                parameter = criterion["parameter"]
                target_value = criterion["target_value"]
                tolerance = criterion["tolerance"]
                tolerance_type = criterion["tolerance_type"]
                
                if tolerance_type == "absolute":
                    lower_bound = target_value - tolerance
                    upper_bound = target_value + tolerance
                elif tolerance_type == "percentage":
                    lower_bound = target_value * (1 - tolerance / 100.0)
                    upper_bound = target_value * (1 + tolerance / 100.0)
                else:
                    print(f"Invalid tolerance_type '{tolerance_type}'. Using 'absolute'.")
                    lower_bound = target_value - tolerance
                    upper_bound = target_value + tolerance
                
                column = getattr(AirfoilGeometry, parameter)
                conditions.append((column >= lower_bound) & (column <= upper_bound))
            
            # Combine conditions based on match_all flag
            if match_all:
                # AND all conditions together
                for condition in conditions:
                    statement = statement.where(condition)
            else:
                # OR all conditions together
                from sqlalchemy import or_
                statement = statement.where(or_(*conditions))
            
            results = session.exec(statement).all()
            
            if results:
                match_type = "all" if match_all else "any"
                print(f"Found {len(results)} airfoils matching {match_type} of the {len(criteria)} criteria:")
                for name in results:
                    print(f"- {name}")
                return results
            else:
                match_type = "all" if match_all else "any"
                print(f"No airfoils found matching {match_type} of the {len(criteria)} criteria.")
                return []
    
    def batch_compute_metrics_for_new_airfoils(self, batch_size=100):
        """
        Computes geometry metrics only for airfoils that don't have geometry data yet.
        This is more efficient when you've added new airfoils to the database.
        """
        with Session(self.engine) as session:
            # Get airfoils that don't have geometry data yet
            subquery = select(AirfoilGeometry.name)
            statement = select(Airfoil.name, Airfoil.pointcloud).where(
                ~Airfoil.name.in_(subquery)
            )
            
            airfoils = session.exec(statement).all()
            
            if not airfoils:
                print("No new airfoils found that need geometry metrics.")
                return
                
            total_airfoils = len(airfoils)
            print(f"Computing geometry metrics for {total_airfoils} new airfoils in batches of {batch_size}")
            
            # Process in batches
            for i in range(0, total_airfoils, batch_size):
                batch = airfoils[i:i+batch_size]
                geometries_to_add = []
                
                for name, pointcloud in batch:
                    if not pointcloud:
                        continue
                        
                    rows = pointcloud.split('\n')
                    rows = [x for x in rows if x.strip()]
                    
                    if not rows:
                        continue
                        
                    points = np.array([np.fromstring(row, dtype=float, sep=' ') for row in rows])
                    
                    # Compute metrics
                    x_coords, thickness, camber = compute_thickness_camber(points)
                    LE_radius = leading_edge_radius(points)
                    TE_angle = trailing_edge_angle(points)
                    chord_length = max(x_coords) - min(x_coords)
                    t_to_c = thickness_to_chord_ratio(thickness, chord_length)
                    aspect_ratio = calculate_aspect_ratio(1, chord_length)
                    max_thickness = max(thickness)
                    max_camber = max(camber)
                    
                    # Calculate normalized chord
                    normalized_chord = np.linspace(0, 1, len(thickness))

                    # Store distribution data as comma-separated strings
                    thickness_dist_str = ",".join(map(str, thickness))
                    camber_dist_str = ",".join(map(str, camber))
                    normalized_chord_str = ",".join(map(str, normalized_chord))
                    
                    # Create new geometry
                    geometry = AirfoilGeometry(
                        name=name,
                        max_thickness=max_thickness,
                        max_camber=max_camber,
                        leading_edge_radius=LE_radius,
                        trailing_edge_angle=TE_angle,
                        chord_length=chord_length,
                        thickness_to_chord_ratio=t_to_c,
                        aspect_ratio=aspect_ratio,
                        thickness_distribution=thickness_dist_str,
                        camber_distribution=camber_dist_str,
                        normalized_chord=normalized_chord_str
                    )
                    geometries_to_add.append(geometry)
                
                # Bulk add
                if geometries_to_add:
                    session.add_all(geometries_to_add)
                    session.commit()
                    print(f"Processed batch {i//batch_size + 1}/{(total_airfoils + batch_size - 1)//batch_size}: "
                          f"Added geometry metrics for {len(geometries_to_add)} airfoils")
    
    def get_similar_airfoils(self, airfoil_name, parameters=None, tolerance_percentage=5.0, limit=10):
        """
        Find airfoils similar to a given airfoil based on geometric parameters.
        
        Args:
            airfoil_name (str): Name of the reference airfoil
            parameters (list): List of parameters to compare (default: all parameters)
            tolerance_percentage (float): Percentage tolerance for similarity
            limit (int): Maximum number of similar airfoils to return
            
        Returns:
            list: Names of similar airfoils, sorted by similarity
        """
        if parameters is None:
            parameters = ["max_thickness", "max_camber", "leading_edge_radius",
                          "trailing_edge_angle", "thickness_to_chord_ratio"]
        
        with Session(self.engine) as session:
            # Get reference airfoil geometry
            statement = select(AirfoilGeometry).where(AirfoilGeometry.name == airfoil_name)
            reference = session.exec(statement).first()
            
            if not reference:
                print(f"Airfoil '{airfoil_name}' not found or has no geometry data.")
                return []
            
            # Get all other airfoils
            statement = select(AirfoilGeometry).where(AirfoilGeometry.name != airfoil_name)
            all_airfoils = session.exec(statement).all()
            
            # Calculate similarity scores
            similarity_scores = []
            for airfoil in all_airfoils:
                score = 0
                valid_params = 0
                
                for param in parameters:
                    ref_value = getattr(reference, param)
                    current_value = getattr(airfoil, param)
                    
                    # Skip if either value is None
                    if ref_value is None or current_value is None:
                        continue
                    
                    # Calculate percentage difference
                    if ref_value != 0:
                        diff_percentage = abs((current_value - ref_value) / ref_value) * 100
                        # Convert difference to similarity (100% = identical, 0% = completely different)
                        param_similarity = max(0, 100 - diff_percentage)
                        score += param_similarity
                        valid_params += 1
                
                # Calculate average similarity if we have valid parameters
                if valid_params > 0:
                    avg_similarity = score / valid_params
                    if avg_similarity >= (100 - tolerance_percentage):
                        similarity_scores.append((airfoil.name, avg_similarity))
            
            # Sort by similarity (highest first) and limit results
            similarity_scores.sort(key=lambda x: x[1], reverse=True)
            top_similar = similarity_scores[:limit]
            
            if top_similar:
                print(f"Top {len(top_similar)} airfoils similar to {airfoil_name}:")
                for name, score in top_similar:
                    print(f"- {name}: {score:.2f}% similarity")
                return [name for name, _ in top_similar]
            else:
                print(f"No airfoils found similar to {airfoil_name} within {tolerance_percentage}% tolerance.")
                return []
