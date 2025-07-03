# analysis/geometry_analyzer.py
import numpy as np
import pandas as pd
import logging
from sqlmodel import Session, select
from typing import List, Optional, Dict, Any, Tuple

from airfoil_database.core.models import Airfoil, AirfoilGeometry
from airfoil_database.utils.helpers import _pointcloud_to_numpy
from airfoil_database.formulas.airfoil.compute_aspect_ratio import calculate_aspect_ratio
from airfoil_database.formulas.airfoil.compute_LE_radius import leading_edge_radius
from airfoil_database.formulas.airfoil.compute_TE_angle import trailing_edge_angle
from airfoil_database.formulas.airfoil.compute_thickness_camber import compute_thickness_camber
from airfoil_database.formulas.airfoil.compute_thickness_to_chord_ratio import thickness_to_chord_ratio

class GeometryAnalyzer:
    def __init__(self, database):
        self.db = database
        self.engine = database.engine
    
    def check_pointcloud_outliers(self, name, threshold=3.0):
        """Checks for outliers in the pointcloud of a given airfoil."""
        with Session(self.engine) as session:
            statement = select(Airfoil.pointcloud).where(Airfoil.name == name)
            result = session.exec(statement).first()

            if result:
                pointcloud_np = _pointcloud_to_numpy(result)
                if pointcloud_np.size == 0:
                    return False, "Empty pointcloud"
                x = pointcloud_np[:, 0]
                y = pointcloud_np[:, 1]

                # Calculate z-scores for x and y coordinates
                z_x = np.abs((x - np.mean(x)) / np.std(x))
                z_y = np.abs((y - np.mean(y)) / np.std(y))

                # Identify outliers based on z-score threshold
                outlier_indices = np.where((z_x > threshold) | (z_y > threshold))[0]

                if len(outlier_indices) > 0:
                    outlier_points = pointcloud_np[outlier_indices]
                    return True, outlier_points
                else:
                    return False, None
            else:
                return False, "Airfoil not found"

    def check_all_pointcloud_outliers(self, threshold=3.0):
        """Checks all airfoils in the database for outliers."""
        outliers_found = {}

        with Session(self.engine) as session:
            statement = select(Airfoil.name)
            airfoils = session.exec(statement).all()

            for name in airfoils:
                has_outliers, outliers = self.check_pointcloud_outliers(name, threshold)
                if has_outliers:
                    outliers_found[name] = outliers

        if outliers_found:
            print("Airfoils with outliers:")
            for name, outliers in outliers_found.items():
                print(f"- {name}:")
                for point in outliers:
                    print(f"  {point}")
        else:
            print("No outliers found in any airfoils.")
        return outliers_found

    def compute_geometry_metrics(self):
        """Computes and stores geometry metrics for all airfoils in the database."""
        with Session(self.engine) as session:
            statement = select(Airfoil.name, Airfoil.pointcloud)
            airfoils = session.exec(statement).all()

            for name, pointcloud in airfoils:
                rows = pointcloud.split('\n')
                rows = [x for x in rows if x.strip()]
                points = np.array([np.fromstring(row, dtype=float, sep=' ') for row in rows])
                
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

                # Check if geometry entry exists
                statement = select(AirfoilGeometry).where(AirfoilGeometry.name == name)
                existing_geometry = session.exec(statement).first()
                
                if existing_geometry:
                    # Update existing geometry
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
                    session.add(geometry)
                
                session.commit()
                print(f"Geometry metrics computed and stored for {name}")

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
