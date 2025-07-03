import os
import csv
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import concurrent.futures
import threading
import seaborn as sns
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
import pandas as pd
import logging
from typing import List, Optional, Dict, Any, Tuple
from sqlmodel import Field, Session, SQLModel, create_engine, select, Column, JSON, String, Float

from airfoil_database.plotting.plot_histogram import plot_histogram
from airfoil_database.plotting.plot_bar_chart import plot_horizontal_bar_chart

from airfoil_database.formulas.airfoil.compute_aspect_ratio import calculate_aspect_ratio
from airfoil_database.formulas.airfoil.compute_LE_radius import leading_edge_radius
from airfoil_database.formulas.airfoil.compute_TE_angle import trailing_edge_angle
from airfoil_database.formulas.airfoil.compute_thickness_camber import compute_thickness_camber
from airfoil_database.formulas.airfoil.compute_thickness_to_chord_ratio import thickness_to_chord_ratio

from airfoil_database.xfoil.fix_airfoil_data import normalize_pointcloud
from airfoil_database.xfoil.fix_airfoil_pointcloud_v2 import *
from airfoil_database.xfoil.interpolate_points import interpolate_points
from airfoil_database.xfoil.calculate_distance import calculate_min_distance_sum
from airfoil_database.xfoil.fix_point_cloud import *

from airfoil_database.classes.XFoilRunner import XFoilRunner
from airfoil_database.classes.AirfoilSeries import AirfoilSeries

from airfoil_database.utilities.get_files_starting_with import get_files_starting_with


# Define SQLModel models
class Airfoil(SQLModel, table=True):
    __tablename__ = "airfoils"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    description: Optional[str] = None
    pointcloud: Optional[str] = None
    airfoil_series: Optional[str] = None
    source: Optional[str] = None


class AeroCoeff(SQLModel, table=True):
    __tablename__ = "aero_coeffs"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True, foreign_key="airfoils.name")
    reynolds_number: float
    mach: float
    ncrit: float
    alpha: float
    cl: Optional[float] = None
    cd: Optional[float] = None
    cm: Optional[float] = None
    
    class Config:
        unique_together = [("name", "reynolds_number", "mach", "alpha")]


class AirfoilGeometry(SQLModel, table=True):
    __tablename__ = "airfoil_geometry"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True, foreign_key="airfoils.name")
    max_thickness: Optional[float] = None
    max_camber: Optional[float] = None
    chord_length: Optional[float] = None
    aspect_ratio: Optional[float] = None
    leading_edge_radius: Optional[float] = None
    trailing_edge_angle: Optional[float] = None
    thickness_to_chord_ratio: Optional[float] = None
    thickness_distribution: Optional[str] = None
    camber_distribution: Optional[str] = None
    normalized_chord: Optional[str] = None


class AirfoilDatabase:
    def __init__(self, db_name="airfoil_data.db", db_dir="."):
        self.db_path = os.path.join(db_dir, db_name)  # Path to the database
        os.makedirs(db_dir, exist_ok=True)  # Create directory if it doesn't exist.
        
        # Create SQLAlchemy engine
        self.engine = create_engine(f"sqlite:///{self.db_path}", echo=False)
        
        # Create tables if they don't exist
        SQLModel.metadata.create_all(self.engine)
        
        logging.info(f"AirfoilDatabase initialized with db: {self.db_path}")

    def store_airfoil_data(self, name, description, pointcloud, airfoil_series, source, overwrite=False):
        try:
            with Session(self.engine) as session:
                # Check if airfoil exists
                statement = select(Airfoil).where(Airfoil.name == name)
                existing_airfoil = session.exec(statement).first()
                
                if existing_airfoil and not overwrite:
                    print(f"Airfoil {name} already exists in the database. Use overwrite=True to update.")
                    return
                
                if existing_airfoil and overwrite:
                    # Update existing airfoil
                    existing_airfoil.description = description
                    existing_airfoil.pointcloud = pointcloud
                    existing_airfoil.airfoil_series = airfoil_series.value
                    existing_airfoil.source = source
                    session.add(existing_airfoil)
                    session.commit()
                    print(f"Updated: {name} in database.")
                else:
                    # Create new airfoil
                    airfoil = Airfoil(
                        name=name,
                        description=description,
                        pointcloud=pointcloud,
                        airfoil_series=airfoil_series.value,
                        source=source
                    )
                    session.add(airfoil)
                    session.commit()
                    print(f"Stored: {name} in database.")
        except Exception as e:
            print(f"Error storing airfoil data: {e}")
    
    def store_bulk_airfoil_data(self, data_list, overwrite=False):
        """
        Stores multiple airfoil data records in the database.

        Args:
            data_list (list): A list of dictionaries. Each dictionary should contain
                              keys: 'name', 'description', 'pointcloud',
                                    'airfoil_series', 'source'.
                              Example: [{'name': 'naca0012', 'description': '...', ...}, ...]
            overwrite (bool): If True, existing entries with the same name will be replaced.
                              If False, entries with names already in the DB will be skipped.

        Returns:
            int: The number of rows successfully inserted or replaced.
        """
        if not data_list:
            logging.info("No data provided for bulk storage.")
            return 0

        inserted_count = 0
        names_to_check = {data['name'] for data in data_list if 'name' in data}  # Get unique names from input

        try:
            with Session(self.engine) as session:
                if not overwrite:
                    # Find names that already exist if we are not overwriting
                    statement = select(Airfoil.name).where(Airfoil.name.in_(names_to_check))
                    existing_names = {row.name for row in session.exec(statement)}
                    logging.info(f"Found {len(existing_names)} existing airfoils matching input names. Skipping them.")
                else:
                    existing_names = set()  # Overwrite mode, don't skip any

                airfoils_to_add = []
                
                for data in data_list:
                    name = data.get('name')
                    if not name:
                        logging.warning("Skipping record due to missing 'name'.")
                        continue
                    
                    if name in existing_names and not overwrite:
                        continue  # Skip if not overwriting and name exists

                    # Check if airfoil exists for overwrite
                    if overwrite:
                        statement = select(Airfoil).where(Airfoil.name == name)
                        existing_airfoil = session.exec(statement).first()
                        
                        if existing_airfoil:
                            # Update existing airfoil
                            existing_airfoil.description = data.get('description', '')
                            existing_airfoil.pointcloud = data.get('pointcloud', '')
                            existing_airfoil.airfoil_series = data.get('airfoil_series', 'UNKNOWN')
                            existing_airfoil.source = data.get('source', '')
                            session.add(existing_airfoil)
                            inserted_count += 1
                            continue

                    # Create new airfoil
                    airfoil = Airfoil(
                        name=name,
                        description=data.get('description', ''),
                        pointcloud=data.get('pointcloud', ''),
                        airfoil_series=data.get('airfoil_series', 'UNKNOWN'),
                        source=data.get('source', '')
                    )
                    airfoils_to_add.append(airfoil)
                    inserted_count += 1

                if airfoils_to_add:
                    session.add_all(airfoils_to_add)
                    session.commit()
                
                logging.info(f"Bulk insert/replace finished. Affected rows: {inserted_count}")
                return inserted_count

        except Exception as e:
            logging.error(f"Error during bulk insert: {e}", exc_info=True)
            return 0

    def add_airfoils_from_csv(self, csv_file, overwrite=False):
        """Adds airfoils from a CSV file."""
        try:
            with open(csv_file, 'r', newline='', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                headers = reader.fieldnames
                if not headers:
                    print("CSV file is empty or has no headers.")
                    return

                with Session(self.engine) as session:
                    for row in reader:
                        name = row.get('name')  # Assuming 'name' column exists
                        if not name:
                            print("Warning: Skipping row with missing 'name'.")
                            continue
                        
                        # Check if airfoil exists
                        statement = select(Airfoil).where(Airfoil.name == name)
                        existing_airfoil = session.exec(statement).first()
                        
                        if existing_airfoil and overwrite:
                            # Delete existing airfoil data if overwrite is True
                            self._delete_airfoil_data(name, session)
                        
                        # Create airfoil data dictionary from valid columns
                        airfoil_data = {}
                        for header in headers:
                            if header in ['name', 'description', 'pointcloud', 'airfoil_series', 'source']:
                                airfoil_data[header] = row.get(header)
                        
                        # Create and add airfoil
                        airfoil = Airfoil(**airfoil_data)
                        session.add(airfoil)
                        
                        try:
                            session.commit()
                            print(f"Added/Updated: {name} from CSV.")
                        except Exception as e:
                            session.rollback()
                            print(f"Error adding {name}: {e}")

        except FileNotFoundError:
            print(f"File not found: {csv_file}")
        except Exception as e:
            print(f"An error occurred: {e}")

    def add_airfoils_from_json(self, json_file, overwrite=False):
        """Adds airfoils from a JSON file."""
        try:
            with open(json_file, 'r', encoding='utf-8') as file:
                data = json.load(file)

            with Session(self.engine) as session:
                for name, airfoil_data in data.items():
                    # Check if airfoil exists
                    statement = select(Airfoil).where(Airfoil.name == name)
                    existing_airfoil = session.exec(statement).first()
                    
                    if existing_airfoil and overwrite:
                        # Delete existing airfoil data if overwrite is True
                        self._delete_airfoil_data(name, session)
                    
                    # Create airfoil
                    airfoil = Airfoil(
                        name=name,
                        description=airfoil_data.get('description'),
                        pointcloud=airfoil_data.get('pointcloud'),
                        airfoil_series=airfoil_data.get('airfoil_series'),
                        source=airfoil_data.get('source')
                    )
                    session.add(airfoil)
                    
                    try:
                        session.commit()
                        print(f"Added/Updated: {name} from JSON.")
                    except Exception as e:
                        session.rollback()
                        print(f"Error adding {name}: {e}")

        except FileNotFoundError:
            print(f"File not found: {json_file}")
        except json.JSONDecodeError:
            print(f"Invalid JSON format in {json_file}")
        except Exception as e:
            print(f"An error occurred: {e}")
    
    def update_airfoil_info(self, old_name, new_name, description, series, source):
        """Updates airfoil info in all related tables."""
        try:
            with Session(self.engine) as session:
                # Update airfoil
                statement = select(Airfoil).where(Airfoil.name == old_name)
                airfoil = session.exec(statement).first()
                
                if airfoil:
                    # Update airfoil fields
                    airfoil.name = new_name
                    airfoil.description = description
                    airfoil.airfoil_series = series
                    airfoil.source = source
                    
                    # Update aero_coeffs
                    statement = select(AeroCoeff).where(AeroCoeff.name == old_name)
                    aero_coeffs = session.exec(statement).all()
                    for coeff in aero_coeffs:
                        coeff.name = new_name
                    
                    # Update airfoil_geometry
                    statement = select(AirfoilGeometry).where(AirfoilGeometry.name == old_name)
                    geometry = session.exec(statement).first()
                    if geometry:
                        geometry.name = new_name
                    
                    session.commit()
                    print(f"Updated airfoil info for {old_name} to {new_name}.")
                else:
                    print(f"Airfoil {old_name} not found.")

        except Exception as e:
            print(f"Error updating airfoil info: {e}")

    def update_airfoil_series(self):
        with Session(self.engine) as session:
            statement = select(Airfoil)
            airfoils = session.exec(statement).all()
            
            for airfoil in airfoils:
                airfoil_series_curr = AirfoilSeries.from_string(airfoil.airfoil_series)
                if airfoil_series_curr == AirfoilSeries.OTHER:
                    airfoil_series_curr = AirfoilSeries.identify_airfoil_series(airfoil.name)
                    if airfoil_series_curr == AirfoilSeries.OTHER:
                        airfoil_series_curr = AirfoilSeries.identify_airfoil_series(airfoil.description or '')
                        #TODO: Add more logic to get the airfoil series
                    airfoil.airfoil_series = airfoil_series_curr.value
            
            session.commit()

    def _delete_airfoil_data(self, name, session):
        """Deletes all data associated with an airfoil."""
        # Delete from aero_coeffs
        statement = select(AeroCoeff).where(AeroCoeff.name == name)
        aero_coeffs = session.exec(statement).all()
        for coeff in aero_coeffs:
            session.delete(coeff)
        
        # Delete from airfoil_geometry
        statement = select(AirfoilGeometry).where(AirfoilGeometry.name == name)
        geometry = session.exec(statement).first()
        if geometry:
            session.delete(geometry)
        
        # Delete from airfoils
        statement = select(Airfoil).where(Airfoil.name == name)
        airfoil = session.exec(statement).first()
        if airfoil:
            session.delete(airfoil)
        
        session.commit()
        print(f"Deleted existing data for {name}.")

    def get_airfoil_data(self, name):
        """Retrieves airfoil data including description, pointcloud, series, and source."""
        try:
            with Session(self.engine) as session:
                statement = select(Airfoil).where(Airfoil.name == name)
                airfoil = session.exec(statement).first()
                
                if airfoil:
                    logging.debug(f"Retrieved data for airfoil: {name}")
                    return (airfoil.description, airfoil.pointcloud, airfoil.airfoil_series, airfoil.source)
                else:
                    logging.warning(f"Airfoil '{name}' not found in database.")
                    return None
        except Exception as e:
            logging.error(f"Error retrieving airfoil data for {name}: {e}")
            return None
    
    def get_airfoil_dataframe(self):
        """Returns a Pandas DataFrame with airfoil names, series, and number of points."""
        with Session(self.engine) as session:
            statement = select(Airfoil.name, Airfoil.airfoil_series, Airfoil.pointcloud)
            results = session.exec(statement).all()

        data = []
        for name, series, pointcloud in results:
            num_points = len(pointcloud.strip().split('\n')) if pointcloud else 0
            data.append({
                'Name': name,
                'Series': series,
                'Num_Points': num_points
            })

        return pd.DataFrame(data)

    def get_airfoil_geometry_dataframe(self):
        """Retrieves airfoil geometry data from the database and returns it as a Pandas DataFrame."""
        try:
            with Session(self.engine) as session:
                statement = select(AirfoilGeometry)
                results = session.exec(statement).all()
                
                if not results:
                    return pd.DataFrame()  # Return an empty DataFrame if no data found
                
                # Convert SQLModel objects to dictionaries
                data = [
                    {
                        "id": geom.id,
                        "name": geom.name,
                        "max_thickness": geom.max_thickness,
                        "max_camber": geom.max_camber,
                        "chord_length": geom.chord_length,
                        "aspect_ratio": geom.aspect_ratio,
                        "leading_edge_radius": geom.leading_edge_radius,
                        "trailing_edge_angle": geom.trailing_edge_angle,
                        "thickness_to_chord_ratio": geom.thickness_to_chord_ratio
                    }
                    for geom in results
                ]
                
                return pd.DataFrame(data)

        except Exception as e:
            print(f"An error occurred: {e}")
            return pd.DataFrame()  # Return empty dataframe on error.
    
    def _pointcloud_to_numpy(self, pointcloud_str):
        """Converts a pointcloud string to a NumPy array."""
        if not pointcloud_str:
            return np.array([])
        rows = pointcloud_str.strip().split('\n')
        rows = [x.strip() for x in rows if x.strip()]
        try:
            return np.array([np.fromstring(row, dtype=float, sep=' ') for row in rows])
        except ValueError:
            return np.array([])
    
    def check_pointcloud_outliers(self, name, threshold=3.0):
        """Checks for outliers in the pointcloud of a given airfoil."""
        with Session(self.engine) as session:
            statement = select(Airfoil.pointcloud).where(Airfoil.name == name)
            result = session.exec(statement).first()

            if result:
                pointcloud_np = self._pointcloud_to_numpy(result)
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

    def fix_all_airfoils(self, config=None):
        """
        Attempts to fix the pointclouds for all airfoils in the database.

        Args:
            config (dict, optional): Configuration for the pointcloud_fixer.
                                     Uses defaults if None.
        """
        logging.info("Starting pointcloud fixing process for all airfoils.")
        airfoil_names = []
        try:
            with Session(self.engine) as session:
                # Select names of airfoils that have some pointcloud data
                statement = select(Airfoil.name).where(
                    (Airfoil.pointcloud != None) & (Airfoil.pointcloud != "")
                ).order_by(Airfoil.name)
                airfoil_names = session.exec(statement).all()
        except Exception as e:
            logging.error(f"Failed to retrieve airfoil names for fixing: {e}")
            return

        if not airfoil_names:
            logging.warning("No airfoils with pointcloud data found to fix.")
            return

        logging.info(f"Found {len(airfoil_names)} airfoils with pointcloud data to check/fix.")
        success_count = 0
        fail_count = 0

        for i, name in enumerate(airfoil_names):
            logging.info(f"--- Processing airfoil {i+1}/{len(airfoil_names)}: {name} ---")
            fixed_array = self.fix_airfoil_pointcloud(name, config=config, store_result=True)
            if fixed_array is not None:
                success_count += 1
            else:
                fail_count += 1

        logging.info("Finished fixing all airfoils.")
        logging.info(f"Successfully processed/fixed: {success_count}")
        logging.info(f"Failed/Skipped: {fail_count}")
    
    def fix_airfoil_pointcloud(self, name, config=None, store_result=True):
        """
        Retrieves, fixes, and optionally updates the pointcloud for a single airfoil.

        Args:
            name (str): The name of the airfoil.
            config (dict, optional): Configuration for the pointcloud_fixer.
                                     Uses defaults if None.
            store_result (bool): If True, updates the database with the fixed pointcloud.

        Returns:
            np.ndarray or None: The fixed point cloud array, or None if fixing failed
                                or the original data was invalid.
        """
        logging.info(f"Attempting to fix pointcloud for airfoil: {name}")
        airfoil_data = self.get_airfoil_data(name)
        if not airfoil_data or not airfoil_data[1]:  # Check data exists and pointcloud string is not empty
            logging.error(f"Cannot fix: Airfoil '{name}' not found or has no pointcloud data.")
            return None

        description, pointcloud_str, airfoil_series, source = airfoil_data

        airfoil_series = AirfoilSeries(airfoil_series)

        # Parse the pointcloud string
        points_array = parse_pointcloud_string(pointcloud_str)
        if points_array is None:
            logging.error(f"Could not parse original pointcloud string for {name}.")
            return None  # Parsing failed

        # Get the fixer configuration
        fixer_config = config if config is not None else DEFAULT_FIXER_CONFIG

        # Call the fixing function
        fixed_points_array = fix_pointcloud(points_array, fixer_config)

        if fixed_points_array is None:
            logging.error(f"Pointcloud fixing failed for {name}. No update performed.")
            return None
        else:
            logging.info(f"Pointcloud fixing successful for {name}.")
            if store_result:
                # Format the fixed array back to string with desired precision
                fixed_pointcloud_str = format_pointcloud_array(fixed_points_array, fixer_config.get("precision", 10))
                # Store the updated data
                self.store_airfoil_data(name, description, fixed_pointcloud_str, airfoil_series, source, overwrite=True)
                logging.info(f"Stored fixed pointcloud for {name} in database.")
            return fixed_points_array  # Return the fixed array

    def output_pointcloud_to_file(self, airfoil_name, file_path):
        """
        Outputs the point cloud of an airfoil to a text file.

        Args:
            airfoil_name (str): The name of the airfoil.
            file_path (str): The path to the output text file.
        """
        try:
            with Session(self.engine) as session:
                statement = select(Airfoil.pointcloud).where(Airfoil.name == airfoil_name)
                result = session.exec(statement).first()

                if result:
                    pointcloud_str = result
                    with open(file_path, 'w') as file:
                        file.write(pointcloud_str)
                else:
                    print(f"Airfoil '{airfoil_name}' not found or point cloud is empty.")

        except Exception as e:
            print(f"Error outputting point cloud: {e}")

    def plot_airfoil_series_pie(self, output_dir=None, output_name=None):
        """Fetches airfoil series data from the database and plots a pie chart."""

        # Connect to database and retrieve all airfoil_series values
        with Session(self.engine) as session:
            statement = select(Airfoil.airfoil_series)
            series_list = [row for row in session.exec(statement) if row]

        if not series_list:
            print("No airfoil series data found in the database.")
            return

        # Count occurrences of each airfoil series
        series_counts = Counter(series_list)

        # Extract labels and counts for pie chart
        labels = list(series_counts.keys())
        counts = list(series_counts.values())

        # Plot pie chart
        plt.figure(figsize=(8, 8))  # Adjust figure size as needed
        plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140)
        plt.title("Airfoil Series Distribution")

        # Save the pie chart
        if output_dir is not None or output_name is not None:
            plt.savefig(os.path.join(output_dir if output_dir is not None else '', output_name if output_name is not None else 'airfoil_series_pie.png'))
        else:
            plt.show()  # Display the pie chart
    
    def plot_airfoil_series_horizontal_bar(self, **kwargs):
        """Fetches airfoil series data from the database and plots a horizontal bar chart."""

        # Connect to database and retrieve all airfoil_series values
        with Session(self.engine) as session:
            statement = select(Airfoil.airfoil_series)
            series_list = [row for row in session.exec(statement) if row]
        
        if not series_list:
            print("No airfoil series data found in the database.")
            return

        # Count occurrences of each airfoil series
        series_counts = Counter(series_list)

        # Extract labels and counts for pie chart
        labels = list(series_counts.keys())
        counts = list(series_counts.values())

        plot_horizontal_bar_chart(counts, labels, **kwargs)

    def plot_airfoil(self, 
                     name, 
                     ax=None,
                     output_dir=None, 
                     output_name=None):
        """Plots the airfoil using its point cloud data, with markers for individual points."""
        data = self.get_airfoil_data(name)
        if data:
            description, pointcloud_str, series, source = data
            pointcloud_np = self._pointcloud_to_numpy(pointcloud_str)
            x = pointcloud_np[:,0]
            y = pointcloud_np[:,1]

            if ax is None:
                fig, ax = plt.subplots()

            ax.plot(x, y, label=f"{name} : {len(x)}", linestyle='-', marker='o', markersize=4)
            ax.set_title(f"Airfoil: {name}")
            ax.set_xlabel("X Coordinate")
            ax.set_ylabel("Y Coordinate")
            ax.grid(True)
            ax.axis("equal")
            ax.legend(loc='upper left')
            if output_dir is not None:
                if output_name is None:
                    output_name = name + '.png'
                plt.savefig(os.path.join(output_dir, output_name))
            else:
                return ax
        else:
            print(f"Airfoil {name} not found in the database.")
    
    def plot_multiple_airfoils(self, 
                            names, 
                            ax=None,
                            output_dir=None, 
                            output_name=None):
        """Plots multiple airfoils on the same figure, optionally on a provided axes."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 7))  # Create a new figure if ax is not provided

        for name in names:
            data = self.get_airfoil_data(name)
            if data:
                description, pointcloud_str, series, source = data
                try:
                    points = [line.split() for line in pointcloud_str.strip().split('\n')]
                    x = [float(p[0]) for p in points if len(p) == 2]
                    y = [float(p[1]) for p in points if len(p) == 2]

                    if x and y:
                        ax.plot(x, y, label=name, linestyle='-', marker='o', markersize=3)  # Markers added
                    else:
                        print(f"No valid point cloud data found for {name}")

                except (ValueError, IndexError) as e:
                    print(f"Error parsing point cloud data for {name}: {e}")
            else:
                print(f"Airfoil {name} not found in the database.")

        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_title("Airfoil Comparison")
        ax.grid(True)
        ax.axis('equal')
        ax.legend()

        if output_dir is None:
            if ax is None:  # If no ax was passed, show the plot.
                plt.show()
        else:
            if output_name is None:
                output_name = ' vs '.join(names) + '.png'
            plt.savefig(os.path.join(output_dir, output_name))

        return ax  # Return the ax object.
    
    def add_airfoil_to_plot(self, airfoil_name, ax, linestyle='-', marker='o', markersize=3, label=None):
        """Adds an airfoil's point cloud to a Matplotlib plot."""
        try:
            with Session(self.engine) as session:
                statement = select(Airfoil.pointcloud).where(Airfoil.name == airfoil_name)
                result = session.exec(statement).first()

                if result:
                    pointcloud_str = result
                    points = [line.split() for line in pointcloud_str.strip().split('\n')]
                    points = np.array([[float(p[0]), float(p[1])] for p in points if len(p) == 2])
                    if label:  # check if label exists, if not, then use airfoil name.
                        ax.plot(points[:, 0], points[:, 1], linestyle=linestyle, marker=marker, markersize=markersize, label=label)
                    else:
                        ax.plot(points[:, 0], points[:, 1], linestyle=linestyle, marker=marker, markersize=markersize, label=airfoil_name)
                else:
                    print(f"Airfoil '{airfoil_name}' not found.")

        except Exception as e:
            print(f"Error plotting airfoil: {e}")
    
    def find_best_matching_airfoils(self, input_pointcloud_str, num_matches=3):
        """
        Compares an input point cloud to the airfoils in the database and returns the best matches.
        """
        input_points = [line.split() for line in input_pointcloud_str.strip().split('\n')]
        input_points = np.array([[float(p[0]), float(p[1])] for p in input_points if len(p) == 2])
        normalized_input_points = normalize_pointcloud(input_points)
        if len(normalized_input_points) == 0:
            return []

        interpolated_input_points = interpolate_points(normalized_input_points)

        matches = []
        with Session(self.engine) as session:
            statement = select(Airfoil.name, Airfoil.pointcloud)
            airfoils = session.exec(statement).all()

            for name, db_pointcloud_str in airfoils:
                db_points = [line.split() for line in db_pointcloud_str.strip().split('\n')]
                db_points = np.array([[float(p[0]), float(p[1])] for p in db_points if len(p) == 2])
                normalized_db_points = normalize_pointcloud(db_points)
                if len(normalized_db_points) == 0:
                    continue
                    
                interpolated_db_points = interpolate_points(normalized_db_points)

                if np.shape(interpolated_input_points)[0] == np.shape(interpolated_db_points)[0]:
                    distance = calculate_min_distance_sum(interpolated_input_points, interpolated_db_points)
                    matches.append((name, distance))

        matches.sort(key=lambda x: x[1])  # Sort by distance
        return matches[:num_matches]  # Return the top matches

    def compute_geometry_metrics(self):
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
    
    def plot_leading_edge_radius(self, parameter="chord_length"):
        """Plots leading-edge radius against a specified parameter."""
        with Session(self.engine) as session:
            # Create dynamic query to select the needed columns
            column = getattr(AirfoilGeometry, parameter)
            statement = select(AirfoilGeometry.name, AirfoilGeometry.leading_edge_radius, column)
            results = session.exec(statement).all()

        if results:
            names, radii, params = zip(*[(r[0], r[1], r[2]) for r in results])
            plt.figure(figsize=(8, 6))
            plt.scatter(params, radii)
            plt.xlabel(parameter)
            plt.ylabel("Leading Edge Radius")
            plt.title("Leading Edge Radius vs. " + parameter)
            plt.grid(True)
            plt.show()

    def plot_trailing_edge_angle(self, parameter="chord_length"):
        """Plots trailing-edge angle against a specified parameter."""
        with Session(self.engine) as session:
            # Create dynamic query to select the needed columns
            column = getattr(AirfoilGeometry, parameter)
            statement = select(AirfoilGeometry.name, AirfoilGeometry.trailing_edge_angle, column)
            results = session.exec(statement).all()

        if results:
            names, angles, params = zip(*[(r[0], r[1], r[2]) for r in results])
            plt.figure(figsize=(8, 6))
            plt.scatter(params, angles)
            plt.xlabel(parameter)
            plt.ylabel("Trailing Edge Angle")
            plt.title("Trailing Edge Angle vs. " + parameter)
            plt.grid(True)
            plt.show()

    def plot_geometry_correlations(self):
        """Plots correlations between geometric parameters using a heatmap."""
        with Session(self.engine) as session:
            statement = select(
                AirfoilGeometry.max_thickness, 
                AirfoilGeometry.max_camber, 
                AirfoilGeometry.leading_edge_radius, 
                AirfoilGeometry.trailing_edge_angle, 
                AirfoilGeometry.chord_length, 
                AirfoilGeometry.thickness_to_chord_ratio, 
                AirfoilGeometry.aspect_ratio
            )
            results = session.exec(statement).all()

        if results:
            # Convert results to a DataFrame
            columns = ["max_thickness", "max_camber", "leading_edge_radius", "trailing_edge_angle", 
                       "chord_length", "thickness_to_chord_ratio", "aspect_ratio"]
            df = pd.DataFrame(results, columns=columns)
            
            correlation_matrix = df.corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
            plt.title("Geometric Parameter Correlations")
            plt.show()
    
    def store_aero_coeffs(self, name, reynolds_number, mach, ncrit, alpha, cl, cd, cm):
        """Stores a single row of aerodynamic coefficient data."""
        try:
            with Session(self.engine) as session:
                # Check if record already exists
                statement = select(AeroCoeff).where(
                    (AeroCoeff.name == name) & 
                    (AeroCoeff.reynolds_number == reynolds_number) & 
                    (AeroCoeff.mach == mach) & 
                    (AeroCoeff.alpha == alpha)
                )
                existing_coeff = session.exec(statement).first()
                
                if existing_coeff:
                    # Update existing record
                    existing_coeff.ncrit = ncrit
                    existing_coeff.cl = cl
                    existing_coeff.cd = cd
                    existing_coeff.cm = cm
                    session.add(existing_coeff)
                else:
                    # Create new record
                    aero_coeff = AeroCoeff(
                        name=name,
                        reynolds_number=reynolds_number,
                        mach=mach,
                        ncrit=ncrit,
                        alpha=alpha,
                        cl=cl,
                        cd=cd,
                        cm=cm
                    )
                    session.add(aero_coeff)
                
                session.commit()
        except Exception as e:
            logging.error(f"Error storing aero coeffs for {name} Re={reynolds_number} M={mach} Ncrit={ncrit} A={alpha}: {e}")

    def run_airfoil_through_xfoil(self, airfoil_name, reynolds_list, mach_list, alpha_list, ncrit_list, xfoil_path=None, max_workers=None):
        """
        Runs a single airfoil through XFoil analysis using the XFoilRunner.

        Args:
            airfoil_name (str): The name of the airfoil to run.
            reynolds_list (list): List of Reynolds numbers.
            mach_list (list): List of Mach numbers.
            alpha_list (list): List of angles of attack.
            ncrit_list (list): List of transition criteria.
            xfoil_path (str, optional): Path to the XFoil executable. Defaults to None.
            max_workers (int, optional): Max parallel workers for conditions. Defaults to None (os.cpu_count()).
        """
        logging.info(f"Initiating XFoil run for airfoil: {airfoil_name}")
        try:
            # Instantiate the runner, passing self (the database instance)
            runner = XFoilRunner(database=self, xfoil_executable=xfoil_path)
            # Run the analysis in parallel across conditions
            runner.run_analysis_parallel(
                airfoil_name=airfoil_name,
                reynolds_list=reynolds_list,
                mach_list=mach_list,
                alpha_list=alpha_list,
                ncrit_list=ncrit_list,
                max_workers=max_workers
            )
            logging.info(f"Completed XFoil run for airfoil: {airfoil_name}")
        except FileNotFoundError:
            # Raised by the runner if the executable is not found
            logging.error(f"XFoil executable not found via path '{xfoil_path if xfoil_path else 'default'}'. Aborting run for {airfoil_name}.")
        except Exception as e:
            logging.error(f"An error occurred during XFoil run for {airfoil_name}: {e}", exc_info=True)

    def run_all_airfoils(self, reynolds_list, mach_list, alpha_list, ncrit_list, xfoil_path=None, max_workers=None):
        """
        Runs XFoil analysis for all airfoils currently in the database that have pointcloud data.

        Args:
            reynolds_list (list): List of Reynolds numbers.
            mach_list (list): List of Mach numbers.
            alpha_list (list): List of angles of attack.
            ncrit_list (list): List of transition criteria.
            xfoil_path (str, optional): Path to the XFoil executable. Defaults to None.
            max_workers (int, optional): Max parallel workers for conditions *per airfoil*.
                                         Defaults to None (os.cpu_count()).
        """
        logging.info("Starting XFoil analysis for all airfoils in the database.")
        airfoil_names = []
        try:
            with Session(self.engine) as session:
                # Fetch only names that have pointcloud data
                statement = select(Airfoil.name).where(
                    (Airfoil.pointcloud != None) & (Airfoil.pointcloud != "")
                ).order_by(Airfoil.name)
                airfoil_names = session.exec(statement).all()
        except Exception as e:
            logging.error(f"Failed to retrieve airfoil names from database: {e}")
            return  # Cannot proceed without airfoil names

        if not airfoil_names:
            logging.warning("No airfoils with pointcloud data found in the database to run.")
            return

        logging.info(f"Found {len(airfoil_names)} airfoils to process.")

        # Run analysis for each airfoil serially.
        for i, name in enumerate(airfoil_names):
            logging.info(f"--- Processing airfoil {i+1}/{len(airfoil_names)}: {name} ---")
            # Check if airfoil still exists
            if self.get_airfoil_data(name):
                self.run_airfoil_through_xfoil(
                    airfoil_name=name,
                    reynolds_list=reynolds_list,
                    mach_list=mach_list,
                    alpha_list=alpha_list,
                    ncrit_list=ncrit_list,
                    xfoil_path=xfoil_path,
                    max_workers=max_workers
                )
            else:
                logging.warning(f"Skipping airfoil {name} as it seems to have been removed or lacks data.")
            logging.info(f"--- Finished processing airfoil: {name} ---")

        logging.info("Completed XFoil analysis for all airfoils.")
    
    def get_aero_coeffs(self, name, Re=None, Mach=None):
        with Session(self.engine) as session:
            # Start building the query
            statement = select(AeroCoeff).where(AeroCoeff.name == name)
            
            # Add optional filters
            if Re is not None:
                statement = statement.where(AeroCoeff.reynolds_number == Re)
            if Mach is not None:
                statement = statement.where(AeroCoeff.mach == Mach)
            
            # Execute the query and return the results
            results = session.exec(statement).all()
            return results

    def find_airfoils_by_xfoil_results(self, parameter, target_value, tolerance, tolerance_type="absolute"):
        """
        Finds airfoils based on XFOIL results.

        Args:
            parameter (str): The XFOIL result parameter (reynolds_number, alpha, cl, cd, cm, mach, ncrit).
            target_value (float): The target value for the parameter.
            tolerance (float): The tolerance for the search.
            tolerance_type (str): "absolute" or "percentage".
        """
        valid_parameters = ["reynolds_number", "alpha", "mach", "ncrit", "cl", "cd", "cm"]

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
            column = getattr(AeroCoeff, parameter)
            statement = select(AeroCoeff.name).where(
                (column >= lower_bound) & (column <= upper_bound)
            ).distinct()
            
            results = session.exec(statement).all()

            if results:
                print(f"Airfoils matching {parameter} = {target_value} ({tolerance} {tolerance_type}):")
                for name in results:
                    print(f"- {name}")
                return results
            else:
                print(f"No airfoils found matching {parameter} = {target_value} ({tolerance} {tolerance_type}).")
                return []

    def plot_polar(self, name, Re, Mach):
        """Plots the polar (Cl vs Cd) for a specific airfoil, Reynolds number, and Mach number."""
        aero_data = self.get_aero_coeffs(name, Re, Mach)
        if not aero_data:
            print(f"No aerodynamic data found for {name} (Re={Re}, Mach={Mach})")
            return
        
        # Extract Cl and Cd values
        cl_values = [data.cl for data in aero_data if data.cl is not None]
        cd_values = [data.cd for data in aero_data if data.cd is not None]
        
        if not cl_values or not cd_values:
            print(f"Insufficient data for polar plot for {name} (Re={Re}, Mach={Mach})")
            return
        
        plt.figure(figsize=(8, 6))
        plt.plot(cd_values, cl_values, marker='o', linestyle='-')
        plt.xlabel("Cd (Drag Coefficient)")
        plt.ylabel("Cl (Lift Coefficient)")
        plt.title(f"Lift-Drag Polar for {name} (Re={Re}, Mach={Mach})")
        plt.grid()
        plt.show()

    def plot_coeff_vs_alpha(self, name, coeff="cl", Re=None, Mach=None):
        """
        Plots a coefficient (cl, cd, or cm) versus angle of attack for a specific airfoil.
        
        Args:
            name (str): The name of the airfoil.
            coeff (str): The coefficient to plot ('cl', 'cd', or 'cm').
            Re (float, optional): Reynolds number filter.
            Mach (float, optional): Mach number filter.
        """
        if coeff.lower() not in ["cl", "cd", "cm"]:
            print("Invalid coefficient. Choose 'cl', 'cd', or 'cm'.")
            return

        aero_data = self.get_aero_coeffs(name, Re, Mach)
        if not aero_data:
            print(f"No aerodynamic data found for {name} (Re={Re}, Mach={Mach})")
            return
        
        # Extract alpha and coefficient values
        alpha_values = []
        coeff_values = []
        
        for data in aero_data:
            coeff_value = getattr(data, coeff.lower())
            if data.alpha is not None and coeff_value is not None:
                alpha_values.append(data.alpha)
                coeff_values.append(coeff_value)
        
        if not alpha_values or not coeff_values:
            print(f"Insufficient data for {coeff.upper()} vs alpha plot for {name} (Re={Re}, Mach={Mach})")
            return
        
        # Sort by alpha for proper line plotting
        sorted_data = sorted(zip(alpha_values, coeff_values))
        alpha_values, coeff_values = zip(*sorted_data)
        
        plt.figure(figsize=(8, 6))
        plt.plot(alpha_values, coeff_values, marker='o', linestyle='-')
        plt.xlabel("Angle of Attack (α)")
        plt.ylabel(f"{coeff.upper()} Coefficient")
        plt.title(f"{coeff.upper()} vs. Alpha for {name}" + 
                 (f" (Re={Re})" if Re is not None else "") + 
                 (f" (Mach={Mach})" if Mach is not None else ""))
        plt.grid()
        plt.show()

    def clear_database(self):
        """Clears all data from the database."""
        try:
            with Session(self.engine) as session:
                # Delete all records from each table
                # Delete in reverse order of dependencies
                session.exec(delete(AeroCoeff))
                session.exec(delete(AirfoilGeometry))
                session.exec(delete(Airfoil))
                session.commit()
                print("Database cleared.")
        except Exception as e:
            print(f"Error clearing database: {e}")

    def close(self):
        """Close the database connection."""
        # SQLModel with Session doesn't require explicit connection closing
        # as it's handled by the context manager, but we can dispose the engine
        if hasattr(self, 'engine'):
            self.engine.dispose()
            logging.info("Database engine disposed.")


if __name__ == "__main__":
    # Example usage of the AirfoilDatabase class:
    reynolds_list = [10000, 200000, 500000, 1000000]
    mach_list = 0.2
    alpha_list = [-20, -15, -10, -5, 0, 5, 10, 15, 20]
    ncrit_list = 9
    airfoil_names = []
    folder = r'D:\Mitchell\School\2025 Winter\github\airfoil_database\airfoil_dat_files'
    files = get_files_starting_with(folder, 'm')
    
    db = AirfoilDatabase(db_dir="airfoil_database", db_name='airfoils.db')
    for file in files:
        airfoil_name = file.split('.')[0]
        db.run_airfoil_through_xfoil(airfoil_name, 
                                    reynolds_list, 
                                    mach_list, 
                                    alpha_list, 
                                    ncrit_list)
    
    db.close()
