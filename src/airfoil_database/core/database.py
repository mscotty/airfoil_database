import os
import csv
import json
import logging
from typing import List, Optional, Dict, Any, Tuple
from sqlmodel import Session, SQLModel, create_engine, select, delete
import pandas as pd

from airfoil_database.core.models import Airfoil, AeroCoeff, AirfoilGeometry
from airfoil_database.classes.AirfoilSeries import AirfoilSeries

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
