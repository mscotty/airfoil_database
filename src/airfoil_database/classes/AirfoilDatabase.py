import os
import sqlite3
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

from airfoil_database.plotting.plot_histogram import plot_histogram
from airfoil_database.plotting.plot_bar_chart import plot_horizontal_bar_chart

from airfoil_database.formulas.airfoil.compute_aspect_ratio import calculate_aspect_ratio
from airfoil_database.formulas.airfoil.compute_LE_radius import leading_edge_radius
from airfoil_database.formulas.airfoil.compute_TE_angle import trailing_edge_angle
from airfoil_database.formulas.airfoil.compute_thickness_camber import compute_thickness_camber
from airfoil_database.formulas.airfoil.compute_span import calculate_span
from airfoil_database.formulas.airfoil.compute_thickness_to_chord_ratio import thickness_to_chord_ratio

from airfoil_database.xfoil.fix_airfoil_data import normalize_pointcloud
from airfoil_database.xfoil.fix_airfoil_pointcloud_v2 import *
from airfoil_database.xfoil.interpolate_points import interpolate_points
from airfoil_database.xfoil.calculate_distance import calculate_min_distance_sum
from airfoil_database.xfoil.fix_point_cloud import *

from airfoil_database.classes.XFoilRunner import XFoilRunner
from airfoil_database.classes.AirfoilSeries import AirfoilSeries

class AirfoilDatabase:
    def __init__(self, db_name="airfoil_data.db", db_dir="."):
        self.db_path = os.path.join(db_dir, db_name) # Path to the database
        os.makedirs(db_dir, exist_ok=True) # Create directory if it doesn't exist.
        # self.write_lock = threading.Lock() # Potentially problematic with multiprocessing
        self._enable_wal()
        self._create_table()
        logging.info(f"AirfoilDatabase initialized with db: {self.db_path}")

    def _enable_wal(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("PRAGMA journal_mode=WAL;")
                logging.info("SQLite WAL mode enabled.")
        except sqlite3.Error as e:
            logging.error(f"Failed to enable WAL mode for {self.db_path}: {e}")

    def _create_table(self):
        # Add error handling
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # Airfoils table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS airfoils (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT UNIQUE NOT NULL,
                        description TEXT,
                        pointcloud TEXT,
                        airfoil_series TEXT,
                        source TEXT
                    )
                """)
                # Aero Coefficients table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS aero_coeffs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        reynolds_number REAL NOT NULL,
                        mach REAL NOT NULL,
                        ncrit REAL NOT NULL,
                        alpha REAL NOT NULL,
                        cl REAL,
                        cd REAL,
                        cm REAL,
                        FOREIGN KEY (name) REFERENCES airfoils(name),
                        UNIQUE (name, reynolds_number, mach, alpha)
                    )
                """)
                # Geometry table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS airfoil_geometry (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT UNIQUE NOT NULL,
                        max_thickness REAL,
                        max_camber REAL,
                        chord_length REAL,
                        span REAL,
                        aspect_ratio REAL,
                        leading_edge_radius REAL,
                        trailing_edge_angle REAL,
                        thickness_to_chord_ratio REAL,
                        thickness_distribution TEXT,
                        camber_distribution TEXT,
                        normalized_chord TEXT,
                        FOREIGN KEY (name) REFERENCES airfoils(name)
                    )
                """)
                conn.commit()
                logging.info("Database tables ensured.")
        except sqlite3.Error as e:
            logging.error(f"Failed to create/verify database tables: {e}")
            raise # Re-raise critical error
        
    def store_airfoil_data(self, name, description, pointcloud, airfoil_series, source, overwrite=False):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                if overwrite:
                    cursor.execute("REPLACE INTO airfoils (name, description, pointcloud, airfoil_series, source) VALUES (?, ?, ?, ?, ?)", (name, description, pointcloud, airfoil_series.value, source))
                else:
                    cursor.execute("INSERT INTO airfoils (name, description, pointcloud, airfoil_series, source) VALUES (?, ?, ?, ?, ?)", (name, description, pointcloud, airfoil_series.value, source))
                conn.commit()
                print(f"Stored: {name} in database.")
        except sqlite3.IntegrityError:
            if overwrite:
                print(f"Updated: {name} in database.")
            else:
                print(f"Airfoil {name} already exists in the database. Use overwrite=True to update.")
    
    def store_bulk_airfoil_data(self, data_list, overwrite=False):
        """
        Stores multiple airfoil data records in the database using executemany.

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

        rows_to_insert = []
        names_to_check = {data['name'] for data in data_list if 'name' in data} # Get unique names from input

        inserted_count = 0

        try:
            with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                cursor = conn.cursor()

                if not overwrite:
                    # Find names that already exist if we are not overwriting
                    placeholders = ','.join('?' * len(names_to_check))
                    query = f"SELECT name FROM airfoils WHERE name IN ({placeholders})"
                    cursor.execute(query, list(names_to_check))
                    existing_names = {row[0] for row in cursor.fetchall()}
                    logging.info(f"Found {len(existing_names)} existing airfoils matching input names. Skipping them.")
                else:
                    existing_names = set() # Overwrite mode, don't skip any

                # Prepare data tuples for executemany
                for data in data_list:
                    name = data.get('name')
                    if not name:
                        logging.warning("Skipping record due to missing 'name'.")
                        continue
                    if name in existing_names:
                        continue # Skip if not overwriting and name exists

                    # Ensure all required fields are present, provide defaults if necessary
                    desc = data.get('description', '')
                    pc_str = data.get('pointcloud', '')
                    series = data.get('airfoil_series', 'UNKNOWN') # Use default/enum value
                    source = data.get('source', '')
                    # Add filepath if you add that column: filepath = data.get('filepath', '')

                    # Order must match the INSERT statement columns
                    rows_to_insert.append((name, desc, pc_str, series, source)) # Add filepath here if needed

                if not rows_to_insert:
                     logging.info("No new data to insert after filtering existing names.")
                     return 0

                # Use INSERT OR REPLACE if overwrite is True, otherwise INSERT OR IGNORE
                sql = """
                    INSERT OR {action} INTO airfoils (name, description, pointcloud, airfoil_series, source)
                    VALUES (?, ?, ?, ?, ?)
                """.format(action="REPLACE" if overwrite else "IGNORE")

                cursor.executemany(sql, rows_to_insert)
                conn.commit()
                inserted_count = cursor.rowcount # executemany returns -1 usually, check changes
                # Getting accurate count after executemany can be tricky, check changes instead
                changes = conn.total_changes
                # This counts total changes since connection, might need refinement if connection is reused
                # A more reliable way might be len(rows_to_insert) if using INSERT OR IGNORE/REPLACE
                logging.info(f"Bulk insert/replace finished. Affected rows (approx): {len(rows_to_insert)}")
                # Return the number intended for insertion, actual count might differ slightly
                # depending on IGNORE/REPLACE behavior and concurrent access.
                return len(rows_to_insert)

        except sqlite3.Error as e:
            logging.error(f"SQLite error during bulk insert: {e}", exc_info=True)
            return 0 # Indicate failure/no insertion
        except Exception as e:
            logging.error(f"Unexpected error during bulk insert: {e}", exc_info=True)
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

                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    for row in reader:
                        name = row.get('name')  # Assuming 'name' column exists
                        if not name:
                            print("Warning: Skipping row with missing 'name'.")
                            continue
                        if overwrite:
                            self._delete_airfoil_data(name, conn, cursor)

                        insert_query = "INSERT OR REPLACE INTO airfoils ("
                        values_query = "VALUES ("
                        values = []

                        for header in headers:
                            if header in ['name', 'description', 'pointcloud', 'airfoil_series', 'source']:
                                insert_query += f"{header}, "
                                values_query += "?, "
                                values.append(row.get(header))

                        insert_query = insert_query.rstrip(', ') + ") "
                        values_query = values_query.rstrip(', ') + ")"
                        query = insert_query + values_query

                        try:
                            cursor.execute(query, values)
                            conn.commit()
                            print(f"Added/Updated: {name} from CSV.")
                        except sqlite3.IntegrityError as e:
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

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                for name, airfoil_data in data.items():
                    if overwrite:
                        self._delete_airfoil_data(name, conn, cursor)
                    try:
                        insert_query = "INSERT OR REPLACE INTO airfoils (name, description, pointcloud, airfoil_series, source) VALUES (?, ?, ?, ?, ?)"
                        cursor.execute(insert_query, (name, airfoil_data.get('description'), airfoil_data.get('pointcloud'), airfoil_data.get('airfoil_series'), airfoil_data.get('source')))
                        conn.commit()
                        print(f"Added/Updated: {name} from JSON.")
                    except sqlite3.IntegrityError as e:
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
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Update airfoils table
                cursor.execute("""
                    UPDATE airfoils
                    SET name = ?, description = ?, airfoil_series = ?, source = ?
                    WHERE name = ?
                """, (new_name, description, series, source, old_name))

                # Update aero_coeffs table
                cursor.execute("""
                    UPDATE aero_coeffs
                    SET name = ?
                    WHERE name = ?
                """, (new_name, old_name))

                # Update airfoil_geometry table
                cursor.execute("""
                    UPDATE airfoil_geometry
                    SET name = ?
                    WHERE name = ?
                """, (new_name, old_name))

                conn.commit()
                print(f"Updated airfoil info for {old_name} to {new_name}.")

        except sqlite3.Error as e:
            print(f"Error updating airfoil info: {e}")

    def update_airfoil_series(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name, description, airfoil_series FROM airfoils")
            airfoils = cursor.fetchall()
            for name, description, airfoil_series in airfoils:
                airfoil_series_curr = AirfoilSeries.from_string(airfoil_series)
                if airfoil_series_curr == AirfoilSeries.OTHER:
                    airfoil_series_curr = AirfoilSeries.identify_airfoil_series(name)
                    if airfoil_series_curr == AirfoilSeries.OTHER:
                        airfoil_series_curr = AirfoilSeries.identify_airfoil_series(description)
                        #TODO: Add more logic to get the airfoil series
                    cursor.execute("UPDATE airfoils SET airfoil_series = ? WHERE name = ?", (airfoil_series_curr.value, name))
                    conn.commit()

    def _delete_airfoil_data(self, name, conn, cursor):
        """Deletes all data associated with an airfoil."""
        cursor.execute("DELETE FROM airfoils WHERE name = ?", (name,))
        cursor.execute("DELETE FROM aero_coeffs WHERE name = ?", (name,))
        cursor.execute("DELETE FROM airfoil_geometry WHERE name = ?", (name,))
        conn.commit()
        print(f"Deleted existing data for {name}.")

    def get_airfoil_data(self, name):
        """Retrieves airfoil data including description, pointcloud, series, and source."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Set row factory for dictionary access if preferred, otherwise tuple is fine
                # conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT description, pointcloud, airfoil_series, source FROM airfoils WHERE name = ?", (name,))
                result = cursor.fetchone()
                if result:
                    logging.debug(f"Retrieved data for airfoil: {name}")
                    return result # Returns tuple: (description, pointcloud, series, source)
                else:
                    logging.warning(f"Airfoil '{name}' not found in database.")
                    return None
        except sqlite3.Error as e:
            logging.error(f"Error retrieving airfoil data for {name}: {e}")
            return None
    
    def get_airfoil_dataframe(self):
        """Returns a Pandas DataFrame with airfoil names, series, and number of points."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name, airfoil_series, pointcloud FROM airfoils")
            results = cursor.fetchall()

        data = []
        for row in results:
            name, series, pointcloud = row
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
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 
                        id, name, max_thickness, max_camber, chord_length, span, 
                        aspect_ratio, leading_edge_radius, trailing_edge_angle, 
                        thickness_to_chord_ratio 
                    FROM airfoil_geometry
                """)
                results = cursor.fetchall()

            if not results:
                return pd.DataFrame()  # Return an empty DataFrame if no data found

            column_names = [description[0] for description in cursor.description]
            df = pd.DataFrame(results, columns=column_names)
            return df

        except sqlite3.Error as e:
            print(f"SQLite error: {e}")
            return pd.DataFrame() #Return empty dataframe on error.
        except Exception as e:
            print(f"An error occurred: {e}")
            return pd.DataFrame() #Return empty dataframe on error.
    
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
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT pointcloud FROM airfoils WHERE name = ?", (name,))
            result = cursor.fetchone()

            if result and result[0]:
                pointcloud_np = self._pointcloud_to_numpy(result[0])
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

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM airfoils")
            airfoils = cursor.fetchall()

            for airfoil in airfoils:
                name = airfoil[0]
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
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # Select names of airfoils that have some pointcloud data
                cursor.execute("SELECT name FROM airfoils WHERE pointcloud IS NOT NULL AND pointcloud != '' ORDER BY name")
                results = cursor.fetchall()
                airfoil_names = [row[0] for row in results]
        except sqlite3.Error as e:
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
            # Optional: Add a small delay if needed to avoid overwhelming resources,
            # though fixing is likely less intensive than XFoil runs.
            # time.sleep(0.1)

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
        if not airfoil_data or not airfoil_data[1]: # Check data exists and pointcloud string is not empty
            logging.error(f"Cannot fix: Airfoil '{name}' not found or has no pointcloud data.")
            return None

        description, pointcloud_str, airfoil_series, source = airfoil_data

        airfoil_series = AirfoilSeries(airfoil_series)

        # Parse the pointcloud string
        points_array = parse_pointcloud_string(pointcloud_str)
        if points_array is None:
            logging.error(f"Could not parse original pointcloud string for {name}.")
            return None # Parsing failed

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
                self.store_airfoil_data(name, description, fixed_pointcloud_str, airfoil_series, source)
                logging.info(f"Stored fixed pointcloud for {name} in database.")
            return fixed_points_array # Return the fixed array

    def output_pointcloud_to_file(self, airfoil_name, file_path):
        """
        Outputs the point cloud of an airfoil to a text file.

        Args:
            airfoil_name (str): The name of the airfoil.
            file_path (str): The path to the output text file.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT pointcloud FROM airfoils WHERE name = ?", (airfoil_name,))
                result = cursor.fetchone()

                if result and result[0]:
                    pointcloud_str = result[0]
                    with open(file_path, 'w') as file:
                        file.write(pointcloud_str)
                else:
                    print(f"Airfoil '{airfoil_name}' not found or point cloud is empty.")

        except sqlite3.Error as e:
            print(f"Error outputting point cloud: {e}")

    def plot_airfoil_series_pie(self, output_dir=None, output_name=None):
        """Fetches airfoil series data from the database and plots a pie chart."""

        # Connect to database and retrieve all airfoil_series values
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT airfoil_series FROM airfoils")
            series_list = [row[0] for row in cursor.fetchall() if row[0]]

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
            plt.show() # Display the pie chart
    
    def plot_airfoil_series_horizontal_bar(self, **kwargs):
        """Fetches airfoil series data from the database and plots a horizontal bar chart."""

        # Connect to database and retrieve all airfoil_series values
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT airfoil_series FROM airfoils")
            series_list = [row[0] for row in cursor.fetchall() if row[0]]
        
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
            if ax is None: #If no ax was passed, show the plot.
                plt.show()
        else:
            if output_name is None:
                output_name = ' vs '.join(names) + '.png'
            plt.savefig(os.path.join(output_dir, output_name))

        return ax #Return the ax object.
    
    def add_airfoil_to_plot(self, airfoil_name, ax, linestyle='-', marker='o', markersize=3, label=None):
        """Adds an airfoil's point cloud to a Matplotlib plot."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT pointcloud FROM airfoils WHERE name = ?", (airfoil_name,))
                result = cursor.fetchone()

                if result and result[0]:
                    pointcloud_str = result[0]
                    points = [line.split() for line in pointcloud_str.strip().split('\n')]
                    points = np.array([[float(p[0]), float(p[1])] for p in points if len(p) == 2])
                    if label: #check if label exists, if not, then use airfoil name.
                        ax.plot(points[:, 0], points[:, 1], linestyle=linestyle, marker=marker, markersize=markersize, label=label)
                    else:
                        ax.plot(points[:, 0], points[:, 1], linestyle=linestyle, marker=marker, markersize=markersize, label=airfoil_name)
                else:
                    print(f"Airfoil '{airfoil_name}' not found.")

        except sqlite3.Error as e:
            print(f"Error plotting airfoil: {e}")
    
    def find_best_matching_airfoils(self, input_pointcloud_str, num_matches=3):
        """
        Compares an input point cloud to the airfoils in the database and returns the best matches.
        """
        # try:
        input_points = [line.split() for line in input_pointcloud_str.strip().split('\n')]
        input_points = np.array([[float(p[0]), float(p[1])] for p in input_points if len(p) == 2])
        normalized_input_points = normalize_pointcloud(input_points)
        if len(normalized_input_points) == 0:
            return []

        interpolated_input_points = interpolate_points(normalized_input_points)

        matches = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name, pointcloud FROM airfoils")
            airfoils = cursor.fetchall()

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

        # except Exception as e:
        #     print(f"Error finding best matching airfoils: {e}")
        #     return []

    def compute_geometry_metrics(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name, pointcloud FROM airfoils")
            airfoils = cursor.fetchall()

            for name, pointcloud in airfoils:
                rows = pointcloud.split('\n')
                rows = [x for x in rows if x.strip()]
                points = np.array([np.fromstring(row, dtype=float, sep=' ') for row in rows])
                #points = reorder_airfoil_data(points)

                x_coords, thickness, camber = compute_thickness_camber(points)
                LE_radius = leading_edge_radius(points)
                TE_angle = trailing_edge_angle(points)
                chord_length = max(x_coords) - min(x_coords)
                t_to_c = thickness_to_chord_ratio(thickness, chord_length)
                span = calculate_span(points)
                aspect_ratio = calculate_aspect_ratio(span, chord_length)
                max_thickness = max(thickness)
                max_camber = max(camber)
                
                # Calculate normalized chord
                normalized_chord = np.linspace(0, 1, len(thickness))

                # Store distribution data as comma-separated strings
                thickness_dist_str = ",".join(map(str, thickness))
                camber_dist_str = ",".join(map(str, camber))
                normalized_chord_str = ",".join(map(str, normalized_chord))

                cursor.execute("""
                    INSERT OR REPLACE INTO airfoil_geometry (name, max_thickness, max_camber, leading_edge_radius, trailing_edge_angle, chord_length, thickness_to_chord_ratio, span, aspect_ratio, thickness_distribution, camber_distribution, normalized_chord)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (name, max_thickness, max_camber, LE_radius, TE_angle, chord_length, t_to_c, span, aspect_ratio, thickness_dist_str, camber_dist_str, normalized_chord_str))
                conn.commit()
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
                            "span", "aspect_ratio"]

        if parameter not in valid_parameters:
            print(f"Invalid parameter. Choose from: {', '.join(valid_parameters)}")
            return []

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            if tolerance_type == "absolute":
                lower_bound = target_value - tolerance
                upper_bound = target_value + tolerance
            elif tolerance_type == "percentage":
                lower_bound = target_value * (1 - tolerance / 100.0)
                upper_bound = target_value * (1 + tolerance / 100.0)
            else:
                print("Invalid tolerance_type. Choose 'absolute' or 'percentage'.")
                return []

            query = f"SELECT name FROM airfoil_geometry WHERE {parameter} BETWEEN ? AND ?"
            cursor.execute(query, (lower_bound, upper_bound))
            results = cursor.fetchall()

            airfoil_names = [row[0] for row in results]
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
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT name, leading_edge_radius, {parameter} FROM airfoil_geometry")
            results = cursor.fetchall()

        if results:
            names, radii, params = zip(*results)
            plt.figure(figsize=(8, 6))
            plt.scatter(params, radii)
            plt.xlabel(parameter)
            plt.ylabel("Leading Edge Radius")
            plt.title("Leading Edge Radius vs. " + parameter)
            plt.grid(True)
            plt.show()

    def plot_trailing_edge_angle(self, parameter="chord_length"):
        """Plots trailing-edge angle against a specified parameter."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT name, trailing_edge_angle, {parameter} FROM airfoil_geometry")
            results = cursor.fetchall()

        if results:
            names, angles, params = zip(*results)
            plt.figure(figsize=(8, 6))
            plt.scatter(params, angles)
            plt.xlabel(parameter)
            plt.ylabel("Trailing Edge Angle")
            plt.title("Trailing Edge Angle vs. " + parameter)
            plt.grid(True)
            plt.show()

    def plot_geometry_correlations(self):
        """Plots correlations between geometric parameters using a heatmap."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT max_thickness, max_camber, leading_edge_radius, trailing_edge_angle, chord_length, thickness_to_chord_ratio, span, aspect_ratio FROM airfoil_geometry")
            results = cursor.fetchall()

        if results:
            df = pd.DataFrame(results, columns=["max_thickness", "max_camber", "leading_edge_radius", "trailing_edge_angle", "chord_length", "thickness_to_chord_ratio", "span", "aspect_ratio"])
            correlation_matrix = df.corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
            plt.title("Geometric Parameter Correlations")
            plt.show()
    
    def store_aero_coeffs(self, name, reynolds_number, mach, ncrit, alpha, cl, cd, cm):
        """Stores a single row of aerodynamic coefficient data."""
        # Ensure this method handles database connection safely (e.g., using 'with')
        # Your existing implementation likely does this already.
        try:
            # Using write_lock if needed for thread safety, although ProcessPoolExecutor
            # might make this less critical unless multiple AirfoilDatabase instances
            # are used across processes with the same DB file without WAL.
            # with self.write_lock: # Uncomment if thread safety issues arise
            with sqlite3.connect(self.db_path, timeout=10.0) as conn: # Added timeout
                cursor = conn.cursor()
                # Use INSERT OR REPLACE to handle potential unique constraint violations gracefully
                cursor.execute("""
                    INSERT OR REPLACE INTO aero_coeffs (name, reynolds_number, mach, ncrit, alpha, cl, cd, cm)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (name, reynolds_number, mach, ncrit, alpha, cl, cd, cm))
                conn.commit()
                # Reduce logging noise, maybe log only periodically or on completion
                # logging.debug(f"Stored aero coeffs for {name} (Re={reynolds_number}, M={mach}, A={alpha})")
        except sqlite3.Error as e:
            logging.error(f"SQLite error storing aero coeffs for {name} Re={reynolds_number} M={mach} Ncrit={ncrit} A={alpha}: {e}")
        except Exception as e:
             logging.error(f"Unexpected error storing aero coeffs for {name} Re={reynolds_number} M={mach} Ncrit={ncrit} A={alpha}: {e}")

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
            # This instance itself is NOT passed to worker processes in the refactored runner
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
             # Optionally re-raise if you want the calling code to handle it
             # raise
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
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # Fetch only names that have pointcloud data
                cursor.execute("SELECT name FROM airfoils WHERE pointcloud IS NOT NULL AND pointcloud != '' ORDER BY name") # Added ORDER BY
                results = cursor.fetchall()
                airfoil_names = [row[0] for row in results]
        except sqlite3.Error as e:
            logging.error(f"Failed to retrieve airfoil names from database: {e}")
            return # Cannot proceed without airfoil names

        if not airfoil_names:
            logging.warning("No airfoils with pointcloud data found in the database to run.")
            return

        logging.info(f"Found {len(airfoil_names)} airfoils to process.")

        # Run analysis for each airfoil serially.
        # Parallelism happens *within* run_airfoil_through_xfoil for conditions.
        # To run different *airfoils* in parallel, this loop would need modification
        # (e.g., using another ProcessPoolExecutor here).
        for i, name in enumerate(airfoil_names):
            logging.info(f"--- Processing airfoil {i+1}/{len(airfoil_names)}: {name} ---")
            # Check if airfoil still exists ( belt-and-suspenders check, might be redundant)
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
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM aero_coeffs WHERE name = ?"
            params = [name]
            if Re is not None:
                query += " AND reynolds_number = ?"
                params.append(Re)
            if Mach is not None:
                query += " AND mach = ?"
                params.append(Mach)
            cursor.execute(query, tuple(params))
            return cursor.fetchall()

    def find_airfoils_by_xfoil_results(self, 
                                       parameter, 
                                       target_value, 
                                       tolerance, 
                                       tolerance_type="absolute"):
        """
        Finds airfoils based on XFOIL results.

        Args:
            parameter (str): The XFOIL result parameter (reynolds, alpha, cl, cd, cm).
            target_value (float): The target value for the parameter.
            tolerance (float): The tolerance for the search.
            tolerance_type (str): "absolute" or "percentage".
        """
        valid_parameters = ["reynolds", "alpha", "cl", "cd", "cm"]

        if parameter not in valid_parameters:
            print(f"Invalid parameter. Choose from: {', '.join(valid_parameters)}")
            return []

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            if tolerance_type == "absolute":
                lower_bound = target_value - tolerance
                upper_bound = target_value + tolerance
            elif tolerance_type == "percentage":
                lower_bound = target_value * (1 - tolerance / 100.0)
                upper_bound = target_value * (1 + tolerance / 100.0)
            else:
                print("Invalid tolerance_type. Choose 'absolute' or 'percentage'.")
                return []

            query = f"SELECT airfoil_name FROM xfoil_results WHERE {parameter} BETWEEN ? AND ?"
            cursor.execute(query, (lower_bound, upper_bound))
            results = cursor.fetchall()

            airfoil_names = [row[0] for row in results]
            if airfoil_names:
                print(f"Airfoils matching {parameter} = {target_value} ({tolerance} {tolerance_type}):")
                for name in airfoil_names:
                    print(f"- {name}")
                return airfoil_names
            else:
                print(f"No airfoils found matching {parameter} = {target_value} ({tolerance} {tolerance_type}).")
                return []

    def plot_polar(self, name, Re, Mach):
        df = self.get_aero_coeffs(name, Re, Mach)
        if df is None or df.empty:
            print(f"No aerodynamic data found for {name} (Re={Re}, Mach={Mach})")
            return
        
        plt.figure(figsize=(8, 6))
        plt.plot(df["Cd"], df["Cl"], marker='o', linestyle='-')
        plt.xlabel("Cd (Drag Coefficient)")
        plt.ylabel("Cl (Lift Coefficient)")
        plt.title(f"Lift-Drag Polar for {name} (Re={Re}, Mach={Mach})")
        plt.grid()
        plt.show()

    def plot_coeff_vs_alpha(self, name, coeff="Cl", Re=None, Mach=None):
        if coeff not in ["Cl", "Cd", "Cm"]:
            print("Invalid coefficient. Choose 'Cl', 'Cd', or 'Cm'.")
            return

        df = self.get_aero_coeffs(name, Re, Mach)
        if df is None or df.empty:
            print(f"No aerodynamic data found for {name} (Re={Re}, Mach={Mach})")
            return
        
        plt.figure(figsize=(8, 6))
        plt.plot(df["Alpha"], df[coeff], marker='o', linestyle='-')
        plt.xlabel("Angle of Attack (α)")
        plt.ylabel(coeff)
        plt.title(f"{coeff} vs. Alpha for {name} (Re={Re}, Mach={Mach})")
        plt.grid()
        plt.show()

    def clear_database(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM airfoils")
            cursor.execute("DELETE FROM aero_coeffs")
            conn.commit()
            print("Database cleared.")

    def close(self):
        # If you manage a persistent connection in __init__, close it here.
        # Since the methods use 'with sqlite3.connect()', explicit close might not be needed.
        logging.info("Database connection closed.")


if __name__ == "__main__":
    # Example usage of the AirfoilDatabase class:
    reynolds_list = [10000, 200000]
    mach_list = 0.2
    alpha_list = [0, 5]
    ncrit_list = 9
    airfoil_name = 'fx67k150'

    db = AirfoilDatabase(db_dir="airfoil_database", db_name='selig_airfoils.db')
    """db.run_airfoil_through_xfoil(airfoil_name, 
                                 reynolds_list, 
                                 mach_list, 
                                 alpha_list, 
                                 ncrit_list)
    out = db.get_aero_coeffs(airfoil_name)"""
    #print(out)
    out = db.get_airfoil_data(airfoil_name)
    out_folder = rf'D:\Mitchell\School\airfoils\{airfoil_name}'
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    with open(os.path.join(out_folder, f'{airfoil_name}_0.dat'), 'w+') as file:
        file.write(out[1])
    print(out)
    """db.fix_all_airfoils()
    out = db.get_airfoil_data('ag10')
    print(out)"""
    # db.update_airfoil_series()
    # db.compute_geometry_metrics()
    # db.check_airfoil_validity()
    # db.fix_all_airfoils()
    # db.check_airfoil_validity()
    db.close()
    

    
    
