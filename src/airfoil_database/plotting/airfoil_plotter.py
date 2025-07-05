# plotting/airfoil_plotter.py
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
from sqlmodel import Session, select

from airfoil_database.core.models import Airfoil
from airfoil_database.utilities.helpers import pointcloud_string_to_array

class AirfoilPlotter:
    def __init__(self, database):
        """
        Initialize the airfoil plotter.
        
        Args:
            database: The airfoil database instance
        """
        self.db = database
        self.engine = database.engine
    
    def plot_airfoil(self, name, ax=None, output_dir=None, output_name=None):
        """
        Plots the airfoil using its point cloud data, with markers for individual points.
        
        Args:
            name (str): The name of the airfoil
            ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, creates a new figure.
            output_dir (str, optional): Directory to save the plot
            output_name (str, optional): Filename for the saved plot
            
        Returns:
            matplotlib.axes.Axes or None: The axes object if successful, None otherwise
        """
        data = self.db.get_airfoil_data(name)
        if data:
            description, pointcloud_str, series, source = data
            pointcloud_np = pointcloud_string_to_array(pointcloud_str)
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
                plt.close()
            
            return ax
        else:
            logging.warning(f"Airfoil {name} not found in the database.")
            return None
    
    def plot_multiple_airfoils(self, names, ax=None, output_dir=None, output_name=None):
        """
        Plots multiple airfoils on the same figure.
        
        Args:
            names (list): List of airfoil names to plot
            ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, creates a new figure.
            output_dir (str, optional): Directory to save the plot
            output_name (str, optional): Filename for the saved plot
            
        Returns:
            matplotlib.axes.Axes: The axes object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 7))  # Create a new figure if ax is not provided

        for name in names:
            data = self.db.get_airfoil_data(name)
            if data:
                description, pointcloud_str, series, source = data
                try:
                    points = [line.split() for line in pointcloud_str.strip().split('\n')]
                    x = [float(p[0]) for p in points if len(p) == 2]
                    y = [float(p[1]) for p in points if len(p) == 2]

                    if x and y:
                        ax.plot(x, y, label=name, linestyle='-', marker='o', markersize=3)  # Markers added
                    else:
                        logging.warning(f"No valid point cloud data found for {name}")

                except (ValueError, IndexError) as e:
                    logging.error(f"Error parsing point cloud data for {name}: {e}")
            else:
                logging.warning(f"Airfoil {name} not found in the database.")

        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_title("Airfoil Comparison")
        ax.grid(True)
        ax.axis('equal')
        ax.legend()

        if output_dir is not None:
            if output_name is None:
                output_name = ' vs '.join(names) + '.png'
            plt.savefig(os.path.join(output_dir, output_name))
            plt.close()
        
        return ax
    
    def add_airfoil_to_plot(self, airfoil_name, ax, linestyle='-', marker='o', markersize=3, label=None):
        """
        Adds an airfoil's point cloud to a Matplotlib plot.
        
        Args:
            airfoil_name (str): The name of the airfoil
            ax (matplotlib.axes.Axes): The axes to plot on
            linestyle (str): Line style for the plot
            marker (str): Marker style for the plot
            markersize (int): Size of markers
            label (str, optional): Custom label for the plot legend
            
        Returns:
            bool: True if successful, False otherwise
        """
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
                    return True
                else:
                    logging.warning(f"Airfoil '{airfoil_name}' not found.")
                    return False

        except Exception as e:
            logging.error(f"Error plotting airfoil: {e}")
            return False
