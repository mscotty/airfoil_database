# plotting/data_plotter.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import logging
from collections import Counter
from sqlmodel import Session, select

from airfoil_database.core.models import Airfoil, AeroCoeff, AirfoilGeometry
from airfoil_database.plotting.plot_histogram import plot_histogram
from airfoil_database.plotting.plot_bar_chart import plot_horizontal_bar_chart

class DataPlotter:
    def __init__(self, database):
        """
        Initialize the data plotter.
        
        Args:
            database: The airfoil database instance
        """
        self.db = database
        self.engine = database.engine
    
    def plot_airfoil_series_pie(self, output_dir=None, output_name=None):
        """
        Fetches airfoil series data from the database and plots a pie chart.
        
        Args:
            output_dir (str, optional): Directory to save the plot
            output_name (str, optional): Filename for the saved plot
        """
        # Connect to database and retrieve all airfoil_series values
        with Session(self.engine) as session:
            statement = select(Airfoil.airfoil_series)
            series_list = [row for row in session.exec(statement) if row]

        if not series_list:
            logging.warning("No airfoil series data found in the database.")
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
            plt.savefig(os.path.join(output_dir if output_dir is not None else '', 
                                     output_name if output_name is not None else 'airfoil_series_pie.png'))
            plt.close()
        else:
            plt.show()  # Display the pie chart
    
    def plot_airfoil_series_horizontal_bar(self, **kwargs):
        """
        Fetches airfoil series data from the database and plots a horizontal bar chart.
        
        Args:
            **kwargs: Additional arguments to pass to plot_horizontal_bar_chart
        """
        # Connect to database and retrieve all airfoil_series values
        with Session(self.engine) as session:
            statement = select(Airfoil.airfoil_series)
            series_list = [row for row in session.exec(statement) if row]
        
        if not series_list:
            logging.warning("No airfoil series data found in the database.")
            return

        # Count occurrences of each airfoil series
        series_counts = Counter(series_list)

        # Extract labels and counts for bar chart
        labels = list(series_counts.keys())
        counts = list(series_counts.values())

        plot_horizontal_bar_chart(counts, labels, **kwargs)

    def plot_leading_edge_radius(self, parameter="chord_length"):
        """
        Plots leading-edge radius against a specified parameter.
        
        Args:
            parameter (str): The parameter to plot against leading edge radius
        """
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
        else:
            logging.warning(f"No data found for leading edge radius vs {parameter}")

    def plot_trailing_edge_angle(self, parameter="chord_length"):
        """
        Plots trailing-edge angle against a specified parameter.
        
        Args:
            parameter (str): The parameter to plot against trailing edge angle
        """
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
        else:
            logging.warning(f"No data found for trailing edge angle vs {parameter}")

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
        else:
            logging.warning("No geometry data found for correlation analysis")
    
    def plot_polar(self, name, Re, Mach):
        """
        Plots the polar (Cl vs Cd) for a specific airfoil, Reynolds number, and Mach number.
        
        Args:
            name (str): The name of the airfoil
            Re (float): Reynolds number
            Mach (float): Mach number
        """
        aero_data = self.db.get_aero_coeffs(name, Re, Mach)
        if not aero_data:
            logging.warning(f"No aerodynamic data found for {name} (Re={Re}, Mach={Mach})")
            return
        
        # Extract Cl and Cd values
        cl_values = [data.cl for data in aero_data if data.cl is not None]
        cd_values = [data.cd for data in aero_data if data.cd is not None]
        
        if not cl_values or not cd_values:
            logging.warning(f"Insufficient data for polar plot for {name} (Re={Re}, Mach={Mach})")
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
            name (str): The name of the airfoil
            coeff (str): The coefficient to plot ('cl', 'cd', or 'cm')
            Re (float, optional): Reynolds number filter
            Mach (float, optional): Mach number filter
        """
        if coeff.lower() not in ["cl", "cd", "cm"]:
            logging.warning("Invalid coefficient. Choose 'cl', 'cd', or 'cm'.")
            return

        aero_data = self.db.get_aero_coeffs(name, Re, Mach)
        if not aero_data:
            logging.warning(f"No aerodynamic data found for {name} (Re={Re}, Mach={Mach})")
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
            logging.warning(f"Insufficient data for {coeff.upper()} vs alpha plot for {name} (Re={Re}, Mach={Mach})")
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
