# analysis/aero_analyzer.py
import logging
from sqlmodel import Session, select
from typing import List, Optional, Dict, Any, Tuple

from airfoil_database.core.models import Airfoil, AeroCoeff

class AeroAnalyzer:
    def __init__(self, database):
        """
        Initialize the aerodynamic analyzer.
        
        Args:
            database: The airfoil database instance
        """
        self.db = database
        self.engine = database.engine
    
    def store_aero_coeffs(self, name, reynolds_number, mach, ncrit, alpha, cl, cd, cm):
        """
        Stores a single row of aerodynamic coefficient data.
        
        Args:
            name (str): Airfoil name
            reynolds_number (float): Reynolds number
            mach (float): Mach number
            ncrit (float): Transition criterion
            alpha (float): Angle of attack
            cl (float): Lift coefficient
            cd (float): Drag coefficient
            cm (float): Moment coefficient
        """
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
    
    def get_aero_coeffs(self, name, Re=None, Mach=None):
        """
        Retrieves aerodynamic coefficient data for an airfoil.
        
        Args:
            name (str): The name of the airfoil
            Re (float, optional): Reynolds number filter
            Mach (float, optional): Mach number filter
            
        Returns:
            list: List of AeroCoeff objects matching the criteria
        """
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
            
        Returns:
            list: List of airfoil names matching the criteria
        """
        valid_parameters = ["reynolds_number", "alpha", "mach", "ncrit", "cl", "cd", "cm"]

        if parameter not in valid_parameters:
            logging.warning(f"Invalid parameter. Choose from: {', '.join(valid_parameters)}")
            return []

        with Session(self.engine) as session:
            if tolerance_type == "absolute":
                lower_bound = target_value - tolerance
                upper_bound = target_value + tolerance
            elif tolerance_type == "percentage":
                lower_bound = target_value * (1 - tolerance / 100.0)
                upper_bound = target_value * (1 + tolerance / 100.0)
            else:
                logging.warning("Invalid tolerance_type. Choose 'absolute' or 'percentage'.")
                return []

            # Create a dynamic query using SQLAlchemy expressions
            column = getattr(AeroCoeff, parameter)
            statement = select(AeroCoeff.name).where(
                (column >= lower_bound) & (column <= upper_bound)
            ).distinct()
            
            results = session.exec(statement).all()

            if results:
                logging.info(f"Found {len(results)} airfoils matching {parameter} = {target_value} ({tolerance} {tolerance_type})")
                return results
            else:
                logging.info(f"No airfoils found matching {parameter} = {target_value} ({tolerance} {tolerance_type}).")
                return []
