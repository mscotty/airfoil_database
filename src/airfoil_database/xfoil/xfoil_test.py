# example_xfoil_analysis.py

import logging
import os
from airfoil_database.core.database import AirfoilDatabase
from airfoil_database.analysis.aero_analyzer import AeroAnalyzer
from airfoil_database.classes.XFoilRunner import XFoilRunner
from airfoil_database.utilities.get_top_level_module import get_project_root

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Initialize the database
    DB_FOLDER = os.path.join(get_project_root(), "airfoil_database")
    DB_NAME = "airfoils.db"
    db_path = os.path.join(DB_FOLDER, DB_NAME)
    db = AirfoilDatabase(db_name=DB_NAME, db_dir=DB_FOLDER)
    
    # Initialize AeroAnalyzer
    analyzer = AeroAnalyzer(db)
    
    # Initialize XFoilRunner with AeroAnalyzer
    # For Windows, you might need: xfoil_path = "path/to/xfoil.exe"
    # For Linux/Mac: xfoil_path = "xfoil" (if in PATH) or "/path/to/xfoil"
    xfoil_path = "xfoil"  # Adjust this path as needed for your system
    xfoil = XFoilRunner(aero_analyzer=analyzer, xfoil_executable=xfoil_path)
    
    # Set analysis parameters
    airfoil_name = "naca2421"  # Make sure this airfoil exists in your database
    reynolds_list = [1e6]  # Reynolds number
    mach_list = [0.2]  # Mach number
    alpha_list = [-5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0]  # Angles of attack
    ncrit_list = [2.0]  # Ncrit parameter
    
    # Run the analysis
    logging.info(f"Starting XFoil analysis for {airfoil_name}")
    xfoil.run_analysis_parallel(
        airfoil_name=airfoil_name,
        reynolds_list=reynolds_list,
        mach_list=mach_list,
        alpha_list=alpha_list,
        ncrit_list=ncrit_list,
        max_workers=4  # Adjust based on your CPU cores
    )
    
    logging.info("Analysis complete!")

if __name__ == "__main__":
    main()
