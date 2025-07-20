# resume_analysis.py

import logging
import os
from airfoil_database.core.database import AirfoilDatabase
from airfoil_database.utilities.get_top_level_module import get_project_root
from workflow_manager import XFoilWorkflowManager

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('xfoil_resume_analysis.log'),
        logging.StreamHandler()
    ]
)

def main():
    # Initialize the database
    DB_FOLDER = os.path.join(get_project_root(), "airfoil_database")
    DB_NAME = "airfoils.db"
    db = AirfoilDatabase(db_name=DB_NAME, db_dir=DB_FOLDER)
    
    # Initialize workflow manager
    xfoil_path = "xfoil"
    workflow_manager = XFoilWorkflowManager(db, xfoil_path)
    
    # Define the same parameters as your original analysis
    reynolds_list = [1e5, 5e5, 1e6, 2e6]
    mach_list = [0.0, 0.1, 0.2, 0.3]
    alpha_list = list(range(-15, 21, 1))
    ncrit_list = [2.0, 9.0]
    
    # Check what's missing
    status = workflow_manager.get_analysis_status(reynolds_list, mach_list, alpha_list, ncrit_list)
    
    if status['total_conditions_missing'] == 0:
        logging.info("No missing conditions found. Analysis is complete!")
        return
    
    logging.info(f"Found {status['total_conditions_missing']} missing conditions across {status['airfoils_incomplete']} airfoils")
    
    # Resume analysis with only missing conditions
    results = workflow_manager.run_batch_analysis(
        reynolds_list=reynolds_list,
        mach_list=mach_list,
        alpha_list=alpha_list,
        ncrit_list=ncrit_list,
        airfoil_names=None,  # Check all airfoils
        max_workers=4,
        skip_existing=True,  # This is key - only run missing conditions
        batch_size=5  # Smaller batches for resume
    )
    
    logging.info(f"Resume analysis completed. Processed {results['processed_airfoils']} airfoils.")

if __name__ == "__main__":
    main()
