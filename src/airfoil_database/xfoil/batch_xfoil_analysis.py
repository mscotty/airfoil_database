# batch_xfoil_analysis.py

import logging
import os
from airfoil_database.core.database import AirfoilDatabase
from airfoil_database.utilities.get_top_level_module import get_project_root
from airfoil_database.xfoil.workflow_manager import XFoilWorkflowManager

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('xfoil_batch_analysis.log'),
        logging.StreamHandler()
    ]
)

def run_analysis():
    """Main analysis function."""
    # Initialize the database
    DB_FOLDER = os.path.join(get_project_root(), "airfoil_database")
    DB_NAME = "airfoils.db"
    db_path = os.path.join(DB_FOLDER, DB_NAME)
    db = AirfoilDatabase(db_name=DB_NAME, db_dir=DB_FOLDER)
    
    # Initialize workflow manager
    xfoil_path = "xfoil"  # Adjust this path as needed for your system
    workflow_manager = XFoilWorkflowManager(db, xfoil_path)
    
    # Define analysis parameters
    reynolds_list = [1e5, 5e5, 1e6, 2e6]
    mach_list = [0.0, 0.1, 0.2, 0.3]
    alpha_list = list(range(-15, 21, 1))
    ncrit_list = [2.0, 9.0]
    
    # Run the batch analysis
    logging.info("Starting batch XFoil analysis...")
    results = workflow_manager.run_batch_analysis(
        reynolds_list=reynolds_list,
        mach_list=mach_list,
        alpha_list=alpha_list,
        ncrit_list=ncrit_list,
        airfoil_names=None,
        max_workers=4,
        skip_existing=True,
        batch_size=10
    )
    
    # Print results
    logging.info(f"Analysis complete: {results['successful_airfoils']}/{results['processed_airfoils']} successful")

if __name__ == '__main__':
    # This is CRITICAL for Windows multiprocessing
    import multiprocessing
    multiprocessing.freeze_support()  # Add this line for Windows compatibility
    
    run_analysis()
