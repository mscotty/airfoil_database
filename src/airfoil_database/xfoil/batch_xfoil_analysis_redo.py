# batch_xfoil_analysis.py

import logging
import os
from airfoil_database.core.database import AirfoilDatabase
from airfoil_database.analysis.aero_analyzer import AeroAnalyzer
from airfoil_database.classes.XFoilRunner import XFoilRunner
from airfoil_database.utilities.get_top_level_module import get_project_root
from sqlmodel import Session, select
from airfoil_database.core.models import Airfoil, AeroCoeff

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_all_airfoil_names(db):
    """Get all airfoil names from the database."""
    with Session(db.engine) as session:
        statement = select(Airfoil.name)
        return session.exec(statement).all()

def check_missing_conditions(db, airfoil_name, reynolds_list, mach_list, alpha_list, ncrit_list):
    """Check if an airfoil has missing conditions for the specified parameters."""
    with Session(db.engine) as session:
        # Get existing conditions for this airfoil
        statement = select(AeroCoeff).where(AeroCoeff.name == airfoil_name)
        existing_coeffs = session.exec(statement).all()
        
        # Create set of existing conditions
        existing_conditions = set()
        for coeff in existing_coeffs:
            condition = (coeff.reynolds_number, coeff.mach, coeff.alpha, coeff.ncrit)
            existing_conditions.add(condition)
        
        # Check if all required conditions exist
        for reynolds in reynolds_list:
            for mach in mach_list:
                for alpha in alpha_list:
                    for ncrit in ncrit_list:
                        condition = (reynolds, mach, alpha, ncrit)
                        if condition not in existing_conditions:
                            return True  # Has missing conditions
        
        return False  # All conditions exist

def main():
    # Initialize the database
    DB_FOLDER = os.path.join(get_project_root(), "airfoil_database")
    DB_NAME = "airfoils.db"
    db_path = os.path.join(DB_FOLDER, DB_NAME)
    db = AirfoilDatabase(db_name=DB_NAME, db_dir=DB_FOLDER)
    
    # Initialize AeroAnalyzer
    analyzer = AeroAnalyzer(db)
    
    # Initialize XFoilRunner with AeroAnalyzer
    xfoil_path = "xfoil"  # Adjust this path as needed for your system
    xfoil = XFoilRunner(aero_analyzer=analyzer, xfoil_executable=xfoil_path)
    
    # Set analysis parameters (same as your original)
    reynolds_list = [1e6]  # Reynolds number
    mach_list = [0.2]  # Mach number
    alpha_list = [-10, -7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0]  # Angles of attack
    ncrit_list = [2.0]  # Ncrit parameter
    
    # Get all airfoils from database
    all_airfoils = get_all_airfoil_names(db)
    logging.info(f"Found {len(all_airfoils)} airfoils in database")
    
    # Filter to only airfoils that need analysis
    airfoils_to_process = []
    for airfoil_name in all_airfoils:
        if check_missing_conditions(db, airfoil_name, reynolds_list, mach_list, alpha_list, ncrit_list):
            airfoils_to_process.append(airfoil_name)
        else:
            logging.info(f"Skipping {airfoil_name} - all conditions already exist")
    
    logging.info(f"Found {len(airfoils_to_process)} airfoils that need analysis")
    
    if not airfoils_to_process:
        logging.info("No airfoils need processing. All data already exists!")
        return
    
    # Process each airfoil
    successful_count = 0
    failed_count = 0
    
    for i, airfoil_name in enumerate(airfoils_to_process, 1):
        try:
            logging.info(f"Processing airfoil {i}/{len(airfoils_to_process)}: {airfoil_name}")
            
            # Run the analysis (same as your original code)
            xfoil.run_analysis_parallel(
                airfoil_name=airfoil_name,
                reynolds_list=reynolds_list,
                mach_list=mach_list,
                alpha_list=alpha_list,
                ncrit_list=ncrit_list,
                max_workers=4  # Adjust based on your CPU cores
            )
            
            successful_count += 1
            logging.info(f"✓ Successfully completed {airfoil_name}")
            
        except Exception as e:
            failed_count += 1
            logging.error(f"✗ Failed to process {airfoil_name}: {e}")
            continue  # Continue with next airfoil
    
    # Final summary
    logging.info("="*60)
    logging.info("BATCH ANALYSIS COMPLETE")
    logging.info(f"Total airfoils processed: {successful_count + failed_count}")
    logging.info(f"Successful: {successful_count}")
    logging.info(f"Failed: {failed_count}")
    logging.info(f"Success rate: {(successful_count/(successful_count + failed_count)*100):.1f}%" if (successful_count + failed_count) > 0 else "N/A")
    logging.info("="*60)

if __name__ == "__main__":
    main()
