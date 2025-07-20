# batch_xfoil_analysis_smart.py

import logging
import os
from airfoil_database.core.database import AirfoilDatabase
from airfoil_database.analysis.aero_analyzer import AeroAnalyzer
from airfoil_database.classes.XFoilRunner import XFoilRunner
from airfoil_database.utilities.get_top_level_module import get_project_root
from sqlmodel import Session, select
from airfoil_database.core.models import Airfoil, AeroCoeff
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_all_airfoil_names(db):
    """Get all airfoil names from the database."""
    with Session(db.engine) as session:
        statement = select(Airfoil.name)
        return session.exec(statement).all()

def get_missing_conditions(db, airfoil_name, reynolds_list, mach_list, alpha_list, ncrit_list):
    """Get the specific missing conditions for an airfoil."""
    with Session(db.engine) as session:
        # Get existing conditions for this airfoil
        statement = select(AeroCoeff).where(AeroCoeff.name == airfoil_name)
        existing_coeffs = session.exec(statement).all()
        
        # Create set of existing conditions
        existing_conditions = set()
        for coeff in existing_coeffs:
            condition = (coeff.reynolds_number, coeff.mach, coeff.alpha, coeff.ncrit)
            existing_conditions.add(condition)
        
        # Find missing conditions grouped by (reynolds, mach, ncrit)
        missing_by_group = defaultdict(list)
        
        for reynolds in reynolds_list:
            for mach in mach_list:
                for ncrit in ncrit_list:
                    missing_alphas = []
                    for alpha in alpha_list:
                        condition = (reynolds, mach, alpha, ncrit)
                        if condition not in existing_conditions:
                            missing_alphas.append(alpha)
                    
                    if missing_alphas:
                        group_key = (reynolds, mach, ncrit)
                        missing_by_group[group_key] = missing_alphas
        
        return missing_by_group

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
    
    # Find airfoils with missing conditions
    airfoils_with_missing = {}
    for airfoil_name in all_airfoils:
        missing_conditions = get_missing_conditions(db, airfoil_name, reynolds_list, mach_list, alpha_list, ncrit_list)
        if missing_conditions:
            airfoils_with_missing[airfoil_name] = missing_conditions
        else:
            logging.info(f"Skipping {airfoil_name} - all conditions already exist")
    
    logging.info(f"Found {len(airfoils_with_missing)} airfoils that need analysis")
    
    if not airfoils_with_missing:
        logging.info("No airfoils need processing. All data already exists!")
        return
    
    # Process each airfoil with only missing conditions
    successful_count = 0
    failed_count = 0
    
    for i, (airfoil_name, missing_conditions) in enumerate(airfoils_with_missing.items(), 1):
        try:
            logging.info(f"Processing airfoil {i}/{len(airfoils_with_missing)}: {airfoil_name}")
            
            # Process each group of missing conditions
            for (reynolds, mach, ncrit), missing_alphas in missing_conditions.items():
                logging.info(f"  Running missing conditions: Re={reynolds}, M={mach}, ncrit={ncrit}, {len(missing_alphas)} alphas")
                
                # Run analysis for this specific group
                xfoil.run_analysis_parallel(
                    airfoil_name=airfoil_name,
                    reynolds_list=[reynolds],
                    mach_list=[mach],
                    alpha_list=missing_alphas,
                    ncrit_list=[ncrit],
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
