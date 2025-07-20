# check_analysis_status.py

import logging
import os
import pandas as pd
from airfoil_database.core.database import AirfoilDatabase
from airfoil_database.utilities.get_top_level_module import get_project_root
from workflow_manager import XFoilWorkflowManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Initialize the database
    DB_FOLDER = os.path.join(get_project_root(), "airfoil_database")
    DB_NAME = "airfoils.db"
    db = AirfoilDatabase(db_name=DB_NAME, db_dir=DB_FOLDER)
    
    # Initialize workflow manager
    workflow_manager = XFoilWorkflowManager(db)
    
    # Define analysis parameters to check
    reynolds_list = [1e5, 5e5, 1e6, 2e6]
    mach_list = [0.0, 0.1, 0.2, 0.3]
    alpha_list = list(range(-15, 21, 1))
    ncrit_list = [2.0, 9.0]
    
    # Get detailed status
    status = workflow_manager.get_analysis_status(reynolds_list, mach_list, alpha_list, ncrit_list)
    
    # Print detailed report
    print_detailed_status_report(status, workflow_manager, reynolds_list, mach_list, alpha_list, ncrit_list)

def print_detailed_status_report(status, workflow_manager, reynolds_list, mach_list, alpha_list, ncrit_list):
    """Print a detailed status report."""
    print("\n" + "="*80)
    print("XFOIL ANALYSIS STATUS REPORT")
    print("="*80)
    
    print(f"Analysis Parameters:")
    print(f"  Reynolds numbers: {len(reynolds_list)} values ({min(reynolds_list):.0e} to {max(reynolds_list):.0e})")
    print(f"  Mach numbers: {len(mach_list)} values ({min(mach_list)} to {max(mach_list)})")
    print(f"  Alpha range: {len(alpha_list)} values ({min(alpha_list)}° to {max(alpha_list)}°)")
    print(f"  Ncrit values: {ncrit_list}")
    print(f"  Conditions per airfoil: {len(reynolds_list) * len(mach_list) * len(alpha_list) * len(ncrit_list):,}")
    
    print(f"\nOverall Status:")
    print(f"  Total airfoils in database: {status['total_airfoils']:,}")
    print(f"  Airfoils with complete data: {status['airfoils_complete']:,}")
    print(f"  Airfoils with missing data: {status['airfoils_incomplete']:,}")
    print(f"  Airfoil completion rate: {status['completion_percentage']:.1f}%")
    
    print(f"\nCondition Status:")
    print(f"  Total possible conditions: {status['total_conditions_possible']:,}")
    print(f"  Existing conditions: {status['total_conditions_existing']:,}")
    print(f"  Missing conditions: {status['total_conditions_missing']:,}")
    print(f"  Condition completion rate: {status['condition_completion_percentage']:.1f}%")
    
    if status['airfoils_incomplete'] > 0:
        print(f"\nIncomplete Airfoils ({status['airfoils_incomplete']} total):")
        
        # Get missing conditions details
        missing_conditions = workflow_manager.database.find_missing_conditions(
            airfoil_names=None,
            reynolds_list=reynolds_list,
            mach_list=mach_list,
            alpha_list=alpha_list,
            ncrit_list=ncrit_list
        )
        
        # Create summary DataFrame
        incomplete_data = []
        for airfoil_name in status['incomplete_airfoils'][:20]:  # Show first 20
            missing_count = len(missing_conditions.get(airfoil_name, []))
            total_conditions = len(reynolds_list) * len(mach_list) * len(alpha_list) * len(ncrit_list)
            completion = ((total_conditions - missing_count) / total_conditions * 100)
            
            incomplete_data.append({
                'Airfoil': airfoil_name,
                'Missing': missing_count,
                'Total': total_conditions,
                'Complete %': f"{completion:.1f}%"
            })
        
        df = pd.DataFrame(incomplete_data)
        print(df.to_string(index=False))
        
        if status['airfoils_incomplete'] > 20:
            print(f"  ... and {status['airfoils_incomplete'] - 20} more airfoils")
    
    # Estimate analysis time
    if status['total_conditions_missing'] > 0:
        estimated_minutes = status['total_conditions_missing'] * 2 / 60  # 2 seconds per condition
        print(f"\nEstimated time to complete missing analysis: {estimated_minutes:.0f} minutes ({estimated_minutes/60:.1f} hours)")
    
    print("="*80)

if __name__ == "__main__":
    main()
