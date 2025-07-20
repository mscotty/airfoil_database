# workflow_manager.py

import logging
import time
from typing import List, Dict, Any, Optional
from collections import defaultdict
import os
from airfoil_database.core.database import AirfoilDatabase
from airfoil_database.analysis.aero_analyzer import AeroAnalyzer
from airfoil_database.classes.XFoilRunner import XFoilRunner

import sys
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

if sys.platform.startswith('win'):
    # Set multiprocessing start method for Windows
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set


class XFoilWorkflowManager:
    """
    Manages batch XFoil analysis workflows across multiple airfoils.
    """
    
    def __init__(self, database: AirfoilDatabase, xfoil_executable: str = "xfoil"):
        """
        Initialize the workflow manager.
        
        Args:
            database: AirfoilDatabase instance
            xfoil_executable: Path to XFoil executable
        """
        self.database = database
        self.analyzer = AeroAnalyzer(database)
        self.xfoil_runner = XFoilRunner(self.analyzer, xfoil_executable)
        
    def run_batch_analysis(self, 
                          reynolds_list: List[float],
                          mach_list: List[float], 
                          alpha_list: List[float],
                          ncrit_list: List[float],
                          airfoil_names: Optional[List[str]] = None,
                          max_workers: Optional[int] = None,
                          skip_existing: bool = True,
                          batch_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Run XFoil analysis for multiple airfoils, checking for existing data.
        
        Args:
            reynolds_list: List of Reynolds numbers
            mach_list: List of Mach numbers
            alpha_list: List of angles of attack  
            ncrit_list: List of ncrit values
            airfoil_names: List of specific airfoil names to analyze. If None, analyzes all.
            max_workers: Number of parallel workers
            skip_existing: If True, skip conditions that already exist in database
            batch_size: Number of airfoils to process in each batch (for memory management)
            
        Returns:
            dict: Summary of analysis results
        """
        start_time = time.time()
        logging.info("Starting batch XFoil analysis workflow")
        
        # Get airfoils to analyze
        if airfoil_names is None:
            airfoil_names = self._get_all_airfoil_names()
        
        total_airfoils = len(airfoil_names)
        logging.info(f"Found {total_airfoils} airfoils to analyze")
        
        if skip_existing:
            # Find missing conditions for all airfoils
            logging.info("Checking for existing data...")
            missing_conditions = self.database.find_missing_conditions(
                airfoil_names=airfoil_names,
                reynolds_list=reynolds_list,
                mach_list=mach_list, 
                alpha_list=alpha_list,
                ncrit_list=ncrit_list
            )
            
            # Filter to only airfoils with missing data
            airfoils_to_process = list(missing_conditions.keys())
            logging.info(f"Found {len(airfoils_to_process)} airfoils with missing data")
        else:
            airfoils_to_process = airfoil_names
            missing_conditions = {}
        
        if not airfoils_to_process:
            logging.info("No airfoils need processing. All data exists.")
            return {
                'total_airfoils': total_airfoils,
                'processed_airfoils': 0,
                'successful_airfoils': 0,
                'failed_airfoils': 0,
                'duration': time.time() - start_time
            }
        
        # Process airfoils in batches if specified
        if batch_size:
            return self._run_batch_analysis_chunked(
                airfoils_to_process, missing_conditions, reynolds_list, 
                mach_list, alpha_list, ncrit_list, max_workers, batch_size, start_time
            )
        else:
            return self._run_batch_analysis_all(
                airfoils_to_process, missing_conditions, reynolds_list,
                mach_list, alpha_list, ncrit_list, max_workers, start_time
            )
    
    def _run_batch_analysis_all(self, airfoils_to_process, missing_conditions, 
                               reynolds_list, mach_list, alpha_list, ncrit_list, 
                               max_workers, start_time):
        """Run analysis for all airfoils at once."""
        successful_airfoils = []
        failed_airfoils = []
        
        for i, airfoil_name in enumerate(airfoils_to_process, 1):
            logging.info(f"Processing airfoil {i}/{len(airfoils_to_process)}: {airfoil_name}")
            
            try:
                if missing_conditions and airfoil_name in missing_conditions:
                    # Run only missing conditions
                    self._run_missing_conditions(airfoil_name, missing_conditions[airfoil_name], max_workers)
                else:
                    # Run all conditions
                    self.xfoil_runner.run_analysis_parallel(
                        airfoil_name=airfoil_name,
                        reynolds_list=reynolds_list,
                        mach_list=mach_list,
                        alpha_list=alpha_list,
                        ncrit_list=ncrit_list,
                        max_workers=max_workers
                    )
                
                successful_airfoils.append(airfoil_name)
                logging.info(f"✓ Successfully completed {airfoil_name}")
                
            except Exception as e:
                logging.error(f"✗ Failed to process {airfoil_name}: {e}")
                failed_airfoils.append(airfoil_name)
        
        duration = time.time() - start_time
        
        return {
            'total_airfoils': len(airfoils_to_process),
            'processed_airfoils': len(successful_airfoils) + len(failed_airfoils),
            'successful_airfoils': len(successful_airfoils),
            'failed_airfoils': len(failed_airfoils),
            'successful_list': successful_airfoils,
            'failed_list': failed_airfoils,
            'duration': duration
        }
    
    def _run_batch_analysis_chunked(self, airfoils_to_process, missing_conditions,
                                   reynolds_list, mach_list, alpha_list, ncrit_list,
                                   max_workers, batch_size, start_time):
        """Run analysis in chunks for better memory management."""
        successful_airfoils = []
        failed_airfoils = []
        
        # Process in chunks
        for chunk_start in range(0, len(airfoils_to_process), batch_size):
            chunk_end = min(chunk_start + batch_size, len(airfoils_to_process))
            chunk = airfoils_to_process[chunk_start:chunk_end]
            
            logging.info(f"Processing chunk {chunk_start//batch_size + 1}: "
                        f"airfoils {chunk_start+1}-{chunk_end}")
            
            for airfoil_name in chunk:
                try:
                    if missing_conditions and airfoil_name in missing_conditions:
                        self._run_missing_conditions(airfoil_name, missing_conditions[airfoil_name], max_workers)
                    else:
                        self.xfoil_runner.run_analysis_parallel(
                            airfoil_name=airfoil_name,
                            reynolds_list=reynolds_list,
                            mach_list=mach_list,
                            alpha_list=alpha_list,
                            ncrit_list=ncrit_list,
                            max_workers=max_workers
                        )
                    
                    successful_airfoils.append(airfoil_name)
                    logging.info(f"✓ Successfully completed {airfoil_name}")
                    
                except Exception as e:
                    logging.error(f"✗ Failed to process {airfoil_name}: {e}")
                    failed_airfoils.append(airfoil_name)
        
        duration = time.time() - start_time
        
        return {
            'total_airfoils': len(airfoils_to_process),
            'processed_airfoils': len(successful_airfoils) + len(failed_airfoils),
            'successful_airfoils': len(successful_airfoils),
            'failed_airfoils': len(failed_airfoils),
            'successful_list': successful_airfoils,
            'failed_list': failed_airfoils,
            'duration': duration
        }
    
    def _run_missing_conditions(self, airfoil_name: str, missing_conditions: List[Dict], max_workers: Optional[int]):
        """Run XFoil analysis for specific missing conditions."""
        # Group missing conditions by reynolds, mach, ncrit
        condition_groups = defaultdict(list)
        
        for condition in missing_conditions:
            key = (condition['reynolds'], condition['mach'], condition['ncrit'])
            condition_groups[key].append(condition['alpha'])
        
        # Run each group
        for (reynolds, mach, ncrit), alphas in condition_groups.items():
            logging.info(f"Running missing conditions for {airfoil_name}: "
                        f"Re={reynolds}, M={mach}, ncrit={ncrit}, {len(alphas)} alphas")
            
            self.xfoil_runner.run_analysis_parallel(
                airfoil_name=airfoil_name,
                reynolds_list=[reynolds],
                mach_list=[mach],
                alpha_list=alphas,
                ncrit_list=[ncrit],
                max_workers=max_workers
            )
    
    def _get_all_airfoil_names(self) -> List[str]:
        """Get all airfoil names from database."""
        from sqlmodel import Session, select
        from airfoil_database.core.models import Airfoil
        
        with Session(self.database.engine) as session:
            statement = select(Airfoil.name)
            return session.exec(statement).all()
    
    def get_analysis_status(self, reynolds_list: List[float], mach_list: List[float], 
                           alpha_list: List[float], ncrit_list: List[float]) -> Dict[str, Any]:
        """
        Get status of analysis completion across all airfoils.
        
        Returns:
            dict: Status information including completion percentages
        """
        airfoil_names = self._get_all_airfoil_names()
        total_airfoils = len(airfoil_names)
        total_conditions_per_airfoil = len(reynolds_list) * len(mach_list) * len(alpha_list) * len(ncrit_list)
        
        missing_conditions = self.database.find_missing_conditions(
            airfoil_names=airfoil_names,
            reynolds_list=reynolds_list,
            mach_list=mach_list,
            alpha_list=alpha_list,
            ncrit_list=ncrit_list
        )
        
        airfoils_complete = total_airfoils - len(missing_conditions)
        total_missing_conditions = sum(len(conditions) for conditions in missing_conditions.values())
        total_possible_conditions = total_airfoils * total_conditions_per_airfoil
        total_existing_conditions = total_possible_conditions - total_missing_conditions
        
        return {
            'total_airfoils': total_airfoils,
            'airfoils_complete': airfoils_complete,
            'airfoils_incomplete': len(missing_conditions),
            'completion_percentage': (airfoils_complete / total_airfoils * 100) if total_airfoils > 0 else 0,
            'total_conditions_possible': total_possible_conditions,
            'total_conditions_existing': total_existing_conditions,
            'total_conditions_missing': total_missing_conditions,
            'condition_completion_percentage': (total_existing_conditions / total_possible_conditions * 100) if total_possible_conditions > 0 else 0,
            'incomplete_airfoils': list(missing_conditions.keys())
        }
    
    def run_specific_airfoils(self, airfoil_names: List[str],
                             reynolds_list: List[float], mach_list: List[float],
                             alpha_list: List[float], ncrit_list: List[float],
                             max_workers: Optional[int] = None,
                             skip_existing: bool = True) -> Dict[str, Any]:
        """
        Run analysis for specific airfoils only.
        
        Args:
            airfoil_names: List of specific airfoil names to analyze
            reynolds_list: List of Reynolds numbers
            mach_list: List of Mach numbers
            alpha_list: List of angles of attack
            ncrit_list: List of ncrit values
            max_workers: Number of parallel workers
            skip_existing: If True, skip conditions that already exist
            
        Returns:
            dict: Analysis results summary
        """
        return self.run_batch_analysis(
            reynolds_list=reynolds_list,
            mach_list=mach_list,
            alpha_list=alpha_list,
            ncrit_list=ncrit_list,
            airfoil_names=airfoil_names,
            max_workers=max_workers,
            skip_existing=skip_existing
        )
    
    def get_airfoil_completion_details(self, reynolds_list: List[float], mach_list: List[float],
                                      alpha_list: List[float], ncrit_list: List[float]) -> Dict[str, Dict]:
        """
        Get detailed completion status for each airfoil.
        
        Returns:
            dict: Detailed completion info for each airfoil
        """
        airfoil_names = self._get_all_airfoil_names()
        total_conditions_per_airfoil = len(reynolds_list) * len(mach_list) * len(alpha_list) * len(ncrit_list)
        
        missing_conditions = self.database.find_missing_conditions(
            airfoil_names=airfoil_names,
            reynolds_list=reynolds_list,
            mach_list=mach_list,
            alpha_list=alpha_list,
            ncrit_list=ncrit_list
        )
        
        airfoil_details = {}
        
        for airfoil_name in airfoil_names:
            missing_count = len(missing_conditions.get(airfoil_name, []))
            existing_count = total_conditions_per_airfoil - missing_count
            completion_percentage = (existing_count / total_conditions_per_airfoil * 100) if total_conditions_per_airfoil > 0 else 0
            
            airfoil_details[airfoil_name] = {
                'total_conditions': total_conditions_per_airfoil,
                'existing_conditions': existing_count,
                'missing_conditions': missing_count,
                'completion_percentage': completion_percentage,
                'is_complete': missing_count == 0,
                'missing_details': missing_conditions.get(airfoil_name, [])
            }
        
        return airfoil_details
    
    def export_missing_conditions_report(self, reynolds_list: List[float], mach_list: List[float],
                                        alpha_list: List[float], ncrit_list: List[float],
                                        output_file: str = "missing_conditions_report.csv"):
        """
        Export a detailed report of missing conditions to CSV.
        
        Args:
            reynolds_list: List of Reynolds numbers
            mach_list: List of Mach numbers
            alpha_list: List of angles of attack
            ncrit_list: List of ncrit values
            output_file: Output CSV file path
        """
        import csv
        
        missing_conditions = self.database.find_missing_conditions(
            airfoil_names=None,
            reynolds_list=reynolds_list,
            mach_list=mach_list,
            alpha_list=alpha_list,
            ncrit_list=ncrit_list
        )
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['airfoil_name', 'reynolds', 'mach', 'alpha', 'ncrit']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for airfoil_name, conditions in missing_conditions.items():
                for condition in conditions:
                    writer.writerow({
                        'airfoil_name': airfoil_name,
                        'reynolds': condition['reynolds'],
                        'mach': condition['mach'],
                        'alpha': condition['alpha'],
                        'ncrit': condition['ncrit']
                    })
        
        total_missing = sum(len(conditions) for conditions in missing_conditions.values())
        logging.info(f"Exported {total_missing} missing conditions to {output_file}")
    
    def run_priority_analysis(self, reynolds_list: List[float], mach_list: List[float],
                             alpha_list: List[float], ncrit_list: List[float],
                             priority_airfoils: List[str], max_workers: Optional[int] = None) -> Dict[str, Any]:
        """
        Run analysis with priority for specific airfoils first.
        
        Args:
            reynolds_list: List of Reynolds numbers
            mach_list: List of Mach numbers
            alpha_list: List of angles of attack
            ncrit_list: List of ncrit values
            priority_airfoils: List of airfoil names to process first
            max_workers: Number of parallel workers
            
        Returns:
            dict: Combined results from priority and remaining airfoils
        """
        start_time = time.time()
        
        # First, run priority airfoils
        logging.info(f"Starting priority analysis for {len(priority_airfoils)} airfoils")
        priority_results = self.run_batch_analysis(
            reynolds_list=reynolds_list,
            mach_list=mach_list,
            alpha_list=alpha_list,
            ncrit_list=ncrit_list,
            airfoil_names=priority_airfoils,
            max_workers=max_workers,
            skip_existing=True
        )
        
        # Get remaining airfoils
        all_airfoils = set(self._get_all_airfoil_names())
        priority_set = set(priority_airfoils)
        remaining_airfoils = list(all_airfoils - priority_set)
        
        if remaining_airfoils:
            logging.info(f"Starting analysis for remaining {len(remaining_airfoils)} airfoils")
            remaining_results = self.run_batch_analysis(
                reynolds_list=reynolds_list,
                mach_list=mach_list,
                alpha_list=alpha_list,
                ncrit_list=ncrit_list,
                airfoil_names=remaining_airfoils,
                max_workers=max_workers,
                skip_existing=True
            )
            
            # Combine results
            combined_results = {
                'total_airfoils': priority_results['total_airfoils'] + remaining_results['total_airfoils'],
                'processed_airfoils': priority_results['processed_airfoils'] + remaining_results['processed_airfoils'],
                'successful_airfoils': priority_results['successful_airfoils'] + remaining_results['successful_airfoils'],
                'failed_airfoils': priority_results['failed_airfoils'] + remaining_results['failed_airfoils'],
                'successful_list': priority_results.get('successful_list', []) + remaining_results.get('successful_list', []),
                'failed_list': priority_results.get('failed_list', []) + remaining_results.get('failed_list', []),
                'duration': time.time() - start_time,
                'priority_results': priority_results,
                'remaining_results': remaining_results
            }
        else:
            combined_results = priority_results
            combined_results['duration'] = time.time() - start_time
        
        return combined_results
    
    def validate_airfoil_data(self, airfoil_names: Optional[List[str]] = None) -> Dict[str, List[str]]:
        """
        Validate that airfoils have the required point cloud data for XFoil analysis.
        
        Args:
            airfoil_names: List of airfoil names to validate. If None, validates all.
            
        Returns:
            dict: Contains 'valid' and 'invalid' lists of airfoil names
        """
        if airfoil_names is None:
            airfoil_names = self._get_all_airfoil_names()
        
        valid_airfoils = []
        invalid_airfoils = []
        
        for airfoil_name in airfoil_names:
            airfoil_data = self.database.get_airfoil_data(airfoil_name)
            
            if airfoil_data and airfoil_data[1]:  # Check if pointcloud exists
                pointcloud = airfoil_data[1].strip()
                if pointcloud and len(pointcloud.split('\n')) >= 10:  # Minimum points check
                    valid_airfoils.append(airfoil_name)
                else:
                    invalid_airfoils.append(airfoil_name)
                    logging.warning(f"Airfoil {airfoil_name} has insufficient point data")
            else:
                invalid_airfoils.append(airfoil_name)
                logging.warning(f"Airfoil {airfoil_name} has no point cloud data")
        
        logging.info(f"Validation complete: {len(valid_airfoils)} valid, {len(invalid_airfoils)} invalid airfoils")
        
        return {
            'valid': valid_airfoils,
            'invalid': invalid_airfoils,
            'total_checked': len(airfoil_names)
        }
    
    def estimate_analysis_time(self, reynolds_list: List[float], mach_list: List[float],
                              alpha_list: List[float], ncrit_list: List[float],
                              airfoil_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Estimate the time required for analysis based on missing conditions.
        
        Args:
            reynolds_list: List of Reynolds numbers
            mach_list: List of Mach numbers
            alpha_list: List of angles of attack
            ncrit_list: List of ncrit values
            airfoil_names: List of airfoil names. If None, estimates for all airfoils.
            
        Returns:
            dict: Time estimates in various units
        """
        missing_conditions = self.database.find_missing_conditions(
            airfoil_names=airfoil_names,
            reynolds_list=reynolds_list,
            mach_list=mach_list,
            alpha_list=alpha_list,
            ncrit_list=ncrit_list
        )
        
        total_missing = sum(len(conditions) for conditions in missing_conditions.values())
        
        # Estimates based on typical XFoil performance
        # These can be adjusted based on your system performance
        seconds_per_condition = 2.0  # Average time per alpha point
        
        total_seconds = total_missing * seconds_per_condition
        total_minutes = total_seconds / 60
        total_hours = total_minutes / 60
        
        return {
            'total_conditions': total_missing,
            'estimated_seconds': total_seconds,
            'estimated_minutes': total_minutes,
            'estimated_hours': total_hours,
            'estimated_time_str': f"{int(total_hours)}h {int(total_minutes % 60)}m" if total_hours >= 1 else f"{int(total_minutes)}m {int(total_seconds % 60)}s"
        }
    
    def cleanup_failed_results(self):
        """
        Clean up any partial or failed results in the database.
        This removes entries where cl, cd, or cm are None.
        """
        from sqlmodel import Session, select, delete
        from airfoil_database.core.models import AeroCoeff
        
        try:
            with Session(self.database.engine) as session:
                # Find records with null coefficients
                statement = select(AeroCoeff).where(
                    (AeroCoeff.cl.is_(None)) | 
                    (AeroCoeff.cd.is_(None)) | 
                    (AeroCoeff.cm.is_(None))
                )
                failed_records = session.exec(statement).all()
                
                if failed_records:
                    logging.info(f"Found {len(failed_records)} failed/incomplete records")
                    
                    # Delete failed records
                    for record in failed_records:
                        session.delete(record)
                    
                    session.commit()
                    logging.info(f"Cleaned up {len(failed_records)} failed records")
                else:
                    logging.info("No failed records found")
                    
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics from the database.
        
        Returns:
            dict: Performance statistics
        """
        from sqlmodel import Session, select, func
        from airfoil_database.core.models import AeroCoeff, Airfoil
        
        try:
            with Session(self.database.engine) as session:
                # Count total records
                total_coeffs = session.exec(select(func.count(AeroCoeff.id))).first()
                total_airfoils = session.exec(select(func.count(Airfoil.id))).first()
                
                # Count unique conditions
                unique_reynolds = session.exec(select(func.count(func.distinct(AeroCoeff.reynolds_number)))).first()
                unique_mach = session.exec(select(func.count(func.distinct(AeroCoeff.mach)))).first()
                unique_alpha = session.exec(select(func.count(func.distinct(AeroCoeff.alpha)))).first()
                unique_ncrit = session.exec(select(func.count(func.distinct(AeroCoeff.ncrit)))).first()
                
                # Count airfoils with data
                airfoils_with_data = session.exec(
                    select(func.count(func.distinct(AeroCoeff.name)))
                ).first()
                
                # Get coefficient ranges
                cl_stats = session.exec(
                    select(func.min(AeroCoeff.cl), func.max(AeroCoeff.cl), func.avg(AeroCoeff.cl))
                ).first()
                
                cd_stats = session.exec(
                    select(func.min(AeroCoeff.cd), func.max(AeroCoeff.cd), func.avg(AeroCoeff.cd))
                ).first()
                
                cm_stats = session.exec(
                    select(func.min(AeroCoeff.cm), func.max(AeroCoeff.cm), func.avg(AeroCoeff.cm))
                ).first()
                
                return {
                    'total_coefficient_records': total_coeffs or 0,
                    'total_airfoils_in_db': total_airfoils or 0,
                    'airfoils_with_analysis_data': airfoils_with_data or 0,
                    'airfoils_without_data': (total_airfoils or 0) - (airfoils_with_data or 0),
                    'unique_reynolds_numbers': unique_reynolds or 0,
                    'unique_mach_numbers': unique_mach or 0,
                    'unique_alpha_values': unique_alpha or 0,
                    'unique_ncrit_values': unique_ncrit or 0,
                    'cl_range': {
                        'min': cl_stats[0] if cl_stats[0] is not None else 0,
                        'max': cl_stats[1] if cl_stats[1] is not None else 0,
                        'avg': cl_stats[2] if cl_stats[2] is not None else 0
                    },
                    'cd_range': {
                        'min': cd_stats[0] if cd_stats[0] is not None else 0,
                        'max': cd_stats[1] if cd_stats[1] is not None else 0,
                        'avg': cd_stats[2] if cd_stats[2] is not None else 0
                    },
                    'cm_range': {
                        'min': cm_stats[0] if cm_stats[0] is not None else 0,
                        'max': cm_stats[1] if cm_stats[1] is not None else 0,
                        'avg': cm_stats[2] if cm_stats[2] is not None else 0
                    }
                }
                
        except Exception as e:
            logging.error(f"Error getting performance stats: {e}")
            return {}
    
    def find_problematic_airfoils(self, reynolds_list: List[float], mach_list: List[float],
                                 alpha_list: List[float], ncrit_list: List[float],
                                 min_success_rate: float = 0.8) -> Dict[str, Any]:
        """
        Find airfoils that have consistently low success rates in XFoil analysis.
        
        Args:
            reynolds_list: List of Reynolds numbers
            mach_list: List of Mach numbers
            alpha_list: List of angles of attack
            ncrit_list: List of ncrit values
            min_success_rate: Minimum success rate threshold (0.0 to 1.0)
            
        Returns:
            dict: Information about problematic airfoils
        """
        airfoil_details = self.get_airfoil_completion_details(
            reynolds_list, mach_list, alpha_list, ncrit_list
        )
        
        problematic_airfoils = []
        good_airfoils = []
        
        for airfoil_name, details in airfoil_details.items():
            success_rate = details['completion_percentage'] / 100.0
            
            if success_rate < min_success_rate and details['existing_conditions'] > 0:
                problematic_airfoils.append({
                    'name': airfoil_name,
                    'success_rate': success_rate,
                    'existing_conditions': details['existing_conditions'],
                    'missing_conditions': details['missing_conditions'],
                    'completion_percentage': details['completion_percentage']
                })
            elif success_rate >= min_success_rate:
                good_airfoils.append(airfoil_name)
        
        # Sort by success rate (worst first)
        problematic_airfoils.sort(key=lambda x: x['success_rate'])
        
        return {
            'problematic_airfoils': problematic_airfoils,
            'good_airfoils': good_airfoils,
            'total_problematic': len(problematic_airfoils),
            'total_good': len(good_airfoils),
            'threshold_used': min_success_rate
        }
    
    def retry_failed_conditions(self, reynolds_list: List[float], mach_list: List[float],
                               alpha_list: List[float], ncrit_list: List[float],
                               max_workers: Optional[int] = None,
                               retry_limit: int = 3) -> Dict[str, Any]:
        """
        Retry analysis for conditions that previously failed.
        
        Args:
            reynolds_list: List of Reynolds numbers
            mach_list: List of Mach numbers
            alpha_list: List of angles of attack
            ncrit_list: List of ncrit values
            max_workers: Number of parallel workers
            retry_limit: Maximum number of retry attempts
            
        Returns:
            dict: Results of retry analysis
        """
        logging.info(f"Starting retry analysis (attempt limit: {retry_limit})")
        
        retry_results = {
            'attempts': 0,
            'initial_missing': 0,
            'final_missing': 0,
            'recovered_conditions': 0,
            'permanently_failed': 0
        }
        
        for attempt in range(retry_limit):
            retry_results['attempts'] = attempt + 1
            
            # Check current missing conditions
            missing_conditions = self.database.find_missing_conditions(
                airfoil_names=None,
                reynolds_list=reynolds_list,
                mach_list=mach_list,
                alpha_list=alpha_list,
                ncrit_list=ncrit_list
            )
            
            current_missing = sum(len(conditions) for conditions in missing_conditions.values())
            
            if attempt == 0:
                retry_results['initial_missing'] = current_missing
            
            if current_missing == 0:
                logging.info(f"All conditions completed after {attempt + 1} attempts")
                break
                
            logging.info(f"Retry attempt {attempt + 1}: {current_missing} missing conditions")
            
            # Run analysis for missing conditions only
            results = self.run_batch_analysis(
                reynolds_list=reynolds_list,
                mach_list=mach_list,
                alpha_list=alpha_list,
                ncrit_list=ncrit_list,
                airfoil_names=None,
                max_workers=max_workers,
                skip_existing=True
            )
            
            # Check if we made progress
            new_missing_conditions = self.database.find_missing_conditions(
                airfoil_names=None,
                reynolds_list=reynolds_list,
                mach_list=mach_list,
                alpha_list=alpha_list,
                ncrit_list=ncrit_list
            )
            
            new_missing_count = sum(len(conditions) for conditions in new_missing_conditions.values())
            
            if new_missing_count == current_missing:
                logging.warning(f"No progress made in attempt {attempt + 1}, stopping retries")
                break
        
        # Final statistics
        retry_results['final_missing'] = new_missing_count
        retry_results['recovered_conditions'] = retry_results['initial_missing'] - retry_results['final_missing']
        retry_results['permanently_failed'] = retry_results['final_missing']
        
        logging.info(f"Retry analysis complete: recovered {retry_results['recovered_conditions']} conditions")
        
        return retry_results
    
    def export_analysis_summary(self, reynolds_list: List[float], mach_list: List[float],
                               alpha_list: List[float], ncrit_list: List[float],
                               output_file: str = "analysis_summary.json"):
        """
        Export a comprehensive analysis summary to JSON.
        
        Args:
            reynolds_list: List of Reynolds numbers
            mach_list: List of Mach numbers
            alpha_list: List of angles of attack
            ncrit_list: List of ncrit values
            output_file: Output JSON file path
        """
        import json
        from datetime import datetime
        
        # Gather all analysis information
        status = self.get_analysis_status(reynolds_list, mach_list, alpha_list, ncrit_list)
        performance_stats = self.get_performance_stats()
        airfoil_details = self.get_airfoil_completion_details(reynolds_list, mach_list, alpha_list, ncrit_list)
        time_estimate = self.estimate_analysis_time(reynolds_list, mach_list, alpha_list, ncrit_list)
        validation_results = self.validate_airfoil_data()
        problematic = self.find_problematic_airfoils(reynolds_list, mach_list, alpha_list, ncrit_list)
        
        summary = {
            'export_timestamp': datetime.now().isoformat(),
            'analysis_parameters': {
                'reynolds_numbers': reynolds_list,
                'mach_numbers': mach_list,
                'alpha_values': alpha_list,
                'ncrit_values': ncrit_list,
                'total_conditions_per_airfoil': len(reynolds_list) * len(mach_list) * len(alpha_list) * len(ncrit_list)
            },
            'overall_status': status,
            'performance_statistics': performance_stats,
            'time_estimates': time_estimate,
            'validation_results': validation_results,
            'problematic_airfoils': problematic,
            'detailed_airfoil_status': {
                name: {
                    'completion_percentage': details['completion_percentage'],
                    'missing_conditions': details['missing_conditions'],
                    'is_complete': details['is_complete']
                }
                for name, details in airfoil_details.items()
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logging.info(f"Analysis summary exported to {output_file}")
    
    def run_test_analysis(self, test_airfoils: Optional[List[str]] = None,
                         test_conditions: Optional[Dict] = None,
                         max_workers: int = 2) -> Dict[str, Any]:
        """
        Run a small test analysis to verify XFoil setup and database connectivity.
        
        Args:
            test_airfoils: List of airfoil names to test. If None, uses first few airfoils.
            test_conditions: Dictionary with test conditions. If None, uses minimal conditions.
            max_workers: Number of workers for test
            
        Returns:
            dict: Test results
        """
        logging.info("Starting test analysis to verify setup")
        
        # Default test conditions
        if test_conditions is None:
            test_conditions = {
                'reynolds_list': [1e6],
                'mach_list': [0.1],
                'alpha_list': [0.0, 5.0],
                'ncrit_list': [2.0]
            }
        
        # Get test airfoils
        if test_airfoils is None:
            all_airfoils = self._get_all_airfoil_names()
            test_airfoils = all_airfoils[:3] if len(all_airfoils) >= 3 else all_airfoils
        
        if not test_airfoils:
            return {'success': False, 'error': 'No airfoils available for testing'}
        
        # Validate test airfoils have data
        validation = self.validate_airfoil_data(test_airfoils)
        valid_test_airfoils = validation['valid']
        
        if not valid_test_airfoils:
            return {'success': False, 'error': 'No valid airfoils available for testing'}
        
        # Use only first valid airfoil for quick test
        test_airfoil = valid_test_airfoils[0]
        
        try:
            start_time = time.time()
            
            # Run test analysis
            results = self.run_batch_analysis(
                reynolds_list=test_conditions['reynolds_list'],
                mach_list=test_conditions['mach_list'],
                alpha_list=test_conditions['alpha_list'],
                ncrit_list=test_conditions['ncrit_list'],
                airfoil_names=[test_airfoil],
                max_workers=max_workers,
                skip_existing=False  # Force run for test
            )
            
            test_duration = time.time() - start_time
            
            # Check if any results were obtained
            test_success = results['successful_airfoils'] > 0
            
            return {
                'success': test_success,
                'test_airfoil': test_airfoil,
                'test_conditions': test_conditions,
                'duration_seconds': test_duration,
                'results': results,
                'message': 'Test completed successfully' if test_success else 'Test failed - no successful results'
            }
            
        except Exception as e:
            logging.error(f"Test analysis failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'test_airfoil': test_airfoil,
                'test_conditions': test_conditions
            }
    
    def optimize_batch_size(self, reynolds_list: List[float], mach_list: List[float],
                           alpha_list: List[float], ncrit_list: List[float],
                           target_memory_gb: float = 4.0) -> int:
        """
        Estimate optimal batch size based on available memory and analysis parameters.
        
        Args:
            reynolds_list: List of Reynolds numbers
            mach_list: List of Mach numbers
            alpha_list: List of angles of attack
            ncrit_list: List of ncrit values
            target_memory_gb: Target memory usage in GB
            
        Returns:
            int: Recommended batch size
        """
        try:
            import psutil
            
            # Get system memory info
            memory = psutil.virtual_memory()
            available_memory_gb = memory.available / (1024**3)
            
            # Use the smaller of target or available memory
            usable_memory_gb = min(target_memory_gb, available_memory_gb * 0.8)  # Use 80% of available
            
        except ImportError:
            logging.warning("psutil not available, using default memory estimate")
            usable_memory_gb = target_memory_gb
        
        # Estimate memory usage per airfoil
        conditions_per_airfoil = len(reynolds_list) * len(mach_list) * len(alpha_list) * len(ncrit_list)
        
        # Rough estimates:
        # - Each XFoil process: ~50MB
        # - Database operations: ~10MB per airfoil
        # - Temporary files: ~5MB per airfoil
        memory_per_airfoil_mb = 65
        
        # Calculate batch size
        usable_memory_mb = usable_memory_gb * 1024
        estimated_batch_size = int(usable_memory_mb / memory_per_airfoil_mb)
        
        # Apply reasonable limits
        min_batch_size = 1
        max_batch_size = 50  # Avoid overwhelming the system
        
        optimal_batch_size = max(min_batch_size, min(estimated_batch_size, max_batch_size))
        
        logging.info(f"Memory optimization: {usable_memory_gb:.1f}GB available, "
                    f"recommended batch size: {optimal_batch_size}")
        
        return optimal_batch_size
    
    def run_incremental_analysis(self, reynolds_list: List[float], mach_list: List[float],
                                alpha_list: List[float], ncrit_list: List[float],
                                max_workers: Optional[int] = None,
                                save_progress_every: int = 10) -> Dict[str, Any]:
        """
        Run analysis with incremental progress saving and detailed logging.
        
        Args:
            reynolds_list: List of Reynolds numbers
            mach_list: List of Mach numbers
            alpha_list: List of angles of attack
            ncrit_list: List of ncrit values
            max_workers: Number of parallel workers
            save_progress_every: Save progress every N airfoils
            
        Returns:
            dict: Detailed analysis results with progress tracking
        """
        import json
        from datetime import datetime
        
        start_time = time.time()
        progress_file = f"analysis_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Initialize progress tracking
        progress_data = {
            'start_time': datetime.now().isoformat(),
            'parameters': {
                'reynolds_list': reynolds_list,
                'mach_list': mach_list,
                'alpha_list': alpha_list,
                'ncrit_list': ncrit_list
            },
            'completed_airfoils': [],
            'failed_airfoils': [],
            'progress_snapshots': []
        }
        
        # Get initial status
        initial_status = self.get_analysis_status(reynolds_list, mach_list, alpha_list, ncrit_list)
        missing_conditions = self.database.find_missing_conditions(
            airfoil_names=None,
            reynolds_list=reynolds_list,
            mach_list=mach_list,
            alpha_list=alpha_list,
            ncrit_list=ncrit_list
        )
        
        airfoils_to_process = list(missing_conditions.keys())
        total_airfoils = len(airfoils_to_process)
        
        if total_airfoils == 0:
            logging.info("No airfoils need processing")
            return {
                'total_airfoils': 0,
                'processed_airfoils': 0,
                'successful_airfoils': 0,
                'failed_airfoils': 0,
                'duration': 0
            }
        
        logging.info(f"Starting incremental analysis for {total_airfoils} airfoils")
        
        # Process airfoils one by one with progress tracking
        for i, airfoil_name in enumerate(airfoils_to_process, 1):
            airfoil_start_time = time.time()
            
            try:
                logging.info(f"Processing {i}/{total_airfoils}: {airfoil_name}")
                
                # Run analysis for this airfoil's missing conditions
                self._run_missing_conditions(
                    airfoil_name, 
                    missing_conditions[airfoil_name], 
                    max_workers
                )
                
                airfoil_duration = time.time() - airfoil_start_time
                progress_data['completed_airfoils'].append({
                    'name': airfoil_name,
                    'completion_time': datetime.now().isoformat(),
                    'duration_seconds': airfoil_duration,
                    'conditions_processed': len(missing_conditions[airfoil_name])
                })
                
                logging.info(f"✓ Completed {airfoil_name} in {airfoil_duration:.1f}s")
                
            except Exception as e:
                airfoil_duration = time.time() - airfoil_start_time
                error_info = {
                    'name': airfoil_name,
                    'error_time': datetime.now().isoformat(),
                    'duration_seconds': airfoil_duration,
                    'error_message': str(e)
                }
                progress_data['failed_airfoils'].append(error_info)
                
                logging.error(f"✗ Failed {airfoil_name}: {e}")
            
            # Save progress periodically
            if i % save_progress_every == 0 or i == total_airfoils:
                current_time = time.time()
                elapsed_time = current_time - start_time
                progress_percentage = (i / total_airfoils) * 100
                estimated_total_time = elapsed_time / progress_percentage * 100 if progress_percentage > 0 else 0
                remaining_time = estimated_total_time - elapsed_time
                
                snapshot = {
                    'timestamp': datetime.now().isoformat(),
                    'airfoils_processed': i,
                    'total_airfoils': total_airfoils,
                    'progress_percentage': progress_percentage,
                    'elapsed_time_seconds': elapsed_time,
                    'estimated_remaining_seconds': remaining_time,
                    'successful_count': len(progress_data['completed_airfoils']),
                    'failed_count': len(progress_data['failed_airfoils'])
                }
                
                progress_data['progress_snapshots'].append(snapshot)
                
                # Save progress to file
                with open(progress_file, 'w') as f:
                    json.dump(progress_data, f, indent=2, default=str)
                
                logging.info(f"Progress: {i}/{total_airfoils} ({progress_percentage:.1f}%) - "
                           f"ETA: {remaining_time/60:.1f} minutes")
        
        # Final results
        total_duration = time.time() - start_time
        final_results = {
            'total_airfoils': total_airfoils,
            'processed_airfoils': len(progress_data['completed_airfoils']) + len(progress_data['failed_airfoils']),
            'successful_airfoils': len(progress_data['completed_airfoils']),
            'failed_airfoils': len(progress_data['failed_airfoils']),
            'duration': total_duration,
            'successful_list': [item['name'] for item in progress_data['completed_airfoils']],
            'failed_list': [item['name'] for item in progress_data['failed_airfoils']],
            'progress_file': progress_file,
            'average_time_per_airfoil': total_duration / total_airfoils if total_airfoils > 0 else 0
        }
        
        # Add final snapshot
        progress_data['final_results'] = final_results
        progress_data['end_time'] = datetime.now().isoformat()
        
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2, default=str)
        
        logging.info(f"Incremental analysis complete. Progress saved to {progress_file}")
        
        return final_results
    
    def generate_analysis_report(self, reynolds_list: List[float], mach_list: List[float],
                                alpha_list: List[float], ncrit_list: List[float],
                                output_file: str = "analysis_report.html") -> str:
        """
        Generate a comprehensive HTML report of the analysis status.
        
        Args:
            reynolds_list: List of Reynolds numbers
            mach_list: List of Mach numbers
            alpha_list: List of angles of attack
            ncrit_list: List of ncrit values
            output_file: Output HTML file path
            
        Returns:
            str: Path to generated report file
        """
        from datetime import datetime
        
        # Gather all data
        status = self.get_analysis_status(reynolds_list, mach_list, alpha_list, ncrit_list)
        performance_stats = self.get_performance_stats()
        problematic = self.find_problematic_airfoils(reynolds_list, mach_list, alpha_list, ncrit_list)
        validation = self.validate_airfoil_data()
        time_estimate = self.estimate_analysis_time(reynolds_list, mach_list, alpha_list, ncrit_list)
        
        # Create problematic airfoils table
        problematic_table = ""
        if problematic['problematic_airfoils']:
            problematic_table = "<table><tr><th>Airfoil</th><th>Success Rate</th><th>Existing</th><th>Missing</th></tr>"
            for airfoil in problematic['problematic_airfoils'][:20]:  # Show top 20
                problematic_table += f"""
                <tr>
                    <td>{airfoil['name']}</td>
                    <td class="warning">{airfoil['success_rate']*100:.1f}%</td>
                    <td>{airfoil['existing_conditions']}</td>
                    <td>{airfoil['missing_conditions']}</td>
                </tr>
                """
            problematic_table += "</table>"
            if len(problematic['problematic_airfoils']) > 20:
                problematic_table += f"<p>... and {len(problematic['problematic_airfoils']) - 20} more</p>"
        
        # Generate HTML report
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>XFoil Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .success {{ color: green; }}
        .warning {{ color: orange; }}
        .error {{ color: red; }}
        .progress-bar {{ background-color: #f0f0f0; border-radius: 10px; overflow: hidden; }}
        .progress-fill {{ background-color: #4CAF50; height: 20px; text-align: center; line-height: 20px; color: white; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>XFoil Analysis Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="section">
        <h2>Analysis Parameters</h2>
        <ul>
            <li>Reynolds Numbers: {len(reynolds_list)} values ({min(reynolds_list):.0e} to {max(reynolds_list):.0e})</li>
            <li>Mach Numbers: {len(mach_list)} values ({min(mach_list)} to {max(mach_list)})</li>
            <li>Alpha Range: {len(alpha_list)} values ({min(alpha_list)}° to {max(alpha_list)}°)</li>
            <li>Ncrit Values: {ncrit_list}</li>
            <li>Conditions per Airfoil: {len(reynolds_list) * len(mach_list) * len(alpha_list) * len(ncrit_list):,}</li>
        </ul>
    </div>
    
    <div class="section">
        <h2>Overall Status</h2>
        <div class="progress-bar">
            <div class="progress-fill" style="width: {status['completion_percentage']:.1f}%">
                {status['completion_percentage']:.1f}% Complete
            </div>
        </div>
        <br>
        <table>
            <tr><td>Total Airfoils</td><td>{status['total_airfoils']:,}</td></tr>
            <tr><td>Complete Airfoils</td><td class="success">{status['airfoils_complete']:,}</td></tr>
            <tr><td>Incomplete Airfoils</td><td class="warning">{status['airfoils_incomplete']:,}</td></tr>
            <tr><td>Total Conditions</td><td>{status['total_conditions_possible']:,}</td></tr>
            <tr><td>Existing Conditions</td><td class="success">{status['total_conditions_existing']:,}</td></tr>
            <tr><td>Missing Conditions</td><td class="error">{status['total_conditions_missing']:,}</td></tr>
        </table>
    </div>
    
    <div class="section">
        <h2>Time Estimates</h2>
        <p>Estimated time to complete missing analysis: <strong>{time_estimate['estimated_time_str']}</strong></p>
        <p>Total missing conditions: {time_estimate['total_conditions']:,}</p>
    </div>
    
    <div class="section">
        <h2>Database Statistics</h2>
        <table>
            <tr><td>Total Coefficient Records</td><td>{performance_stats.get('total_coefficient_records', 0):,}</td></tr>
            <tr><td>Airfoils with Analysis Data</td><td>{performance_stats.get('airfoils_with_analysis_data', 0):,}</td></tr>
            <tr><td>Airfoils without Data</td><td>{performance_stats.get('airfoils_without_data', 0):,}</td></tr>
            <tr><td>Unique Reynolds Numbers</td><td>{performance_stats.get('unique_reynolds_numbers', 0)}</td></tr>
            <tr><td>Unique Alpha Values</td><td>{performance_stats.get('unique_alpha_values', 0)}</td></tr>
        </table>
    </div>
    
    <div class="section">
        <h2>Data Validation</h2>
        <table>
            <tr><td>Valid Airfoils</td><td class="success">{len(validation['valid'])}</td></tr>
            <tr><td>Invalid Airfoils</td><td class="error">{len(validation['invalid'])}</td></tr>
        </table>
        {f'<p class="error">Invalid airfoils: {", ".join(validation["invalid"][:10])}{"..." if len(validation["invalid"]) > 10 else ""}</p>' if validation['invalid'] else ''}
    </div>
    
    <div class="section">
        <h2>Problematic Airfoils</h2>
        <p>Airfoils with success rate below 80%: <span class="warning">{problematic['total_problematic']}</span></p>
        {problematic_table}
    </div>
    
    <div class="section">
        <h2>Coefficient Statistics</h2>
        <table>
            <tr><th>Coefficient</th><th>Min</th><th>Max</th><th>Average</th></tr>
            <tr>
                <td>CL</td>
                <td>{performance_stats.get('cl_range', {}).get('min', 0):.3f}</td>
                <td>{performance_stats.get('cl_range', {}).get('max', 0):.3f}</td>
                <td>{performance_stats.get('cl_range', {}).get('avg', 0):.3f}</td>
            </tr>
            <tr>
                <td>CD</td>
                <td>{performance_stats.get('cd_range', {}).get('min', 0):.6f}</td>
                <td>{performance_stats.get('cd_range', {}).get('max', 0):.3f}</td>
                <td>{performance_stats.get('cd_range', {}).get('avg', 0):.6f}</td>
            </tr>
            <tr>
                <td>CM</td>
                <td>{performance_stats.get('cm_range', {}).get('min', 0):.3f}</td>
                <td>{performance_stats.get('cm_range', {}).get('max', 0):.3f}</td>
                <td>{performance_stats.get('cm_range', {}).get('avg', 0):.3f}</td>
            </tr>
        </table>
    </div>
    
    <div class="section">
        <h2>Recommendations</h2>
        <ul>
"""
        
        # Add recommendations based on analysis
        if status['total_conditions_missing'] > 0:
            html_content += f"<li class='warning'>Run analysis for {status['total_conditions_missing']:,} missing conditions</li>"
        
        if validation['invalid']:
            html_content += f"<li class='error'>Fix {len(validation['invalid'])} airfoils with invalid point cloud data</li>"
        
        if problematic['total_problematic'] > 0:
            html_content += f"<li class='warning'>Review {problematic['total_problematic']} problematic airfoils with low success rates</li>"
        
        if status['completion_percentage'] == 100:
            html_content += "<li class='success'>Analysis is complete! All airfoils have been processed.</li>"
        
        html_content += """
        </ul>
    </div>
    
</body>
</html>
"""
        
        # Write HTML file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logging.info(f"Analysis report generated: {output_file}")
        return output_file
    
    def backup_analysis_data(self, backup_file: Optional[str] = None) -> str:
        """
        Create a backup of all aerodynamic coefficient data.
        
        Args:
            backup_file: Optional backup file path. If None, generates timestamped filename.
            
        Returns:
            str: Path to backup file
        """
        import json
        from datetime import datetime
        from sqlmodel import Session, select
        from airfoil_database.core.models import AeroCoeff
        
        if backup_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = f"aero_coeffs_backup_{timestamp}.json"
        
        try:
            with Session(self.database.engine) as session:
                statement = select(AeroCoeff)
                coeffs = session.exec(statement).all()
                
                backup_data = {
                    'backup_timestamp': datetime.now().isoformat(),
                    'total_records': len(coeffs),
                    'data': [
                        {
                            'name': coeff.name,
                            'reynolds_number': coeff.reynolds_number,
                            'mach': coeff.mach,
                            'ncrit': coeff.ncrit,
                            'alpha': coeff.alpha,
                            'cl': coeff.cl,
                            'cd': coeff.cd,
                            'cm': coeff.cm
                        }
                        for coeff in coeffs
                    ]
                }
            
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2, default=str)
            
            logging.info(f"Backup created: {backup_file} ({len(coeffs):,} records)")
            return backup_file
            
        except Exception as e:
            logging.error(f"Error creating backup: {e}")
            raise
    
    def restore_analysis_data(self, backup_file: str, overwrite: bool = False) -> Dict[str, int]:
        """
        Restore aerodynamic coefficient data from backup.
        
        Args:
            backup_file: Path to backup file
            overwrite: If True, overwrite existing records
            
        Returns:
            dict: Statistics about restored data
        """
        import json
        from sqlmodel import Session, select
        from airfoil_database.core.models import AeroCoeff
        
        try:
            with open(backup_file, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)
            
            records_to_restore = backup_data['data']
            restored_count = 0
            skipped_count = 0
            error_count = 0
            
            with Session(self.database.engine) as session:
                for record_data in records_to_restore:
                    try:
                        # Check if record exists
                        existing_statement = select(AeroCoeff).where(
                            (AeroCoeff.name == record_data['name']) &
                            (AeroCoeff.reynolds_number == record_data['reynolds_number']) &
                            (AeroCoeff.mach == record_data['mach']) &
                            (AeroCoeff.alpha == record_data['alpha'])
                        )
                        existing_record = session.exec(existing_statement).first()
                        
                        if existing_record and not overwrite:
                            skipped_count += 1
                            continue
                        
                        if existing_record and overwrite:
                            # Update existing record
                            existing_record.ncrit = record_data['ncrit']
                            existing_record.cl = record_data['cl']
                            existing_record.cd = record_data['cd']
                            existing_record.cm = record_data['cm']
                            session.add(existing_record)
                        else:
                            # Create new record
                            new_record = AeroCoeff(**record_data)
                            session.add(new_record)
                        
                        restored_count += 1
                        
                        # Commit in batches
                        if restored_count % 1000 == 0:
                            session.commit()
                            logging.info(f"Restored {restored_count:,} records...")
                    
                    except Exception as e:
                        error_count += 1
                        logging.error(f"Error restoring record: {e}")
                
                session.commit()
            
            result = {
                'total_in_backup': len(records_to_restore),
                'restored': restored_count,
                'skipped': skipped_count,
                'errors': error_count
            }
            
            logging.info(f"Restore complete: {restored_count:,} restored, {skipped_count:,} skipped, {error_count} errors")
            return result
            
        except Exception as e:
            logging.error(f"Error restoring from backup: {e}")
            raise
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get system information relevant to XFoil analysis performance.
        
        Returns:
            dict: System information
        """
        import platform
        import os
        
        system_info = {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'architecture': platform.architecture(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'cpu_count': os.cpu_count(),
            'xfoil_executable': self.xfoil_runner.xfoil_executable
        }
        
        # Try to get memory information
        try:
            import psutil
            memory = psutil.virtual_memory()
            system_info.update({
                'total_memory_gb': memory.total / (1024**3),
                'available_memory_gb': memory.available / (1024**3),
                'memory_usage_percent': memory.percent
            })
        except ImportError:
            system_info['memory_info'] = 'psutil not available'
        
        # Check XFoil availability
        try:
            import subprocess
            result = subprocess.run([self.xfoil_runner.xfoil_executable], 
                                  input="quit\n", 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=5)
            system_info['xfoil_available'] = True
        except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError):
            system_info['xfoil_available'] = False
        
        # Database information
        system_info.update({
            'database_path': self.database.db_path,
            'database_exists': os.path.exists(self.database.db_path)
        })
        
        if os.path.exists(self.database.db_path):
            system_info['database_size_mb'] = os.path.getsize(self.database.db_path) / (1024**2)
        
        return system_info
    
    def run_comprehensive_workflow(self, reynolds_list: List[float], mach_list: List[float],
                                  alpha_list: List[float], ncrit_list: List[float],
                                  max_workers: Optional[int] = None,
                                  generate_report: bool = True,
                                  create_backup: bool = True,
                                  run_validation: bool = True) -> Dict[str, Any]:
        """
        Run a comprehensive analysis workflow with all features.
        
        Args:
            reynolds_list: List of Reynolds numbers
            mach_list: List of Mach numbers
            alpha_list: List of angles of attack
            ncrit_list: List of ncrit values
            max_workers: Number of parallel workers
            generate_report: Whether to generate HTML report
            create_backup: Whether to create backup before analysis
            run_validation: Whether to run validation checks
            
        Returns:
            dict: Comprehensive workflow results
        """
        from datetime import datetime
        
        workflow_start_time = time.time()
        workflow_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        logging.info(f"Starting comprehensive workflow {workflow_id}")
        
        workflow_results = {
            'workflow_id': workflow_id,
            'start_time': datetime.now().isoformat(),
            'parameters': {
                'reynolds_list': reynolds_list,
                'mach_list': mach_list,
                'alpha_list': alpha_list,
                'ncrit_list': ncrit_list,
                'max_workers': max_workers
            },
            'steps_completed': [],
            'files_generated': []
        }
        
        try:
            # Step 1: System validation
            logging.info("Step 1: System validation")
            system_info = self.get_system_info()
            workflow_results['system_info'] = system_info
            workflow_results['steps_completed'].append('system_validation')
            
            if not system_info['xfoil_available']:
                raise RuntimeError("XFoil executable not available")
            
            # Step 2: Data validation
            if run_validation:
                logging.info("Step 2: Data validation")
                validation_results = self.validate_airfoil_data()
                workflow_results['validation_results'] = validation_results
                workflow_results['steps_completed'].append('data_validation')
                
                if validation_results['invalid']:
                    logging.warning(f"Found {len(validation_results['invalid'])} invalid airfoils")
            
            # Step 3: Initial status check
            logging.info("Step 3: Initial status assessment")
            initial_status = self.get_analysis_status(reynolds_list, mach_list, alpha_list, ncrit_list)
            workflow_results['initial_status'] = initial_status
            workflow_results['steps_completed'].append('initial_status')
            
            # Step 4: Backup creation
            if create_backup:
                logging.info("Step 4: Creating backup")
                backup_file = self.backup_analysis_data(f"pre_analysis_backup_{workflow_id}.json")
                workflow_results['backup_file'] = backup_file
                workflow_results['files_generated'].append(backup_file)
                workflow_results['steps_completed'].append('backup_creation')
            
            # Step 5: Optimize batch size
            logging.info("Step 5: Optimizing batch size")
            optimal_batch_size = self.optimize_batch_size(reynolds_list, mach_list, alpha_list, ncrit_list)
            workflow_results['optimal_batch_size'] = optimal_batch_size
            workflow_results['steps_completed'].append('batch_optimization')
            
            # Step 6: Run test analysis
            logging.info("Step 6: Running test analysis")
            test_results = self.run_test_analysis(max_workers=min(2, max_workers or 2))
            workflow_results['test_results'] = test_results
            workflow_results['steps_completed'].append('test_analysis')
            
            if not test_results['success']:
                raise RuntimeError(f"Test analysis failed: {test_results.get('error', 'Unknown error')}")
            
            # Step 7: Main analysis
            if initial_status['total_conditions_missing'] > 0:
                logging.info("Step 7: Running main analysis")
                
                # Choose analysis method based on scale
                if initial_status['total_conditions_missing'] > 10000:
                    # Use incremental analysis for large datasets
                    analysis_results = self.run_incremental_analysis(
                        reynolds_list, mach_list, alpha_list, ncrit_list, max_workers
                    )
                else:
                    # Use regular batch analysis
                    analysis_results = self.run_batch_analysis(
                        reynolds_list, mach_list, alpha_list, ncrit_list,
                        max_workers=max_workers, skip_existing=True,
                        batch_size=optimal_batch_size
                    )
                
                workflow_results['analysis_results'] = analysis_results
                workflow_results['steps_completed'].append('main_analysis')
                
                # Step 8: Retry failed conditions
                if analysis_results['failed_airfoils'] > 0:
                    logging.info("Step 8: Retrying failed conditions")
                    retry_results = self.retry_failed_conditions(
                        reynolds_list, mach_list, alpha_list, ncrit_list, max_workers
                    )
                    workflow_results['retry_results'] = retry_results
                    workflow_results['steps_completed'].append('retry_analysis')
            else:
                logging.info("Step 7: Skipping analysis - no missing conditions")
                workflow_results['analysis_results'] = {'message': 'No analysis needed - all conditions complete'}
                workflow_results['steps_completed'].append('analysis_skipped')
            
            # Step 9: Final status check
            logging.info("Step 9: Final status assessment")
            final_status = self.get_analysis_status(reynolds_list, mach_list, alpha_list, ncrit_list)
            workflow_results['final_status'] = final_status
            workflow_results['steps_completed'].append('final_status')
            
            # Step 10: Generate reports
            if generate_report:
                logging.info("Step 10: Generating reports")
                
                # HTML report
                html_report = self.generate_analysis_report(
                    reynolds_list, mach_list, alpha_list, ncrit_list,
                    f"analysis_report_{workflow_id}.html"
                )
                workflow_results['html_report'] = html_report
                workflow_results['files_generated'].append(html_report)
                
                # JSON summary
                json_summary = f"analysis_summary_{workflow_id}.json"
                self.export_analysis_summary(
                    reynolds_list, mach_list, alpha_list, ncrit_list, json_summary
                )
                workflow_results['json_summary'] = json_summary
                workflow_results['files_generated'].append(json_summary)
                
                # CSV missing conditions (if any)
                if final_status['total_conditions_missing'] > 0:
                    csv_missing = f"missing_conditions_{workflow_id}.csv"
                    self.export_missing_conditions_report(
                        reynolds_list, mach_list, alpha_list, ncrit_list, csv_missing
                    )
                    workflow_results['csv_missing'] = csv_missing
                    workflow_results['files_generated'].append(csv_missing)
                
                workflow_results['steps_completed'].append('report_generation')
            
            # Step 11: Cleanup
            logging.info("Step 11: Cleanup")
            self.cleanup_failed_results()
            workflow_results['steps_completed'].append('cleanup')
            
            # Calculate final statistics
            workflow_duration = time.time() - workflow_start_time
            workflow_results.update({
                'end_time': datetime.now().isoformat(),
                'total_duration_seconds': workflow_duration,
                'total_duration_minutes': workflow_duration / 60,
                'success': True,
                'improvement': {
                    'conditions_added': (final_status['total_conditions_existing'] - 
                                       initial_status['total_conditions_existing']),
                    'completion_improvement': (final_status['condition_completion_percentage'] - 
                                             initial_status['condition_completion_percentage'])
                }
            })
            
            logging.info(f"Comprehensive workflow {workflow_id} completed successfully in {workflow_duration/60:.1f} minutes")
            
        except Exception as e:
            workflow_duration = time.time() - workflow_start_time
            workflow_results.update({
                'end_time': datetime.now().isoformat(),
                'total_duration_seconds': workflow_duration,
                'success': False,
                'error': str(e),
                'error_step': workflow_results['steps_completed'][-1] if workflow_results['steps_completed'] else 'initialization'
            })
            
            logging.error(f"Comprehensive workflow {workflow_id} failed: {e}")
            raise
        
        return workflow_results
    
    def schedule_analysis(self, reynolds_list: List[float], mach_list: List[float],
                         alpha_list: List[float], ncrit_list: List[float],
                         schedule_time: str, max_workers: Optional[int] = None) -> Dict[str, Any]:
        """
        Schedule analysis to run at a specific time (basic implementation).
        
        Args:
            reynolds_list: List of Reynolds numbers
            mach_list: List of Mach numbers
            alpha_list: List of angles of attack
            ncrit_list: List of ncrit values
            schedule_time: Time to run analysis (HH:MM format)
            max_workers: Number of parallel workers
            
        Returns:
            dict: Scheduling information
        """
        from datetime import datetime, time as dt_time
        import time
        
        try:
            # Parse schedule time
            hour, minute = map(int, schedule_time.split(':'))
            target_time = dt_time(hour, minute)
            
            # Calculate wait time
            now = datetime.now()
            target_datetime = datetime.combine(now.date(), target_time)
            
            # If target time has passed today, schedule for tomorrow
            if target_datetime <= now:
                from datetime import timedelta
                target_datetime += timedelta(days=1)
            
            wait_seconds = (target_datetime - now).total_seconds()
            
            logging.info(f"Analysis scheduled for {target_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
            logging.info(f"Waiting {wait_seconds/3600:.1f} hours...")
            
            # Wait until scheduled time
            time.sleep(wait_seconds)
            
            # Run the analysis
            logging.info("Starting scheduled analysis")
            results = self.run_comprehensive_workflow(
                reynolds_list, mach_list, alpha_list, ncrit_list, max_workers
            )
            
            results['scheduled_time'] = target_datetime.isoformat()
            results['wait_time_seconds'] = wait_seconds
            
            return results
            
        except Exception as e:
            logging.error(f"Scheduled analysis failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'scheduled_time': schedule_time
            }

