import subprocess
import re
import time
import os
import tempfile
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import shlex
from pathlib import Path

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(process)d - %(levelname)s - %(message)s')

# --- Worker Function (Static/Top-Level) ---
def _run_xfoil_alpha_worker(xfoil_executable, airfoil_name, points_str, reynolds, mach, alpha, ncrit):
    """
    Optimized static worker function to run XFoil for a single alpha.
    
    Returns:
        tuple: (alpha, dict_of_coeffs) or (alpha, None) if failed.
    """
    temp_airfoil_path = None
    output_file_path = None
    XFOIL_TIMEOUT = 30  # Reduced timeout for faster failure detection
    
    try:
        # --- Create Temporary Files with better error handling ---
        try:
            # Use pathlib for better path handling
            temp_dir = Path(tempfile.gettempdir())
            temp_airfoil_path = temp_dir / f"airfoil_{os.getpid()}_{int(time.time()*1000)}.dat"
            output_file_path = temp_dir / f"output_{os.getpid()}_{int(time.time()*1000)}.pol"
            
            # Write airfoil points
            with open(temp_airfoil_path, 'w', encoding='utf-8') as f:
                f.write(points_str)
                
        except Exception as e:
            logging.error(f"[Worker] Error creating temp files: {e}")
            return alpha, None

        # --- Construct XFoil Commands (Optimized) ---
        # Removed redundant commands and optimized for single alpha runs
        xfoil_input = f"""load {temp_airfoil_path}
{airfoil_name}
pane
ppar
n 240
t 1


oper
vpar
n {ncrit}

iter 20
re {reynolds}
mach {mach}
visc
pacc
{output_file_path}

alfa {alpha}
pacc

quit
"""

        # --- Execute XFoil with optimized process handling ---
        start_time = time.time()
        process = None
        
        try:
            # Use a working directory to avoid path issues
            process = subprocess.Popen(
                [str(xfoil_executable)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='ignore',
                cwd=temp_dir,  # Set working directory
                bufsize=0  # Unbuffered for immediate response
            )
            
            # Send input and get results with timeout
            stdout, stderr = process.communicate(input=xfoil_input, timeout=XFOIL_TIMEOUT)
            end_time = time.time()
            run_duration = end_time - start_time

        except subprocess.TimeoutExpired:
            logging.warning(f"[Worker] XFoil timeout ({XFOIL_TIMEOUT}s) for {airfoil_name} A={alpha}")
            if process:
                try:
                    process.terminate()
                    process.wait(timeout=2)
                except (subprocess.TimeoutExpired, ProcessLookupError):
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass  # Process already dead
            return alpha, None
            
        except FileNotFoundError:
            logging.error(f"[Worker] XFoil executable not found: '{xfoil_executable}'")
            return alpha, None
            
        except Exception as popen_err:
            logging.error(f"[Worker] Process error for {airfoil_name} A={alpha}: {popen_err}")
            return alpha, None

        # --- Enhanced Result Processing ---
        if process.returncode != 0:
            # Check for specific XFoil error patterns in stderr/stdout
            error_msg = stderr.strip() + stdout.strip()
            if "not converged" in error_msg.lower():
                logging.debug(f"[Worker] Convergence failure for {airfoil_name} A={alpha}")
            elif "stall" in error_msg.lower():
                logging.debug(f"[Worker] Stall condition for {airfoil_name} A={alpha}")
            else:
                logging.warning(f"[Worker] XFoil error (Code {process.returncode}) for {airfoil_name} A={alpha}")
            return alpha, None

        # Check if output file was actually created and has content
        if not output_file_path.exists() or output_file_path.stat().st_size == 0:
            logging.debug(f"[Worker] No output file generated for {airfoil_name} A={alpha}")
            return alpha, None

        # Parse the generated polar file
        coefficients = XFoilRunner._parse_polar_file(output_file_path)

        if coefficients:
            # Validate coefficients are reasonable
            if XFoilRunner._validate_coefficients(coefficients, alpha):
                logging.debug(f"[Worker] Success: {airfoil_name} A={alpha} -> {coefficients} ({run_duration:.2f}s)")
                return alpha, coefficients
            else:
                logging.debug(f"[Worker] Invalid coefficients for {airfoil_name} A={alpha}: {coefficients}")
                return alpha, None
        else:
            logging.debug(f"[Worker] Failed to parse coefficients for {airfoil_name} A={alpha}")
            return alpha, None

    except Exception as e:
        logging.error(f"[Worker] Unexpected error for {airfoil_name} A={alpha}: {e}", exc_info=True)
        return alpha, None
        
    finally:
        # --- Cleanup with better error handling ---
        for f_path in [temp_airfoil_path, output_file_path]:
            if f_path and Path(f_path).exists():
                try:
                    Path(f_path).unlink()
                except Exception as e:
                    logging.warning(f"[Worker] Could not remove {f_path}: {e}")


class XFoilRunner:
    """
    Optimized XFoilRunner with improved error handling and performance.
    """

    def __init__(self, aero_analyzer, xfoil_executable=None):
        """Initialize XFoilRunner with validation."""
        self.analyzer = aero_analyzer
        self.database = aero_analyzer.db
        self.xfoil_executable = xfoil_executable if xfoil_executable else "xfoil"
        
        # Validate XFoil executable exists
        if not self._validate_xfoil_executable():
            raise FileNotFoundError(f"XFoil executable not found or not executable: {self.xfoil_executable}")
            
        logging.info(f"Using XFoil executable: {self.xfoil_executable}")

    def _validate_xfoil_executable(self):
        """Validate that XFoil executable exists and is executable."""
        try:
            result = subprocess.run([self.xfoil_executable], 
                                  input="quit\n", 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=5)
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError):
            return False

    @staticmethod
    def _validate_coefficients(coefficients, alpha):
        """Validate that aerodynamic coefficients are reasonable."""
        if not coefficients:
            return False
            
        cl, cd, cm = coefficients.get('Cl', 0), coefficients.get('Cd', 0), coefficients.get('Cm', 0)
        
        # Basic sanity checks
        if not (-5 <= cl <= 5):  # Reasonable Cl range
            return False
        if not (0 <= cd <= 1):   # Reasonable Cd range
            return False
        if not (-1 <= cm <= 1):  # Reasonable Cm range
            return False
        if cd < 0.001:           # Cd too small (likely convergence issue)
            return False
            
        return True

    @staticmethod
    def _parse_polar_file(filepath):
        """
        Enhanced parsing of XFoil polar output with better error handling.
        """
        try:
            filepath = Path(filepath)
            if not filepath.exists():
                return None
                
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()

            # Look for data header more robustly
            data_start_line = -1
            for i, line in enumerate(lines):
                cleaned_line = line.strip().lower()
                # Look for the header line containing column names
                if ('alpha' in cleaned_line and 'cl' in cleaned_line and 
                    'cd' in cleaned_line and 'cm' in cleaned_line):
                    # Skip any separator lines (typically dashes)
                    potential_start = i + 1
                    while (potential_start < len(lines) and 
                           lines[potential_start].strip().startswith('-')):
                        potential_start += 1
                    data_start_line = potential_start
                    break

            if data_start_line == -1 or data_start_line >= len(lines):
                return None

            # Find the first non-empty data line
            for line_idx in range(data_start_line, len(lines)):
                data_line = lines[line_idx].strip()
                if not data_line:
                    continue
                    
                values = data_line.split()
                if len(values) >= 5:
                    try:
                        # XFoil polar format: alpha CL CD CDp CM
                        alpha = float(values[0])
                        cl = float(values[1])
                        cd = float(values[2])
                        cm = float(values[4])
                        return {"Cl": cl, "Cd": cd, "Cm": cm}
                    except (ValueError, IndexError) as e:
                        logging.debug(f"Could not parse line '{data_line}': {e}")
                        continue

            return None

        except Exception as e:
            logging.error(f"Error parsing polar file {filepath}: {e}")
            return None

    @staticmethod
    def _run_condition_set_static(xfoil_executable, airfoil_name, points_str, reynolds, mach, ncrit, alphas):
        """
        Optimized static worker function with better batch processing.
        """
        successful_results = []
        failed_alphas = []
        
        logging.info(f"[Worker] Processing {len(alphas)} alphas for {airfoil_name}, Re={reynolds}, M={mach}")
        
        # Sort alphas to process from low to high for potentially better convergence
        sorted_alphas = sorted(alphas)
        
        for alpha in sorted_alphas:
            _, coeffs = _run_xfoil_alpha_worker(
                xfoil_executable, airfoil_name, points_str, reynolds, mach, alpha, ncrit
            )
            
            if coeffs:
                successful_results.append({'alpha': alpha, **coeffs})
            else:
                failed_alphas.append(alpha)
                
        success_rate = len(successful_results) / len(alphas) * 100
        logging.info(f"[Worker] Completed {airfoil_name} Re={reynolds} M={mach}: "
                    f"{len(successful_results)}/{len(alphas)} ({success_rate:.1f}%) successful")
        
        if failed_alphas:
            logging.debug(f"[Worker] Failed alphas: {failed_alphas}")
            
        return airfoil_name, reynolds, mach, ncrit, successful_results

    def run_analysis_parallel(self, airfoil_name, reynolds_list, mach_list, alpha_list, ncrit_list, max_workers=None):
        """
        Optimized parallel analysis with better resource management.
        """
        # Input validation
        if not isinstance(reynolds_list, list): reynolds_list = [reynolds_list]
        if not isinstance(mach_list, list): mach_list = [mach_list]
        if not isinstance(alpha_list, list): alpha_list = [alpha_list]
        if not isinstance(ncrit_list, list): ncrit_list = [ncrit_list]

        start_total_time = time.time()
        logging.info(f"Starting optimized parallel analysis for: {airfoil_name}")

        # Validate airfoil data
        airfoil_db_data = self.database.get_airfoil_data(airfoil_name)
        if not airfoil_db_data or not airfoil_db_data[1]:
            logging.error(f"Airfoil '{airfoil_name}' not found. Aborting.")
            return

        _, points_str, _, _ = airfoil_db_data

        # Optimize worker count based on system resources
        if not max_workers:
            cores = os.cpu_count()
            # Use fewer workers than cores to avoid overwhelming the system
            max_workers = max(1, cores - 1) if cores and cores > 2 else 1
            logging.info(f"Using {max_workers} workers (system has {cores} cores)")

        # Prepare tasks
        tasks_args = []
        for reynolds in reynolds_list:
            for mach in mach_list:
                for ncrit in ncrit_list:
                    tasks_args.append((
                        self.xfoil_executable,
                        airfoil_name,
                        points_str,
                        reynolds,
                        mach,
                        ncrit,
                        alpha_list
                    ))

        total_conditions = len(tasks_args)
        total_alphas = total_conditions * len(alpha_list)
        
        logging.info(f"Processing {total_conditions} conditions ({total_alphas} total alpha points)")

        # Execute with progress tracking
        completed_conditions = 0
        total_successful = 0
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(XFoilRunner._run_condition_set_static, *args) 
                      for args in tasks_args]

            for future in as_completed(futures):
                try:
                    airfoil, re, mach, ncrit, results = future.result()
                    completed_conditions += 1
                    
                    # Progress reporting
                    progress = (completed_conditions / total_conditions) * 100
                    logging.info(f"Progress: {completed_conditions}/{total_conditions} "
                               f"({progress:.1f}%) - Latest: Re={re}, M={mach}")

                    # Store results
                    if results:
                        total_successful += len(results)
                        for result in results:
                            try:
                                self.analyzer.store_aero_coeffs(
                                    airfoil, re, mach, ncrit, result['alpha'],
                                    result['Cl'], result['Cd'], result['Cm']
                                )
                            except Exception as db_err:
                                logging.error(f"Database error: {db_err}")

                except Exception as exc:
                    logging.error(f"Worker task exception: {exc}", exc_info=True)

        # Final summary
        end_time = time.time()
        duration = end_time - start_total_time
        success_rate = (total_successful / total_alphas) * 100
        
        logging.info(f"Analysis complete for {airfoil_name}")
        logging.info(f"Total time: {duration:.1f}s ({duration/60:.1f} minutes)")
        logging.info(f"Success rate: {total_successful}/{total_alphas} ({success_rate:.1f}%)")
        logging.info(f"Average time per alpha: {duration/total_alphas:.2f}s")