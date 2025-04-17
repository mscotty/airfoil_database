import subprocess
import re
import time
import os
import tempfile
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import shlex # Import shlex for safer command splitting if needed

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class XFoilRunner:
    """
    Runs XFoil simulations for individual angles of attack, handling specific
    setup commands and integrating with an AirfoilDatabase. Incorporates
    parallel processing for different run conditions.
    """

    def __init__(self, database, xfoil_executable=None):
        """
        Initializes the XFoilRunner.

        Args:
            database: An instance of the AirfoilDatabase class.
            xfoil_executable (str, optional): The path to the XFoil executable.
                If None or empty, defaults to "xfoil" assuming it's in PATH.
        """
        self.database = database
        # Use 'xfoil' as default if no path is provided or if it's empty
        self.xfoil_executable = xfoil_executable if xfoil_executable else "xfoil"
        logging.info(f"Using XFoil executable: {self.xfoil_executable}")

    def _write_temp_airfoil_file(self, points_str):
        """Writes airfoil points to a temporary file for XFoil."""
        try:
            # Create a temporary file that will be automatically deleted
            with tempfile.NamedTemporaryFile(delete=False, suffix=".dat", mode="w", encoding='utf-8') as temp_file:
                temp_file.write(points_str)
                return temp_file.name
        except Exception as e:
            logging.error(f"Error creating temporary airfoil file: {e}")
            return None

    def _run_single_alpha_task(self, airfoil_name, points_str, reynolds, mach, alpha, ncrit):
        """
        Internal method to run XFoil for a single alpha.

        Returns:
            tuple: (alpha, dict_of_coeffs) or (alpha, None) if failed.
                   dict_of_coeffs contains {'Cl': float, 'Cd': float, 'Cm': float}
        """
        temp_airfoil_path = None
        output_file_path = None

        try:
            # --- Create Temporary Files ---
            temp_airfoil_path = self._write_temp_airfoil_file(points_str)
            if not temp_airfoil_path:
                return alpha, None # Failed to create temp file

            # Create a unique temporary file for the output of this specific alpha run
            # This avoids race conditions if multiple alphas were ever run in parallel
            # and simplifies cleanup.
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pol", mode="w+", encoding='utf-8') as temp_out_file:
                output_file_path = temp_out_file.name
            if os.path.exists(output_file_path):
                os.remove(output_file_path)
            # Ensure the output file exists but is empty before the run
            #with open(output_file_path, 'w') as f:
            #    pass

            # --- Construct XFoil Commands ---
            # Note: Using PANE commands as specified
            xfoil_input = f"""
            load {temp_airfoil_path}
            {airfoil_name}
            pane
            ppar
            n 240
            t 1

            
            oper
            vpar
            n {ncrit}

            iter 200
            re {reynolds}
            mach {mach}
            visc
            pacc
            {output_file_path}

            alfa {alpha}
            pacc

            quit
            """
            # Remove leading whitespace for each line for robustness
            xfoil_input = "\n".join(line.strip() for line in xfoil_input.strip().split('\n')) + "\n"

            # --- Execute XFoil ---
            start_time = time.time()
            process = subprocess.Popen(
                [self.xfoil_executable],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, # Capture stdout for potential debugging
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='ignore' # Ignore decoding errors if XFoil outputs weird characters
            )
            stdout, stderr = process.communicate(input=xfoil_input)
            end_time = time.time()
            run_duration = end_time - start_time

            with open(r'D:\Mitchell\School\airfoils\ag12\stdout.txt', 'w+') as file:
                file.write(stdout)

            # --- Process Results ---
            if process.returncode != 0:
                logging.warning(f"XFoil process error (Code {process.returncode}) for {airfoil_name} Re={reynolds} M={mach} A={alpha}. Stderr: {stderr.strip()}")
                return alpha, None

            # Parse the generated polar file
            coefficients = self._parse_polar_file(output_file_path)

            if coefficients:
                 logging.debug(f"Success: {airfoil_name} Re={reynolds} M={mach} A={alpha} -> {coefficients} (Time: {run_duration:.2f}s)")
                 return alpha, coefficients
            else:
                 logging.warning(f"Failed to parse/find coeffs for {airfoil_name} Re={reynolds} M={mach} A={alpha}. Stdout: {stdout.strip()}")
                 return alpha, None

        except FileNotFoundError:
            # This error should only happen once if the executable is wrong
            logging.error(f"XFoil executable not found at '{self.xfoil_executable}'. Ensure it's installed and path is correct.")
            # Raise it so the main loop knows to stop
            raise
        except Exception as e:
            logging.error(f"Error running XFoil task for {airfoil_name} A={alpha}: {e}")
            return alpha, None
        finally:
            # --- Cleanup Temporary Files ---
            if temp_airfoil_path and os.path.exists(temp_airfoil_path):
                try:
                    os.remove(temp_airfoil_path)
                except Exception as e:
                    logging.warning(f"Could not remove temp airfoil file {temp_airfoil_path}: {e}")
            if output_file_path and os.path.exists(output_file_path):
                 try:
                     os.remove(output_file_path)
                 except Exception as e:
                     logging.warning(f"Could not remove temp output file {output_file_path}: {e}")

    def _parse_polar_file(self, filepath):
        """
        Parses the XFoil polar output file (.pol) generated by PACC.

        Args:
            filepath (str): The path to the .pol file.

        Returns:
            dict: A dictionary containing the aerodynamic coefficients
                  {'Cl': float, 'Cd': float, 'Cm': float}, or None if parsing fails
                  or the file is empty/invalid.
        """
        try:
            # Read the file, skipping header lines. Header length can vary slightly.
            # We look for the line starting with 'alpha'
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()

            data_start_line = -1
            for i, line in enumerate(lines):
                if line.strip().startswith('alpha'):
                    # Data usually starts 2 lines after the header
                    data_start_line = i + 2
                    break

            if data_start_line == -1 or data_start_line >= len(lines):
                logging.debug(f"Could not find data header in {filepath}")
                return None # Header not found or no data lines

            # Extract the first data line (should be the only one for single alpha)
            data_line = lines[data_start_line].strip()
            values = data_line.split()

            # Expecting at least alpha, Cl, Cd, Cdp, Cm (5 values)
            if len(values) >= 5:
                cl = float(values[1])
                cd = float(values[2])
                cm = float(values[4]) # Cm is typically the 5th value
                return {"Cl": cl, "Cd": cd, "Cm": cm}
            else:
                logging.debug(f"Insufficient values in data line of {filepath}: {data_line}")
                return None

        except FileNotFoundError:
             logging.warning(f"Polar output file not found: {filepath}")
             return None
        except (ValueError, IndexError) as e:
            logging.warning(f"Error parsing polar file {filepath}: {e}. Line: '{data_line}'")
            return None
        except Exception as e:
            logging.error(f"Unexpected error reading polar file {filepath}: {e}")
            return None


    def _run_condition_set(self, airfoil_name, points_str, reynolds, mach, ncrit, alphas):
        """
        Worker function to run all alphas for a specific (Re, Mach, Ncrit) set.
        This function is intended to be run in a separate process.

        Args:
            airfoil_name (str): Airfoil name.
            points_str (str): Airfoil geometry as a string.
            reynolds (float): Reynolds number.
            mach (float): Mach number.
            ncrit (float): Transition criterion.
            alphas (list): List of angles of attack.

        Returns:
            tuple: (reynolds, mach, ncrit, list_of_successful_results)
                   list_of_successful_results contains dicts: {'alpha': a, 'Cl': cl, 'Cd': cd, 'Cm': cm}
        """
        successful_results = []
        logging.info(f"Starting analysis for {airfoil_name}, Re={reynolds}, M={mach}, Ncrit={ncrit}")

        for alpha in alphas:
            try:
                # Run XFoil for this single alpha
                _, coeffs = self._run_single_alpha_task(airfoil_name, points_str, reynolds, mach, alpha, ncrit)

                # If successful, add to results list
                if coeffs:
                    successful_results.append({
                        'alpha': alpha,
                        'Cl': coeffs['Cl'],
                        'Cd': coeffs['Cd'],
                        'Cm': coeffs['Cm']
                    })
            except Exception as e:
                # Log errors from the task function itself (e.g., FileNotFoundError for executable)
                logging.error(f"Critical error in worker for {airfoil_name}, Re={reynolds}, M={mach}, Ncrit={ncrit}, Alpha={alpha}: {e}")
                # Depending on the error, we might want to stop processing this condition set
                # For now, we log and continue to the next alpha
                continue # Continue to next alpha

        logging.info(f"Finished analysis for {airfoil_name}, Re={reynolds}, M={mach}, Ncrit={ncrit}. Successes: {len(successful_results)}/{len(alphas)}")
        return reynolds, mach, ncrit, successful_results


    def run_analysis_parallel(self, 
                              airfoil_name, 
                              reynolds_list, 
                              mach_list, 
                              alpha_list, 
                              ncrit_list, 
                              max_workers=None):
        """
        Runs XFoil analysis in parallel for the specified airfoil across
        multiple conditions (Re, Mach, Ncrit). Each condition set runs its
        alpha sequence serially.

        Args:
            airfoil_name (str): The name of the airfoil.
            reynolds_list (list): List of Reynolds numbers.
            mach_list (list): List of Mach numbers.
            alpha_list (list): List of angles of attack.
            ncrit_list (list): List of transition criteria.
            max_workers (int, optional): Maximum number of parallel processes.
                                         Defaults to os.cpu_count().
        """
        if not isinstance(reynolds_list, list):
            reynolds_list = [reynolds_list]
        if not isinstance(mach_list, list):
            mach_list = [mach_list]
        if not isinstance(alpha_list, list):
            alpha_list = [alpha_list]
        if not isinstance(ncrit_list, list):
            ncrit_list = [ncrit_list]
        
        start_total_time = time.time()
        logging.info(f"Starting parallel analysis for airfoil: {airfoil_name}")

        # Fetch airfoil data once
        airfoil_db_data = self.database.get_airfoil_data(airfoil_name)
        if not airfoil_db_data or not airfoil_db_data[1]: # Check if data exists and pointcloud is not empty
            logging.error(f"Airfoil '{airfoil_name}' not found in database or has no pointcloud data. Skipping.")
            return
        _, points_str, _, _ = airfoil_db_data # points_str is the pointcloud

        if not max_workers:
             # os.cpu_count() might be None
            cores = os.cpu_count()
            max_workers = cores if cores else 1 # Default to 1 if count is unavailable
            logging.info(f"Using default max_workers: {max_workers}")

        tasks = []
        for reynolds in reynolds_list:
            for mach in mach_list:
                for ncrit in ncrit_list:
                    tasks.append((airfoil_name, points_str, reynolds, mach, ncrit, alpha_list))

        if not tasks:
            logging.warning("No simulation tasks generated. Check input lists.")
            return

        logging.info(f"Submitting {len(tasks)} condition sets to ProcessPoolExecutor with {max_workers} workers.")

        total_alphas_processed = 0
        total_alphas_successful = 0

        # Using ProcessPoolExecutor for CPU-bound XFoil tasks
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_condition = {
                executor.submit(self._run_condition_set, *task_args): task_args
                for task_args in tasks
            }

            for future in as_completed(future_to_condition):
                condition_args = future_to_condition[future]
                current_airfoil, _, current_re, current_m, current_ncrit, current_alphas = condition_args
                num_alphas_in_task = len(current_alphas)
                total_alphas_processed += num_alphas_in_task
                try:
                    # Get the results from the completed future
                    reynolds, mach, ncrit, successful_results = future.result()

                    # Store successful results in the database
                    if successful_results:
                        total_alphas_successful += len(successful_results)
                        for result in successful_results:
                            try:
                                self.database.store_aero_coeffs(
                                    airfoil_name,
                                    reynolds,
                                    mach,
                                    result['alpha'],
                                    result['Cl'],
                                    result['Cd'],
                                    result['Cm']
                                    # Add ncrit here if your store_aero_coeffs supports it
                                )
                            except Exception as db_err:
                                logging.error(f"Database error storing result for {airfoil_name} Re={reynolds} M={mach} A={result['alpha']}: {db_err}")
                        logging.info(f"Stored {len(successful_results)} results for {airfoil_name}, Re={reynolds}, M={mach}, Ncrit={ncrit}")
                    else:
                         logging.info(f"No successful results to store for {airfoil_name}, Re={reynolds}, M={mach}, Ncrit={ncrit}")

                except Exception as exc:
                    logging.error(f"Condition set {current_airfoil} Re={current_re} M={current_m} Ncrit={current_ncrit} generated an exception: {exc}", exc_info=True)

        end_total_time = time.time()
        logging.info(f"Finished parallel analysis for {airfoil_name}.")
        logging.info(f"Total time: {end_total_time - start_total_time:.2f} seconds")
        logging.info(f"Total alphas processed: {total_alphas_processed}")
        logging.info(f"Total successful alphas stored: {total_alphas_successful}")

