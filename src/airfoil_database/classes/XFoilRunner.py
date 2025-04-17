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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(process)d - %(levelname)s - %(message)s')

# --- Worker Function (Static/Top-Level) ---
# This function runs outside the class context in the worker process

def _run_xfoil_alpha_worker(xfoil_executable, airfoil_name, points_str, reynolds, mach, alpha, ncrit):
    """
    Static worker function to run XFoil for a single alpha.
    Designed to be picklable for multiprocessing.

    Returns:
        tuple: (alpha, dict_of_coeffs) or (alpha, None) if failed.
               dict_of_coeffs contains {'Cl': float, 'Cd': float, 'Cm': float}
    """
    temp_airfoil_path = None
    output_file_path = None
    # Timeout for XFoil process communication (seconds)
    XFOIL_TIMEOUT = 60 # Adjust as needed

    try:
        # --- Create Temporary Files ---
        # Write airfoil points
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".dat", mode="w", encoding='utf-8') as temp_file:
                temp_file.write(points_str)
                temp_airfoil_path = temp_file.name
        except Exception as e:
            logging.error(f"[Worker] Error creating temp airfoil file: {e}")
            return alpha, None

        # Create empty temp output file
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pol", mode="w", encoding='utf-8') as temp_out_file:
                output_file_path = temp_out_file.name
            # Ensure it's truly empty (paranoia check)
            if os.path.exists(output_file_path):
                os.remove(output_file_path)
        except Exception as e:
            logging.error(f"[Worker] Error creating temp output file: {e}")
            # Cleanup airfoil file if output file failed
            if temp_airfoil_path and os.path.exists(temp_airfoil_path):
                 try: os.remove(temp_airfoil_path)
                 except OSError: pass
            return alpha, None


        # --- Construct XFoil Commands ---
        # Corrected commands: NCRIT under OPER, removed VPAR, removed second PACC
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
        process = None # Initialize process to None
        try:
            process = subprocess.Popen(
                [xfoil_executable], # Use the executable path passed as argument
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='ignore'
            )
            # Communicate with a timeout
            stdout, stderr = process.communicate(input=xfoil_input, timeout=XFOIL_TIMEOUT)
            end_time = time.time()
            run_duration = end_time - start_time

        except subprocess.TimeoutExpired:
             logging.warning(f"[Worker] XFoil process timed out ({XFOIL_TIMEOUT}s) for {airfoil_name} A={alpha}. Terminating.")
             if process:
                 process.terminate() # Try to terminate
                 try:
                     # Wait briefly for termination
                     process.wait(timeout=5)
                 except subprocess.TimeoutExpired:
                     process.kill() # Force kill if terminate doesn't work
             return alpha, None
        except FileNotFoundError:
            # This error should only happen once if the executable is wrong
            logging.error(f"[Worker] XFoil executable not found at '{xfoil_executable}'.")
            # Re-raise to potentially stop the whole run? Or just fail this task?
            # For now, just fail this task. The main process might catch repeated errors.
            return alpha, None # Fail this specific alpha run
        except Exception as popen_err:
             logging.error(f"[Worker] Error during Popen/communicate for {airfoil_name} A={alpha}: {popen_err}")
             return alpha, None

        # --- Process Results ---
        if process.returncode != 0:
            logging.warning(f"[Worker] XFoil process error (Code {process.returncode}) for {airfoil_name} Re={reynolds} M={mach} A={alpha}. Stderr: {stderr.strip()}")
            return alpha, None

        # Parse the generated polar file (use static method or include logic)
        coefficients = XFoilRunner._parse_polar_file(output_file_path) # Call static method

        if coefficients:
             logging.debug(f"[Worker] Success: {airfoil_name} Re={reynolds} M={mach} A={alpha} -> {coefficients} (Time: {run_duration:.2f}s)")
             return alpha, coefficients
        else:
             # Log stdout only if parsing failed, might contain clues
             logging.warning(f"[Worker] Failed to parse/find coeffs for {airfoil_name} Re={reynolds} M={mach} A={alpha}. Stdout: {stdout.strip()}")
             return alpha, None

    except Exception as e:
        # Catch-all for unexpected errors within the worker task
        logging.error(f"[Worker] Unexpected error processing {airfoil_name} A={alpha}: {e}", exc_info=True)
        return alpha, None
    finally:
        # --- Cleanup Temporary Files ---
        for f_path in [temp_airfoil_path, output_file_path]:
            if f_path and os.path.exists(f_path):
                try:
                    os.remove(f_path)
                except Exception as e:
                    # Log warning but don't crash the worker
                    logging.warning(f"[Worker] Could not remove temp file {f_path}: {e}")

# --- XFoilRunner Class ---

class XFoilRunner:
    """
    Runs XFoil simulations for individual angles of attack, handling specific
    setup commands and integrating with an AirfoilDatabase. Incorporates
    parallel processing for different run conditions.
    """

    def __init__(self, database, xfoil_executable=None):
        """
        Initializes the XFoilRunner. Args remain the same.
        """
        self.database = database # Needed for storing results in the main process
        self.xfoil_executable = xfoil_executable if xfoil_executable else "xfoil"
        logging.info(f"Using XFoil executable: {self.xfoil_executable}")
        # No instance variables that would cause pickling issues are stored here.

    # Make parsing static as it doesn't depend on instance state
    @staticmethod
    def _parse_polar_file(filepath):
        """
        Parses the XFoil polar output file (.pol) generated by PACC.
        (Implementation is the same as before, just marked static)
        """
        try:
            # Read the file, skipping header lines. Header length can vary slightly.
            # We look for the line starting with 'alpha'
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()

            data_start_line = -1
            for i, line in enumerate(lines):
                # More robust header check
                cleaned_line = line.strip().lower()
                if cleaned_line.startswith('alpha') and 'cl' in cleaned_line and 'cd' in cleaned_line:
                    # Data usually starts 2 lines after the header
                    data_start_line = i + 2
                    break

            if data_start_line == -1 or data_start_line >= len(lines):
                logging.debug(f"Could not find data header or data lines in {filepath}")
                return None # Header not found or no data lines

            # Extract the first data line (should be the only one for single alpha)
            data_line = lines[data_start_line].strip()
            values = data_line.split()

            # Expecting at least alpha, Cl, Cd, Cdp, Cm (5 values)
            if len(values) >= 5:
                # Add explicit error handling for float conversion
                try:
                    cl = float(values[1])
                    cd = float(values[2])
                    cm = float(values[4]) # Cm is typically the 5th value
                    return {"Cl": cl, "Cd": cd, "Cm": cm}
                except ValueError as ve:
                    logging.warning(f"Could not convert values to float in {filepath}: {values}. Error: {ve}")
                    return None
            else:
                logging.debug(f"Insufficient values in data line of {filepath}: {data_line}")
                return None

        except FileNotFoundError:
             # This might happen if XFoil failed to write the file
             logging.warning(f"Polar output file not found: {filepath}")
             return None
        except Exception as e:
            # Catch other potential errors like permission issues
            logging.error(f"Unexpected error reading or parsing polar file {filepath}: {e}")
            return None

    # This is the function submitted to the executor - now static
    @staticmethod
    def _run_condition_set_static(xfoil_executable, airfoil_name, points_str, reynolds, mach, ncrit, alphas):
        """
        Static worker function to run all alphas for a specific (Re, Mach, Ncrit) set.
        Calls the global _run_xfoil_alpha_worker function.
        """
        successful_results = []
        logging.info(f"[Worker Start] Analyzing {airfoil_name}, Re={reynolds}, M={mach}, Ncrit={ncrit}")

        for alpha in alphas:
            # Call the top-level worker function
            _, coeffs = _run_xfoil_alpha_worker(
                xfoil_executable, airfoil_name, points_str, reynolds, mach, alpha, ncrit
            )
            if coeffs:
                successful_results.append({'alpha': alpha, **coeffs}) # Combine alpha with coeffs dict

        logging.info(f"[Worker End] Finished {airfoil_name}, Re={reynolds}, M={mach}, Ncrit={ncrit}. Successes: {len(successful_results)}/{len(alphas)}")
        # Return results along with identifying info needed by the main process
        return airfoil_name, reynolds, mach, ncrit, successful_results


    def run_analysis_parallel(self,
                              airfoil_name,
                              reynolds_list,
                              mach_list,
                              alpha_list,
                              ncrit_list,
                              max_workers=None):
        """
        Runs XFoil analysis in parallel using static worker functions.
        """
        # --- Input Validation and Setup --- (Same as before)
        if not isinstance(reynolds_list, list): reynolds_list = [reynolds_list]
        if not isinstance(mach_list, list): mach_list = [mach_list]
        if not isinstance(alpha_list, list): alpha_list = [alpha_list]
        if not isinstance(ncrit_list, list): ncrit_list = [ncrit_list]

        start_total_time = time.time()
        logging.info(f"Starting parallel analysis for airfoil: {airfoil_name}")

        airfoil_db_data = self.database.get_airfoil_data(airfoil_name)
        if not airfoil_db_data or not airfoil_db_data[1]:
            logging.error(f"Airfoil '{airfoil_name}' not found or has no pointcloud data. Skipping.")
            return
        _, points_str, _, _ = airfoil_db_data

        if not max_workers:
            cores = os.cpu_count()
            max_workers = cores if cores else 1
            logging.info(f"Using default max_workers: {max_workers}")

        # --- Prepare Tasks ---
        tasks_args = []
        for reynolds in reynolds_list:
            for mach in mach_list:
                for ncrit in ncrit_list:
                    # Arguments for the static worker function
                    tasks_args.append((
                        self.xfoil_executable, # Pass executable path
                        airfoil_name,
                        points_str,
                        reynolds,
                        mach,
                        ncrit,
                        alpha_list
                    ))

        if not tasks_args:
            logging.warning("No simulation tasks generated. Check input lists.")
            return

        logging.info(f"Submitting {len(tasks_args)} condition sets to ProcessPoolExecutor with {max_workers} workers.")

        # --- Execute Tasks and Collect Results ---
        total_alphas_processed = 0
        total_alphas_successful = 0

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit the static method with explicit args
            # We don't need future_to_condition anymore as results contain identifying info
            futures = [executor.submit(XFoilRunner._run_condition_set_static, *args) for args in tasks_args]

            for future in as_completed(futures):
                try:
                    # Unpack results from the static worker
                    res_airfoil, res_re, res_mach, res_ncrit, successful_results = future.result()

                    # Increment processed count based on the alpha list for that task
                    # Find the original task args to know how many alphas were attempted
                    # This is a bit less direct than future_to_condition, alternative below
                    num_alphas_in_task = len(alpha_list) # Assuming alpha_list is constant for now
                    total_alphas_processed += num_alphas_in_task

                    # Store results using self.database (in the main process)
                    if successful_results:
                        num_success = len(successful_results)
                        total_alphas_successful += num_success
                        logging.info(f"Received {num_success} results for {res_airfoil}, Re={res_re}, M={res_mach}, Ncrit={res_ncrit}. Storing...")
                        for result in successful_results:
                            try:
                                # Call the database method from the main process
                                self.database.store_aero_coeffs(
                                    res_airfoil,
                                    res_re,
                                    res_mach,
                                    res_ncrit,
                                    result['alpha'],
                                    result['Cl'],
                                    result['Cd'],
                                    result['Cm']
                                )
                            except Exception as db_err:
                                logging.error(f"Database error storing result for {res_airfoil} Re={res_re} M={res_mach} A={result['alpha']}: {db_err}")
                    else:
                         logging.info(f"No successful results returned for {res_airfoil}, Re={res_re}, M={res_mach}, Ncrit={res_ncrit}")

                except Exception as exc:
                    # Log exceptions raised by the worker process OR during future.result()
                    # Getting original args is harder without future_to_condition, log generically
                    logging.error(f"A worker task generated an exception: {exc}", exc_info=True)

        # --- Final Logging ---
        end_total_time = time.time()
        logging.info(f"Finished parallel analysis for {airfoil_name}.")
        logging.info(f"Total time: {end_total_time - start_total_time:.2f} seconds")
        # Note: total_alphas_processed might be slightly inaccurate if alpha_list varied per task
        logging.info(f"Total alphas attempted (approx): {total_alphas_processed}")
        logging.info(f"Total successful alphas stored: {total_alphas_successful}")

