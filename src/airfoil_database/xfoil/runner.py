# xfoil/runner.py
import os
import logging
import tempfile
import subprocess
import concurrent.futures
import threading
from typing import List, Optional, Any

class XFoilRunner:
    """Class to run XFoil analysis on airfoils."""
    
    def __init__(self, database, xfoil_executable=None):
        """
        Initialize the XFoil runner.
        
        Args:
            database: The airfoil database instance
            xfoil_executable (str, optional): Path to XFoil executable. If None, tries to find it in PATH.
        """
        self.database = database
        self.xfoil_executable = xfoil_executable or "xfoil"
        self._lock = threading.Lock()  # Lock for thread-safe operations
        
        # Verify XFoil exists
        self._verify_xfoil_exists()
    
    def _verify_xfoil_exists(self):
        """Verify that the XFoil executable exists and is accessible."""
        try:
            # Try to run XFoil with the version flag
            subprocess.run([self.xfoil_executable], 
                          input=b"QUIT\n", 
                          stdout=subprocess.PIPE, 
                          stderr=subprocess.PIPE,
                          timeout=2,
                          check=False)
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            logging.error(f"XFoil executable not found or not working: {e}")
            raise FileNotFoundError(f"XFoil executable not found at: {self.xfoil_executable}")
    
    def run_analysis(self, airfoil_name, reynolds, mach, alpha, ncrit):
        """
        Run XFoil analysis for a single set of parameters.
        
        Args:
            airfoil_name (str): Name of the airfoil
            reynolds (float): Reynolds number
            mach (float): Mach number
            alpha (float): Angle of attack
            ncrit (float): Transition criterion
            
        Returns:
            tuple: (cl, cd, cm) coefficients or (None, None, None) if analysis fails
        """
        # Get airfoil data from database
        airfoil_data = self.database.get_airfoil_data(airfoil_name)
        if not airfoil_data:
            logging.error(f"Airfoil {airfoil_name} not found in database")
            return None, None, None
        
        _, pointcloud, _, _ = airfoil_data
        
        # Create a temporary directory for the files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Write airfoil coordinates to a file
            airfoil_file = os.path.join(temp_dir, f"{airfoil_name}.dat")
            with open(airfoil_file, 'w') as f:
                f.write(f"{airfoil_name}\n")
                f.write(pointcloud)
            
            # Create XFoil command file
            command_file = os.path.join(temp_dir, "commands.txt")
            with open(command_file, 'w') as f:
                f.write(f"LOAD {airfoil_file}\n")
                f.write("PANE\n")  # Panelize airfoil
                f.write("OPER\n")  # Enter OPER menu
                f.write(f"VISC {reynolds}\n")  # Set Reynolds number
                f.write(f"MACH {mach}\n")  # Set Mach number
                f.write(f"ITER 100\n")  # Set max iterations
                f.write(f"VPAR\n")  # Enter VPAR menu
                f.write(f"N {ncrit}\n")  # Set Ncrit
                f.write("\n")  # Return to OPER menu
                f.write(f"ALFA {alpha}\n")  # Set angle of attack
                f.write("DUMP\n")  # Dump polar data
                f.write("CPWR temp.cp\n")  # Write pressure coefficient
                f.write("\n")  # Exit OPER menu
                f.write("QUIT\n")  # Exit XFoil
            
            # Run XFoil with the command file
            try:
                process = subprocess.run(
                    [self.xfoil_executable],
                    input=open(command_file, 'rb').read(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=30,  # 30 second timeout
                    cwd=temp_dir
                )
                
                # Check for successful execution
                if process.returncode != 0:
                    logging.warning(f"XFoil process returned non-zero exit code: {process.returncode}")
                
                # Parse the output for cl, cd, cm
                stdout = process.stdout.decode('utf-8')
                cl, cd, cm = self._parse_xfoil_output(stdout)
                
                return cl, cd, cm
                
            except subprocess.TimeoutExpired:
                logging.error(f"XFoil analysis timed out for {airfoil_name} at Re={reynolds}, M={mach}, α={alpha}")
                return None, None, None
            except Exception as e:
                logging.error(f"Error running XFoil analysis: {e}")
                return None, None, None
    
    def _parse_xfoil_output(self, output_text):
        """
        Parse XFoil output to extract cl, cd, and cm values.
        
        Args:
            output_text (str): The XFoil output text
            
        Returns:
            tuple: (cl, cd, cm) or (None, None, None) if parsing fails
        """
        try:
            # Look for lines containing cl, cd, cm data
            for line in output_text.splitlines():
                if "CL =" in line and "CD =" in line and "CM =" in line:
                    # Extract values using string manipulation
                    parts = line.split()
                    cl_idx = parts.index("CL") + 2  # CL =
                    cd_idx = parts.index("CD") + 2  # CD =
                    cm_idx = parts.index("CM") + 2  # CM =
                    
                    cl = float(parts[cl_idx])
                    cd = float(parts[cd_idx])
                    cm = float(parts[cm_idx])
                    
                    return cl, cd, cm
            
            # If we reach here, we didn't find the data
            logging.warning("Could not find CL, CD, CM values in XFoil output")
            return None, None, None
            
        except (ValueError, IndexError) as e:
            logging.error(f"Error parsing XFoil output: {e}")
            return None, None, None
    
    def run_analysis_parallel(self, airfoil_name, reynolds_list, mach_list, alpha_list, ncrit_list, max_workers=None):
        """
        Run XFoil analysis in parallel for multiple parameter combinations.
        
        Args:
            airfoil_name (str): Name of the airfoil
            reynolds_list (list): List of Reynolds numbers
            mach_list (list or float): List of Mach numbers or single value
            alpha_list (list or float): List of angles of attack or single value
            ncrit_list (list or float): List of transition criteria or single value
            max_workers (int, optional): Maximum number of parallel workers
            
        Returns:
            bool: True if at least one analysis succeeded, False otherwise
        """
        # Convert single values to lists for consistent processing
        if not isinstance(reynolds_list, list):
            reynolds_list = [reynolds_list]
        if not isinstance(mach_list, list):
            mach_list = [mach_list]
        if not isinstance(alpha_list, list):
            alpha_list = [alpha_list]
        if not isinstance(ncrit_list, list):
            ncrit_list = [ncrit_list]
        
        # Create all parameter combinations
        tasks = []
        for re in reynolds_list:
            for m in mach_list:
                for a in alpha_list:
                    for n in ncrit_list:
                        tasks.append((re, m, a, n))
        
        logging.info(f"Running {len(tasks)} XFoil analyses for {airfoil_name}")
        
        # Run analyses in parallel
        success_count = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_params = {
                executor.submit(self._run_and_store_analysis, airfoil_name, re, m, a, n): (re, m, a, n)
                for re, m, a, n in tasks
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_params):
                params = future_to_params[future]
                try:
                    success = future.result()
                    if success:
                        success_count += 1
                except Exception as e:
                    logging.error(f"Exception during analysis of {airfoil_name} with params {params}: {e}")
        
        logging.info(f"Completed {success_count}/{len(tasks)} successful analyses for {airfoil_name}")
        return success_count > 0
    
    def _run_and_store_analysis(self, airfoil_name, reynolds, mach, alpha, ncrit):
        """
        Run a single analysis and store the results in the database.
        
        Args:
            airfoil_name (str): Name of the airfoil
            reynolds (float): Reynolds number
            mach (float): Mach number
            alpha (float): Angle of attack
            ncrit (float): Transition criterion
            
        Returns:
            bool: True if analysis succeeded, False otherwise
        """
        cl, cd, cm = self.run_analysis(airfoil_name, reynolds, mach, alpha, ncrit)
        
        if cl is None and cd is None and cm is None:
            return False
        
        # Store results in database
        with self._lock:  # Use lock to prevent concurrent database writes
            self.database.store_aero_coeffs(
                name=airfoil_name,
                reynolds_number=reynolds,
                mach=mach,
                ncrit=ncrit,
                alpha=alpha,
                cl=cl,
                cd=cd,
                cm=cm
            )
        
        return True
