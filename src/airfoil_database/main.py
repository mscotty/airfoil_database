# main.py
import argparse
import logging
import os
import time

# --- Database and Analyzers ---
# These are the core components we'll be orchestrating.
from airfoil_database.core.database import AirfoilDatabase
from airfoil_database.analysis.geometry_analyzer import GeometryAnalyzer
from airfoil_database.analysis.aero_analyzer import AeroAnalyzer
from airfoil_database.xfoil.processor import PointcloudProcessor

# --- Workflow Functions ---
# We import the main function from your scraping script to act as a module.
from airfoil_database.utilities.web.download_airfoil_dat_files import download_and_store_airfoils

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
DB_DIR = "airfoil_database"
DB_NAME = "airfoils.db"
DAT_FILE_DIR = "airfoil_dat_files"
START_URL = "https://m-selig.ae.illinois.edu/ads/coord_database.html"


def run_populate(args):
    """Phase 1: Scrape, download, process, and store all airfoil data."""
    logging.info("--- Starting Phase 1: Populate Database ---")
    start_time = time.time()
    # This function, imported from your script, runs the entire data acquisition pipeline.
    download_and_store_airfoils(
        start_url=START_URL,
        save_dir=DAT_FILE_DIR,
        db_dir=DB_DIR,
        db_name=DB_NAME,
        overwrite=args.overwrite,
        max_download_workers=30,
        max_parse_workers=os.cpu_count()
    )
    end_time = time.time()
    logging.info(f"--- Population Phase finished in {end_time - start_time:.2f} seconds ---")


def run_analyze_geometry(args):
    """Phase 2: Compute and store geometry metrics for all airfoils."""
    logging.info("--- Starting Phase 2: Analyze Geometry ---")
    start_time = time.time()
    db = AirfoilDatabase(db_name=DB_NAME, db_dir=DB_DIR)
    geom_analyzer = GeometryAnalyzer(database=db)
    # This method iterates through all airfoils and saves their geometric properties.
    geom_analyzer.compute_geometry_metrics()
    end_time = time.time()
    logging.info(f"--- Geometry Analysis finished in {end_time - start_time:.2f} seconds ---")


def run_analyze_aero(args):
    """Phase 3: Instructions for the manual aerodynamic analysis step."""
    logging.info("--- Phase 3: Aerodynamic Analysis (Manual Step) ---")
    print(f"""
    Aerodynamic analysis requires an external tool like XFOIL.
    The provided code can STORE the results, but cannot GENERATE them.

    Your workflow should be:
    1. For each airfoil, run an XFOIL simulation across a range of Reynolds numbers and angles of attack.
    2. Create a Python script to parse the XFOIL output files.
    3. In that script, import AeroAnalyzer and AirfoilDatabase.
    4. Instantiate the analyzer and store each result using `store_aero_coeffs`:

       db = AirfoilDatabase(db_name='{DB_NAME}', db_dir='{DB_DIR}')
       aero_analyzer = AeroAnalyzer(database=db)

       # Example for one data point:
       aero_analyzer.store_aero_coeffs(
           name='naca0012', reynolds_number=500000, mach=0.0, ncrit=9.0,
           alpha=5.0, cl=0.543, cd=0.009, cm=-0.002
       )
    """)
    logging.info("--- End of Phase 3 Instructions ---")


def run_find_similar(args):
    """Phase 4a: Find airfoils similar to an input .dat file."""
    logging.info(f"--- Finding airfoils similar to {args.filepath} ---")
    if not os.path.exists(args.filepath):
        logging.error(f"File not found: {args.filepath}")
        return

    with open(args.filepath, 'r') as f:
        pointcloud_str = f.read()

    db = AirfoilDatabase(db_name=DB_NAME, db_dir=DB_DIR)
    processor = PointcloudProcessor()

    # This function compares an input pointcloud to all in the database.
    matches = processor.find_best_matching_airfoils(
        input_pointcloud_str=pointcloud_str,
        database=db,
        num_matches=args.num_matches
    )

    if matches:
        print(f"Top {len(matches)} matches for {os.path.basename(args.filepath)}:")
        for name, distance in matches:
            print(f"  - {name} (Distance: {distance:.6f})")
    else:
        print("No suitable matches found in the database.")


def run_find_by_geo(args):
    """Phase 4b: Find airfoils by geometric parameters."""
    logging.info(f"--- Finding airfoils by geometry: {args.parameter} ~ {args.value} ---")
    db = AirfoilDatabase(db_name=DB_NAME, db_dir=DB_DIR)
    geom_analyzer = GeometryAnalyzer(database=db)
    
    # This function queries the database for airfoils matching the geometric criteria.
    geom_analyzer.find_airfoils_by_geometry(
        parameter=args.parameter,
        target_value=args.value,
        tolerance=args.tolerance,
        tolerance_type=args.type
    )


def run_find_by_aero(args):
    """Phase 4c: Find airfoils by aerodynamic parameters."""
    logging.info(f"--- Finding airfoils by aero performance: {args.parameter} ~ {args.value} ---")
    db = AirfoilDatabase(db_name=DB_NAME, db_dir=DB_DIR)
    aero_analyzer = AeroAnalyzer(database=db)

    # This function queries the database for airfoils matching the aero criteria.
    results = aero_analyzer.find_airfoils_by_xfoil_results(
        parameter=args.parameter,
        target_value=args.value,
        tolerance=args.tolerance,
        tolerance_type=args.type
    )
    if results:
        print(f"Found {len(results)} airfoils matching the criteria:")
        for res_tuple in results:
            print(f"  - {res_tuple[0]}")

def main():
    """Main function to parse command-line arguments and run the selected workflow."""
    parser = argparse.ArgumentParser(description="Airfoil Database Workflow Manager.")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # --- Parser for 'populate' command ---
    parser_populate = subparsers.add_parser("populate", help="Phase 1: Scrape, process, and populate the database.")
    parser_populate.add_argument("--overwrite", action="store_true", help="Overwrite existing data in the database.")
    parser_populate.set_defaults(func=run_populate)

    # --- Parser for 'analyze-geometry' command ---
    parser_analyze_geo = subparsers.add_parser("analyze-geometry", help="Phase 2: Compute and store geometric metrics.")
    parser_analyze_geo.set_defaults(func=run_analyze_geometry)

    # --- Parser for 'analyze-aero' command ---
    parser_analyze_aero = subparsers.add_parser("analyze-aero", help="Phase 3: Show instructions for aero analysis.")
    parser_analyze_aero.set_defaults(func=run_analyze_aero)
    
    # --- Parser for 'find' command ---
    parser_find = subparsers.add_parser("find", help="Phase 4: Query the database to find airfoils.")
    find_subparsers = parser_find.add_subparsers(dest="find_command", required=True)

    # Sub-parser for 'find similar'
    parser_find_similar = find_subparsers.add_parser("similar", help="Find airfoils similar to an input file.")
    parser_find_similar.add_argument("filepath", type=str, help="Path to the input .dat file.")
    parser_find_similar.add_argument("-n", "--num_matches", type=int, default=5, help="Number of matches to return.")
    parser_find_similar.set_defaults(func=run_find_similar)

    # Sub-parser for 'find by-geo'
    parser_find_geo = find_subparsers.add_parser("by-geo", help="Find airfoils by a geometric parameter.")
    parser_find_geo.add_argument("parameter", type=str, help="e.g., max_thickness, leading_edge_radius")
    parser_find_geo.add_argument("value", type=float, help="Target value for the parameter.")
    parser_find_geo.add_argument("--tolerance", type=float, default=0.01)
    parser_find_geo.add_argument("--type", type=str, choices=["absolute", "percentage"], default="absolute")
    parser_find_geo.set_defaults(func=run_find_by_geo)
    
    # Sub-parser for 'find by-aero'
    parser_find_aero = find_subparsers.add_parser("by-aero", help="Find airfoils by an aerodynamic parameter.")
    parser_find_aero.add_argument("parameter", type=str, help="e.g., cl, cd, reynolds_number")
    parser_find_aero.add_argument("value", type=float, help="Target value for the parameter.")
    parser_find_aero.add_argument("--tolerance", type=float, default=0.05)
    parser_find_aero.add_argument("--type", type=str, choices=["absolute", "percentage"], default="absolute")
    parser_find_aero.set_defaults(func=run_find_by_aero)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()