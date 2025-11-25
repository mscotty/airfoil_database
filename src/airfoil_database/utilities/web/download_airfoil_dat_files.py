import requests
from bs4 import BeautifulSoup
import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from urllib.parse import urljoin, urlparse  # For joining relative URLs
import numpy as np
from sqlmodel import Session

# --- Import from refactored code ---
from airfoil_database.core.database import AirfoilDatabase
from airfoil_database.core.models import Airfoil, AeroCoeff, AirfoilGeometry
from airfoil_database.classes.AirfoilSeries import AirfoilSeries
from airfoil_database.xfoil.processor import PointcloudProcessor
from airfoil_database.xfoil.fix_point_cloud import DEFAULT_FIXER_CONFIG
from airfoil_database.utilities.web.custom_parser import parse_file, validate_pointcloud
from airfoil_database.preprocessing.closure import ensure_pointcloud_closed

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# --- Scraping Function ---
def get_all_dat_links(start_url):
    """Scrapes the website starting from start_url to find all .dat file links."""
    all_links = set()
    current_url = start_url
    session = requests.Session()  # Use a session for potential connection reuse
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
    )  # Mimic browser

    logging.info(f"Starting link scraping from: {start_url}")
    page_count = 0
    while current_url:
        page_count += 1
        logging.info(f"Scraping page {page_count}: {current_url}")
        try:
            response = session.get(current_url, timeout=30)  # Add timeout
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            soup = BeautifulSoup(response.text, "html.parser")

            # Find .dat links on the current page
            found_on_page = 0
            for link in soup.find_all("a", href=True):
                href = link["href"]
                if href.endswith(".dat"):
                    # Construct absolute URL if necessary (assuming links are relative)
                    absolute_url = urljoin(current_url, href)
                    all_links.add(absolute_url)
                    found_on_page += 1
            logging.info(f"Found {found_on_page} .dat links on page {page_count}.")

            # Find the "Next" link
            next_page_link = soup.find(
                "a", string=lambda t: t and "next" in t.lower()
            )  # More robust search for "Next"
            if next_page_link and next_page_link.has_attr("href"):
                next_url_relative = next_page_link["href"]
                current_url = urljoin(current_url, next_url_relative)
                time.sleep(0.5)  # Small delay between page requests to be polite
            else:
                logging.info("No 'Next' link found or link invalid. Ending scraping.")
                current_url = None  # End loop

        except requests.exceptions.RequestException as e:
            logging.error(f"HTTP Error scraping {current_url}: {e}")
            current_url = None  # Stop if a page fails
        except Exception as e:
            logging.error(f"Error parsing page {current_url}: {e}")
            current_url = None  # Stop on parsing errors

    logging.info(f"Finished scraping. Found {len(all_links)} unique .dat links.")
    return list(all_links)


# --- Download Worker ---
def download_file(url, save_dir, session, overwrite=False):
    """Downloads a single file."""
    try:
        filename = os.path.basename(urlparse(url).path)  # Get filename from URL path
        filepath = os.path.join(save_dir, filename)

        # Skip if file already exists (optional, could add overwrite logic)
        if not overwrite and os.path.exists(filepath):
            logging.debug(f"Skipping download, file exists: {filepath}")
            return filepath

        response = session.get(url, stream=True, timeout=60)  # Use session, add timeout
        response.raise_for_status()
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(8192):  # Use slightly larger chunk size
                f.write(chunk)
        logging.debug(f"Downloaded: {filepath}")
        return filepath
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to download {url}: {e}")
        return None
    except Exception as e:
        logging.error(f"Error saving file from {url} to {filepath}: {e}")
        return None


# --- Concurrent Downloader ---
def download_all_files_concurrently(urls, save_dir, max_workers=10):
    """Downloads all files from the list of URLs concurrently."""
    downloaded_files = []
    # Use a single session for all downloads in the pool for connection reuse
    with requests.Session() as session:
        session.headers.update({"User-Agent": "Mozilla/5.0"})  # Set user agent
        logging.info(
            f"Starting concurrent download of {len(urls)} files with max {max_workers} workers..."
        )
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {
                executor.submit(download_file, url, save_dir, session): url
                for url in urls
            }
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    filepath = future.result()
                    if filepath:
                        downloaded_files.append(filepath)
                except Exception as exc:
                    logging.error(
                        f"URL {url} generated an exception during download task: {exc}"
                    )

    logging.info(
        f"Finished downloading. Successfully downloaded {len(downloaded_files)} files."
    )
    return downloaded_files


def format_pointcloud_array(points_array, precision=10):
    """Format a numpy array of points back to a string with specified precision."""
    try:
        if points_array is None or len(points_array) == 0:
            return None

        # Format each point to a string with specified precision
        formatted_lines = []
        for point in points_array:
            x, y = point
            formatted_lines.append(f"{x:.{precision}f} {y:.{precision}f}")

        # Join lines with newline characters
        return "\n".join(formatted_lines)
    except Exception as e:
        logging.error(f"Error formatting pointcloud array: {e}")
        return None


# --- Parsing Worker ---
def parse_file_worker(filepath):
    """Parses a single .dat file using the custom parser."""
    try:
        # Use the custom parser to read the file
        pointcloud_array = parse_file(filepath)

        # Validate the parsed pointcloud
        if not validate_pointcloud(pointcloud_array):
            logging.warning(f"Pointcloud validation failed for: {filepath}")
            return None  # Skip files that fail validation

        # Derive name from filename
        base_name = os.path.splitext(os.path.basename(filepath))[0]

        # Apply additional processing if needed using PointcloudProcessor
        try:
            # Convert pointcloud_array to string format for processing
            pointcloud_str = format_pointcloud_array(pointcloud_array)

            # Process the pointcloud if needed
            # processed_points_array = PointcloudProcessor.fix_airfoil_pointcloud(
            #     name=base_name,
            #     pointcloud_str=pointcloud_str,
            #     config=DEFAULT_FIXER_CONFIG,
            # )
            processed_points_array = ensure_pointcloud_closed(pointcloud_array)

            # Use processed points if available, otherwise use original
            if processed_points_array is not None:
                pointcloud_array = processed_points_array
        except Exception as process_err:
            logging.warning(
                f"Error during additional processing for {filepath}: {process_err}"
            )
            # Continue with the original parsed pointcloud

        # Format points to string for DB storage
        pointcloud_str = format_pointcloud_array(
            pointcloud_array, DEFAULT_FIXER_CONFIG.get("precision", 10)
        )
        if not pointcloud_str:
            logging.warning(f"Formatting parsed points failed for: {filepath}")
            return None

        # Identify series
        airfoil_series = AirfoilSeries.identify_airfoil_series(base_name)
        description = f"{base_name} Airfoil"
        source_url = f"file://{filepath}"  # Placeholder

        # Return a dictionary that matches the SQLModel Airfoil class fields
        return {
            "name": base_name,
            "description": description,
            "pointcloud": pointcloud_str,
            "airfoil_series": airfoil_series.value,
            "source": source_url,
        }

    except Exception as e:
        logging.error(f"Error parsing file {filepath}: {e}", exc_info=True)
        return None


# --- Concurrent Parser ---
def parse_all_files_concurrently(filepaths, max_workers=None):
    """Parses all downloaded .dat files concurrently."""
    parsed_data_list = []
    if not max_workers:
        max_workers = os.cpu_count()  # Default to number of CPU cores for parsing

    logging.info(
        f"Starting concurrent parsing of {len(filepaths)} files with max {max_workers} workers..."
    )
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_filepath = {
            executor.submit(parse_file_worker, filepath): filepath
            for filepath in filepaths
        }
        for future in as_completed(future_to_filepath):
            filepath = future_to_filepath[future]
            try:
                parsed_data = future.result()
                if parsed_data:
                    parsed_data_list.append(parsed_data)
            except Exception as exc:
                logging.error(
                    f"File {filepath} generated an exception during parsing task: {exc}",
                    exc_info=True,
                )

    logging.info(
        f"Finished parsing. Successfully parsed {len(parsed_data_list)} files."
    )
    return parsed_data_list


# --- Main Download and Store Function ---
def download_and_store_airfoils(
    start_url,
    save_dir="airfoil_dat_files",
    db_name="airfoil_data.db",
    db_dir="airfoil_database",
    overwrite=False,
    max_download_workers=10,
    max_parse_workers=None,
):
    """
    Main function to scrape, download, parse, and store airfoil data.
    """
    # 1. Scrape all links
    all_dat_urls = get_all_dat_links(start_url)
    if not all_dat_urls:
        logging.error("No .dat file links found. Exiting.")
        return

    # 2. Download all files concurrently
    os.makedirs(save_dir, exist_ok=True)
    downloaded_filepaths = download_all_files_concurrently(
        all_dat_urls, save_dir, max_download_workers
    )
    if not downloaded_filepaths:
        logging.error("No files were successfully downloaded. Exiting.")
        return

    # 3. Parse all downloaded files concurrently
    parsed_airfoil_data = parse_all_files_concurrently(
        downloaded_filepaths, max_parse_workers
    )
    if not parsed_airfoil_data:
        logging.error("No files were successfully parsed. Exiting.")
        return

    # 4. Store data in SQLModel database
    logging.info(
        f"Attempting to store data for {len(parsed_airfoil_data)} airfoils in the database."
    )
    airfoil_db = AirfoilDatabase(db_name=db_name, db_dir=db_dir)

    try:
        # Use the store_bulk_airfoil_data method which has been updated for SQLModel
        success_count = airfoil_db.store_bulk_airfoil_data(
            parsed_airfoil_data, overwrite=overwrite
        )
        logging.info(f"Successfully stored data for {success_count} airfoils.")
    except Exception as e:
        logging.error(f"An error occurred during database storage: {e}", exc_info=True)

        # Fallback to individual storage if bulk method fails
        logging.info("Attempting to store individually (slower)...")
        success_count = 0

        for data in parsed_airfoil_data:
            try:
                # Store individual airfoil
                airfoil_db.store_airfoil_data(
                    name=data["name"],
                    description=data["description"],
                    pointcloud=data["pointcloud"],
                    airfoil_series=AirfoilSeries.from_string(data["airfoil_series"]),
                    source=data["source"],
                    overwrite=overwrite,
                )
                success_count += 1
            except Exception as store_err:
                logging.error(
                    f"Failed to store airfoil {data.get('name', 'N/A')} individually: {store_err}"
                )

        logging.info(
            f"Successfully stored data for {success_count} airfoils individually."
        )

    logging.info("Finished processing all .dat files.")


def load_and_store_local_airfoils(
    save_dir="airfoil_dat_files",
    db_name="airfoil_data.db",
    db_dir="airfoil_database",
    overwrite=False,
    max_parse_workers=None,
):
    """
    Load airfoils from local directory and store in database (bypass downloading).

    Args:
        save_dir: Directory containing .dat files
        db_name: Name of the database file
        db_dir: Directory where database will be stored
        overwrite: Whether to overwrite existing data
        max_parse_workers: Number of workers for parsing (None = CPU count)
    """

    # 1. Find all .dat files in the save directory
    if not os.path.exists(save_dir):
        logging.error(
            f"Save directory '{save_dir}' does not exist. Please check the path."
        )
        return

    # Get all .dat files from the directory
    dat_files = []
    for filename in os.listdir(save_dir):
        if filename.lower().endswith(".dat"):
            filepath = os.path.join(save_dir, filename)
            dat_files.append(filepath)

    if not dat_files:
        logging.error(f"No .dat files found in directory '{save_dir}'. Exiting.")
        return

    logging.info(f"Found {len(dat_files)} .dat files in '{save_dir}'")

    # 2. Parse all files concurrently (same as original)
    parsed_airfoil_data = parse_all_files_concurrently(dat_files, max_parse_workers)

    if not parsed_airfoil_data:
        logging.error("No files were successfully parsed. Exiting.")
        return

    # 3. Store data in SQLModel database (same as original)
    logging.info(
        f"Attempting to store data for {len(parsed_airfoil_data)} airfoils in the database."
    )
    airfoil_db = AirfoilDatabase(db_name=db_name, db_dir=db_dir)

    try:
        # Use the store_bulk_airfoil_data method
        success_count = airfoil_db.store_bulk_airfoil_data(
            parsed_airfoil_data, overwrite=overwrite
        )
        logging.info(f"Successfully stored data for {success_count} airfoils.")
    except Exception as e:
        logging.error(f"An error occurred during database storage: {e}", exc_info=True)
        # Fallback to individual storage if bulk method fails
        logging.info("Attempting to store individually (slower)...")
        success_count = 0
        for data in parsed_airfoil_data:
            try:
                # Store individual airfoil
                airfoil_db.store_airfoil_data(
                    name=data["name"],
                    description=data["description"],
                    pointcloud=data["pointcloud"],
                    airfoil_series=AirfoilSeries.from_string(data["airfoil_series"]),
                    source=data["source"],
                    overwrite=overwrite,
                )
                success_count += 1
            except Exception as store_err:
                logging.error(
                    f"Failed to store airfoil {data.get('name', 'N/A')} individually: {store_err}"
                )

        logging.info(
            f"Successfully stored data for {success_count} airfoils individually."
        )

    logging.info("Finished processing all local .dat files.")
    return success_count


# --- Entry Point ---
if __name__ == "__main__":
    start_time = time.time()
    download = False
    load = True
    if download and load:
        download_and_store_airfoils(
            start_url="https://m-selig.ae.illinois.edu/ads/coord_database.html",
            save_dir="airfoil_dat_files",  # Directory for .dat files
            db_dir="airfoil_database",  # Directory for the SQLite DB
            db_name="airfoils.db",  # Specific DB name
            overwrite=True,  # Overwrite existing data in DB for this run
            max_download_workers=30,  # Increase download concurrency
            max_parse_workers=None,  # Use default (CPU count) for parsing
        )
    elif load:
        load_and_store_local_airfoils(
            save_dir="airfoil_dat_files",  # Directory containing your .dat files
            db_dir="airfoil_database",  # Directory for the SQLite DB
            db_name="airfoils.db",  # Specific DB name
            overwrite=True,  # Overwrite existing data in DB
            max_parse_workers=None,  # Use default (CPU count) for parsing
        )
    end_time = time.time()
    logging.info(f"Total execution time: {end_time - start_time:.2f} seconds")
