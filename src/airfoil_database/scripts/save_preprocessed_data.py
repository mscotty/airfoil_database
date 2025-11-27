# scripts/save_preprocessed_data.py
"""
Save Week 1 preprocessed results to airfoils_preprocessed.db
Supporting transition to Week 2 classical modeling phase.
"""

import numpy as np
import json
from pathlib import Path
from airfoil_database.core.database import AirfoilDatabase
from airfoil_database.core.models import Airfoil
import logging


def save_preprocessed_to_database():
    """
    Save preprocessed 1000-point standardized airfoils to new database.
    Supports document Table 3. Feature Summary requirements.
    """

    # Load preprocessed data from Week 1
    processed_data_dir = "processed_data"

    # Load the standardized point clouds and names
    try:
        point_clouds = np.load(f"{processed_data_dir}/airfoil_pointclouds_1000pts.npy")
        airfoil_names = np.load(f"{processed_data_dir}/airfoil_names.npy")

        logging.info(f"Loaded {len(point_clouds)} preprocessed airfoils")

    except FileNotFoundError:
        logging.error("Preprocessed data not found. Run Week 1 data preparation first.")
        return False

    # Create new preprocessed database
    preprocessed_db = AirfoilDatabase(db_name="airfoils_preprocessed.db", db_dir=".")

    # Prepare bulk data for insertion
    bulk_data = []

    for i, airfoil_name in enumerate(airfoil_names):
        # Convert numpy array to string format for storage
        point_cloud_array = point_clouds[i]

        # Format as space-separated coordinate pairs
        pointcloud_str = "\n".join(
            [f"{point[0]:.6f} {point[1]:.6f}" for point in point_cloud_array]
        )

        bulk_data.append(
            {
                "name": str(airfoil_name),
                "description": f"Preprocessed standardized airfoil with 1000 points",
                "pointcloud": pointcloud_str,
                "airfoil_series": "UNKNOWN",  # Will be determined later
                "source": "Week1_Preprocessing_Pipeline",
            }
        )

    # Bulk insert preprocessed data
    logging.info(f"Saving {len(bulk_data)} preprocessed airfoils to database...")
    inserted_count = preprocessed_db.store_bulk_airfoil_data(bulk_data, overwrite=True)

    logging.info(
        f"Successfully saved {inserted_count} airfoils to airfoils_preprocessed.db"
    )

    # Update airfoil series classification
    preprocessed_db.update_airfoil_series()

    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = save_preprocessed_to_database()
    if success:
        print("✅ Preprocessed data successfully saved to airfoils_preprocessed.db")
        print("Ready for Week 2 geometric analysis and classical modeling")
    else:
        print("❌ Failed to save preprocessed data")
