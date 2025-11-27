# scripts/run_geometric_analysis.py
"""
Week 2 Geometric Analysis using GeometryAnalyzer class.
Supports document Table 4 classical modeling requirements.
"""

import logging
from pathlib import Path
from airfoil_database.core.database import AirfoilDatabase
from airfoil_database.analysis.geometry_analyzer import GeometryAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
from airfoil_database.scripts.save_preprocessed_data import (
    save_preprocessed_to_database,
)


def run_comprehensive_geometric_analysis():
    """
    Execute comprehensive geometric analysis supporting document requirements:
    - Table 3. Feature Summary generation
    - Classical modeling feature preparation
    - Figure 2 geometric distribution analysis
    """

    logging.info("Starting Week 2: Comprehensive Geometric Analysis")
    logging.info("Supporting HW4/HW5 classical modeling requirements")

    # Initialize preprocessed database
    try:
        preprocessed_db = AirfoilDatabase(
            db_name="airfoils_preprocessed.db", db_dir="."
        )
        logging.info("Connected to preprocessed database")
    except Exception as e:
        logging.error(f"Failed to connect to preprocessed database: {e}")
        logging.info("Creating preprocessed database from Week 1 data...")

        if not save_preprocessed_to_database():
            raise Exception("Failed to create preprocessed database")
        preprocessed_db = AirfoilDatabase(
            db_name="airfoils_preprocessed.db", db_dir="."
        )

    # Initialize geometry analyzer
    analyzer = GeometryAnalyzer(preprocessed_db)

    # Step 1: Compute geometric metrics for all airfoils
    logging.info("Computing geometric metrics for 1608 standardized airfoils...")
    processed_count = analyzer.compute_geometry_metrics_parallel(
        num_processes=4, batch_size=50  # Adjust based on available CPUs
    )

    logging.info(f"Successfully computed metrics for {processed_count} airfoils")

    # Step 2: Generate Table 3 Feature Summary data
    logging.info("Generating Table 3 Feature Summary for document...")
    feature_summary = generate_table3_feature_summary(preprocessed_db)

    # Step 3: Create Figure 2 geometric distribution plots
    logging.info("Creating Figure 2 geometric distribution visualizations...")
    create_figure2_distributions(preprocessed_db)

    # Step 4: Outlier detection and validation
    logging.info("Performing outlier detection on preprocessed data...")
    outliers = analyzer.check_all_pointcloud_outliers(threshold=3.0)

    # Step 5: Generate classical modeling feature matrix
    logging.info("Preparing features for classical modeling algorithms...")
    features_df = prepare_classical_modeling_features(preprocessed_db)

    # Save results
    output_dir = Path("geometric_analysis_results")
    output_dir.mkdir(exist_ok=True)

    feature_summary.to_csv(output_dir / "table3_feature_summary.csv", index=False)
    features_df.to_csv(output_dir / "classical_modeling_features.csv", index=False)

    print(f"\n=== Geometric Analysis Complete ===")
    print(f"Processed: {processed_count} airfoils")
    print(
        f"Features extracted: {len(features_df.columns)-1}"
    )  # -1 for airfoil name column
    print(f"Outliers detected: {len(outliers)} airfoils")
    print(f"Ready for classical modeling phase")
    print(f"Results saved to: {output_dir}")

    return features_df


def generate_table3_feature_summary(db):
    """Generate Table 3. Feature Summary as required by document."""

    # Get airfoil geometry data
    geometry_df = db.get_airfoil_geometry_dataframe()

    # Get point cloud statistics
    airfoil_df = db.get_airfoil_dataframe()

    # Create feature summary matching document requirements
    feature_data = []

    # Add coordinate features (x, y for each of 1000 points)
    feature_data.append(
        {
            "Feature Name": "Point Cloud Coordinates",
            "Description": "Normalized x,y coordinates for 1000 points per airfoil",
            "Data Type": "Continuous",
            "Range/Categories": "[0.0, 1.0] for x; [-0.5, 0.5] typical for y",
            "Missing Values": "0%",
            "Feature Count": "2000 (1000 x-coords + 1000 y-coords)",
        }
    )

    # Add geometric parameter features
    geometric_features = [
        ("max_thickness", "Maximum thickness", "Continuous", "Airfoil geometry"),
        ("max_camber", "Maximum camber", "Continuous", "Airfoil geometry"),
        (
            "leading_edge_radius",
            "Leading edge radius",
            "Continuous",
            "Airfoil geometry",
        ),
        (
            "trailing_edge_angle",
            "Trailing edge angle",
            "Continuous",
            "Airfoil geometry",
        ),
        (
            "thickness_to_chord_ratio",
            "Thickness to chord ratio",
            "Continuous",
            "Airfoil geometry",
        ),
    ]

    for feature, description, data_type, category in geometric_features:
        if feature in geometry_df.columns:
            values = geometry_df[feature].dropna()
            if len(values) > 0:
                feature_data.append(
                    {
                        "Feature Name": feature,
                        "Description": description,
                        "Data Type": data_type,
                        "Range/Categories": f"[{values.min():.4f}, {values.max():.4f}]",
                        "Missing Values": f"{geometry_df[feature].isna().sum()} ({(geometry_df[feature].isna().sum()/len(geometry_df)*100):.1f}%)",
                        "Feature Count": "1",
                    }
                )

    # Add target variable
    feature_data.append(
        {
            "Feature Name": "Airfoil ID",
            "Description": "Unique identifier for each of 1608 airfoil classes",
            "Data Type": "Categorical",
            "Range/Categories": "1608 unique classes (0-1607)",
            "Missing Values": "0%",
            "Feature Count": "1 (target variable)",
        }
    )

    table3_df = pd.DataFrame(feature_data)

    print("\nTable 3. Feature Summary")
    print(table3_df.to_string(index=False))

    return table3_df


def create_figure2_distributions(db):
    """
    Create Figure 2 geometric distribution plots as specified in document.
    "Histograms illustrating airfoil calculated maximum thickness, t/c ratio,
    maximum camber, and the number of points in the point cloud."
    """

    # Get geometry data
    geometry_df = db.get_airfoil_geometry_dataframe()
    airfoil_df = db.get_airfoil_dataframe()

    # Create 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        "Geometric Characteristics of 1608 Airfoil Database\n"
        + "Supporting Document Figure 2 Requirements",
        fontsize=16,
        fontweight="bold",
    )

    # Plot 1: Maximum thickness distribution
    ax1 = axes[0, 0]
    if "max_thickness" in geometry_df.columns:
        thickness_data = geometry_df["max_thickness"].dropna()
        ax1.hist(thickness_data, bins=50, alpha=0.7, color="blue", edgecolor="black")
        ax1.set_xlabel("Maximum Thickness")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Distribution of Maximum Thickness")
        ax1.grid(True, alpha=0.3)

    # Plot 2: Thickness-to-chord ratio distribution
    ax2 = axes[0, 1]
    if "thickness_to_chord_ratio" in geometry_df.columns:
        tc_data = geometry_df["thickness_to_chord_ratio"].dropna()
        ax2.hist(tc_data, bins=50, alpha=0.7, color="green", edgecolor="black")
        ax2.set_xlabel("Thickness-to-Chord Ratio")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Distribution of T/C Ratio")
        ax2.grid(True, alpha=0.3)

    # Plot 3: Maximum camber distribution
    ax3 = axes[1, 0]
    if "max_camber" in geometry_df.columns:
        camber_data = geometry_df["max_camber"].dropna()
        ax3.hist(camber_data, bins=50, alpha=0.7, color="red", edgecolor="black")
        ax3.set_xlabel("Maximum Camber")
        ax3.set_ylabel("Frequency")
        ax3.set_title("Distribution of Maximum Camber")
        ax3.grid(True, alpha=0.3)

    # Plot 4: Number of points distribution (should all be 1000 after preprocessing)
    ax4 = axes[1, 1]
    if "Num_Points" in airfoil_df.columns:
        points_data = airfoil_df["Num_Points"]
        ax4.hist(points_data, bins=20, alpha=0.7, color="orange", edgecolor="black")
        ax4.set_xlabel("Number of Points in Point Cloud")
        ax4.set_ylabel("Frequency")
        ax4.set_title("Distribution of Point Cloud Density\n(Post-Standardization)")
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(
        output_dir / "figure2_geometric_distributions.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(output_dir / "figure2_geometric_distributions.pdf", bbox_inches="tight")
    plt.show()


def prepare_classical_modeling_features(db):
    """Prepare feature matrix for classical ML algorithms."""

    # Get geometric features
    geometry_df = db.get_airfoil_geometry_dataframe()

    # Select features for classical modeling
    feature_columns = [
        "max_thickness",
        "max_camber",
        "leading_edge_radius",
        "trailing_edge_angle",
        "thickness_to_chord_ratio",
    ]

    # Create feature matrix
    features_df = geometry_df[["name"] + feature_columns].copy()
    features_df = features_df.dropna()  # Remove rows with missing values

    logging.info(
        f"Prepared {len(features_df)} samples with {len(feature_columns)} features"
    )

    return features_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("WEEK 2: GEOMETRIC ANALYSIS AND CLASSICAL MODELING PREP")
    print("Airfoil Geometric Similarity Search Project")
    print("Supporting HW4/HW5 Document Requirements")
    print("=" * 60)

    features_df = run_comprehensive_geometric_analysis()

    print("\n" + "=" * 60)
    print("Geometric Analysis Complete!")
    print("Ready for Classical Modeling Implementation")
    print(f"Dataset: {len(features_df)} airfoils with standardized features")
    print("Next: Implement classical ML algorithms for Table 4 generation")
    print("=" * 60)
