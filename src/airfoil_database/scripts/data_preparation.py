# scripts/run_data_preparation.py
import logging
from pathlib import Path
import sys
import os

import numpy as np

from airfoil_database.core.database import AirfoilDatabase
from airfoil_database.core.data_preparation import AirfoilDataPreparator
from airfoil_database.utilities.get_top_level_module import get_project_root
from airfoil_database.scripts.visualize_data_preparation import (
    create_targeted_data_preparation_figures,
    create_point_distribution_analysis,
    create_data_distribution_analysis,
)


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("data_preparation.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def main():
    """Main execution function for Week 1 data preparation."""
    setup_logging()
    logging.info("Starting Week 1 Data Preparation Phase")

    # Configuration
    DB_PATH = os.path.join(get_project_root(), "airfoil_database", "airfoils.db")
    OUTPUT_DIR = "processed_data"
    FIGURES_DIR = "figures"
    TARGET_POINTS = 1000

    try:
        # Initialize database connection
        db_dir = str(Path(DB_PATH).parent)
        db_name = Path(DB_PATH).name
        db = AirfoilDatabase(db_name=db_name, db_dir=db_dir)

        # Initialize data preparator
        preparator = AirfoilDataPreparator(db, target_points=TARGET_POINTS)

        # Step 1: Analyze current data distribution
        logging.info("Step 1: Analyzing current data distribution...")
        create_data_distribution_analysis(DB_PATH, FIGURES_DIR)

        # Step 2: Enhanced processing with robust reordering (UPDATED)
        logging.info("Step 2: Processing airfoils with enhanced robust reordering...")
        processed_data, diagnostics_data = (
            preparator.process_all_airfoils_with_diagnostics()
        )

        # Step 2.5: Validate success rate meets requirements (NEW)
        success_rate = (len(processed_data) / len(diagnostics_data)) * 100
        if success_rate < 95.0:
            logging.error(
                f"Success rate {success_rate:.1f}% below 95% requirement for deep learning"
            )

        # Step 3: Generate validation report
        logging.info("Step 3: Generating validation report...")
        generate_validation_report(diagnostics_data, OUTPUT_DIR)

        # Step 4: Create debug plots for failed airfoils
        failed_airfoils = [
            name
            for name, diag in diagnostics_data.items()
            if not diag.get("processing_successful", True)
        ]

        if failed_airfoils:
            logging.info(
                f"Step 4: Creating debug plots for {len(failed_airfoils)} failed airfoils..."
            )
            preparator.create_failed_airfoil_debug_plots(failed_airfoils)

        # Step 5: Save processed data (only successful ones)
        logging.info("Step 5: Saving successfully processed data...")
        Path(OUTPUT_DIR).mkdir(exist_ok=True)

        preparator.save_processed_data(
            processed_data, f"{OUTPUT_DIR}/airfoil_1000pts_processed.json"
        )
        np.save(f"{OUTPUT_DIR}/airfoil_names.npy", list(processed_data.keys()))
        point_arrays = np.array(
            [processed_data[name] for name in processed_data.keys()]
        )
        np.save(f"{OUTPUT_DIR}/airfoil_pointclouds_1000pts.npy", point_arrays)

        # Step 6: Create targeted visualizations
        logging.info("Step 6: Creating targeted visualizations...")
        create_targeted_data_preparation_figures(DB_PATH, FIGURES_DIR)

        # Step 7: Generate summary statistics
        logging.info("Step 7: Generating summary statistics...")
        generate_preparation_summary_with_validation(
            processed_data, diagnostics_data, OUTPUT_DIR
        )

        logging.info("Week 1 Data Preparation with validation completed successfully!")

    except Exception as e:
        logging.error(f"Error in data preparation: {e}", exc_info=True)
        raise


def generate_validation_report(diagnostics_data: dict, output_dir: str):
    """Generate comprehensive validation report."""

    successful_count = sum(
        1
        for diag in diagnostics_data.values()
        if diag.get("processing_successful", True)
    )
    total_count = len(diagnostics_data)
    failure_rate = ((total_count - successful_count) / total_count) * 100

    # Analyze failure patterns
    failure_types = {}
    for name, diag in diagnostics_data.items():
        if not diag.get("processing_successful", True):
            final_validation = diag.get("final_validation", {})
            issues = final_validation.get("issues", [])
            for issue in issues:
                issue_type = issue.split(":")[0]
                failure_types[issue_type] = failure_types.get(issue_type, 0) + 1

    # Create validation report
    validation_report = {
        "summary": {
            "total_airfoils": total_count,
            "successful_processing": successful_count,
            "failed_processing": total_count - successful_count,
            "success_rate_percent": ((successful_count / total_count) * 100),
            "failure_rate_percent": failure_rate,
        },
        "failure_analysis": {
            "common_failure_types": sorted(
                failure_types.items(), key=lambda x: x[1], reverse=True
            ),
            "failure_threshold": failure_rate,
        },
        "quality_metrics": {
            "meets_95_percent_target": successful_count >= (0.95 * total_count),
            "ready_for_deep_learning": failure_rate
            < 5.0,  # Less than 5% failure acceptable
        },
    }

    # Save report
    import json

    with open(f"{output_dir}/validation_report.json", "w") as f:
        json.dump(validation_report, f, indent=2)

    # Print summary
    print(f"\n=== Data Preparation Validation Report ===")
    print(f"Total airfoils: {total_count}")
    print(
        f"Successfully processed: {successful_count} ({successful_count/total_count*100:.1f}%)"
    )
    print(f"Failed processing: {total_count - successful_count} ({failure_rate:.1f}%)")

    if validation_report["quality_metrics"]["meets_95_percent_target"]:
        print("✅ SUCCESS: Meets 95% processing target for deep metric learning")
    else:
        print("❌ WARNING: Processing success rate below 95% target")

    if failure_types:
        print(f"\nMost common failure types:")
        for failure_type, count in validation_report["failure_analysis"][
            "common_failure_types"
        ][:3]:
            print(f"  • {failure_type}: {count} airfoils")


# scripts/run_data_preparation.py


def generate_preparation_summary_with_validation(
    processed_data: dict, diagnostics_data: dict, output_dir: str
):
    """
    Generate comprehensive summary statistics including validation results.
    This supports the HW5 Data Preparation section documentation requirements.
    """

    # Calculate basic statistics
    total_airfoils = len(diagnostics_data)
    successful_airfoils = len(processed_data)
    failed_airfoils = total_airfoils - successful_airfoils

    # Verify all have target number of points
    point_counts = [len(points) for points in processed_data.values()]
    target_achieved = all(count == 1000 for count in point_counts)

    # Calculate geometric ranges for successful airfoils
    all_x_coords = []
    all_y_coords = []
    for points in processed_data.values():
        all_x_coords.extend(points[:, 0])
        all_y_coords.extend(points[:, 1])

    # Analyze validation results
    validation_summary = analyze_validation_results(diagnostics_data)

    # Compile comprehensive summary
    summary_stats = {
        "processing_overview": {
            "total_airfoils_in_database": total_airfoils,
            "successfully_processed": successful_airfoils,
            "failed_processing": failed_airfoils,
            "success_rate_percent": (successful_airfoils / total_airfoils) * 100,
            "target_points_per_airfoil": 1000,
            "standardization_achieved": target_achieved,
        },
        "coordinate_statistics": {
            "x_coordinate_range": {
                "min": float(np.min(all_x_coords)) if all_x_coords else 0,
                "max": float(np.max(all_x_coords)) if all_x_coords else 0,
                "mean": float(np.mean(all_x_coords)) if all_x_coords else 0,
            },
            "y_coordinate_range": {
                "min": float(np.min(all_y_coords)) if all_y_coords else 0,
                "max": float(np.max(all_y_coords)) if all_y_coords else 0,
                "mean": float(np.mean(all_y_coords)) if all_y_coords else 0,
            },
        },
        "validation_analysis": validation_summary,
        "deep_learning_readiness": {
            "meets_95_percent_success": successful_airfoils >= (0.95 * total_airfoils),
            "geometric_consistency_verified": validation_summary[
                "ordering_success_rate"
            ]
            > 90,
            "ready_for_1608_class_classification": target_achieved
            and (successful_airfoils >= 1500),
            "supports_dod_mission_requirements": validation_summary["critical_failures"]
            == 0,
        },
        "sample_airfoil_names": list(processed_data.keys())[:10],
    }

    # Save comprehensive summary
    import json

    with open(f"{output_dir}/preparation_summary_with_validation.json", "w") as f:
        json.dump(summary_stats, f, indent=2)

    # Print detailed summary
    print_comprehensive_summary(summary_stats)

    return summary_stats


def analyze_validation_results(diagnostics_data: dict) -> dict:
    """Analyze validation diagnostics to extract key metrics."""

    total_count = len(diagnostics_data)
    ordering_successes = 0
    critical_failures = 0
    failure_types = {}

    for airfoil_name, diagnostics in diagnostics_data.items():
        # Check if processing was successful
        if diagnostics.get("processing_successful", False):
            # Check validation details
            final_validation = diagnostics.get("final_validation", {})
            if final_validation.get("success", False):
                ordering_successes += 1

            # Categorize issues
            issues = final_validation.get("issues", [])
            for issue in issues:
                issue_category = issue.split(":")[0].strip()
                failure_types[issue_category] = failure_types.get(issue_category, 0) + 1

                # Identify critical failures that would impact deep learning
                if any(
                    critical_term in issue.lower()
                    for critical_term in ["closure", "duplicate", "unknown"]
                ):
                    critical_failures += 1

    return {
        "total_airfoils": total_count,
        "ordering_successes": ordering_successes,
        "ordering_success_rate": (
            (ordering_successes / total_count) * 100 if total_count > 0 else 0
        ),
        "critical_failures": critical_failures,
        "failure_type_breakdown": failure_types,
        "most_common_failure": (
            max(failure_types.items(), key=lambda x: x[1]) if failure_types else None
        ),
    }


def print_comprehensive_summary(summary_stats: dict):
    """Print formatted summary supporting HW5 documentation requirements."""

    print(f"\n" + "=" * 70)
    print(f"COMPREHENSIVE DATA PREPARATION SUMMARY")
    print(f"Airfoil Geometric Similarity Search - Week 1 Results")
    print(f"=" * 70)

    # Processing Overview
    overview = summary_stats["processing_overview"]
    print(f"\n📊 PROCESSING OVERVIEW:")
    print(f"  • Total airfoils in database: {overview['total_airfoils_in_database']:,}")
    print(f"  • Successfully processed: {overview['successfully_processed']:,}")
    print(f"  • Failed processing: {overview['failed_processing']:,}")
    print(f"  • Success rate: {overview['success_rate_percent']:.1f}%")
    print(
        f"  • Standardization to 1000 points: {'✅ ACHIEVED' if overview['standardization_achieved'] else '❌ FAILED'}"
    )

    # Deep Learning Readiness Assessment
    readiness = summary_stats["deep_learning_readiness"]
    print(f"\n🎯 DEEP LEARNING READINESS ASSESSMENT:")
    print(
        f"  • Meets 95% success target: {'✅ YES' if readiness['meets_95_percent_success'] else '❌ NO'}"
    )
    print(
        f"  • Geometric consistency verified: {'✅ YES' if readiness['geometric_consistency_verified'] else '❌ NO'}"
    )
    print(
        f"  • Ready for 1608-class classification: {'✅ YES' if readiness['ready_for_1608_class_classification'] else '❌ NO'}"
    )
    print(
        f"  • Supports DoD mission requirements: {'✅ YES' if readiness['supports_dod_mission_requirements'] else '❌ NO'}"
    )

    # Validation Analysis
    validation = summary_stats["validation_analysis"]
    print(f"\n🔍 VALIDATION ANALYSIS:")
    print(
        f"  • Point ordering success rate: {validation['ordering_success_rate']:.1f}%"
    )
    print(f"  • Critical failures detected: {validation['critical_failures']}")

    if validation["most_common_failure"]:
        failure_type, count = validation["most_common_failure"]
        print(f"  • Most common failure type: {failure_type} ({count} airfoils)")

    # Coordinate Statistics
    coords = summary_stats["coordinate_statistics"]
    print(f"\n📐 COORDINATE STATISTICS:")
    print(
        f"  • X-coordinate range: [{coords['x_coordinate_range']['min']:.3f}, {coords['x_coordinate_range']['max']:.3f}]"
    )
    print(
        f"  • Y-coordinate range: [{coords['y_coordinate_range']['min']:.3f}, {coords['y_coordinate_range']['max']:.3f}]"
    )

    # Mission Alignment
    print(f"\n🎖️ MISSION ALIGNMENT:")
    success_rate = overview["success_rate_percent"]
    if success_rate >= 95:
        print(
            f"  ✅ MISSION READY: {success_rate:.1f}% success rate supports DoD rapid identification goals"
        )
        print(
            f"  ✅ Quality assurance acceleration achieved for captured vehicle components"
        )
    elif success_rate >= 90:
        print(
            f"  ⚠️ NEARLY READY: {success_rate:.1f}% success rate - minor issues to resolve"
        )
    else:
        print(
            f"  ❌ REQUIRES ATTENTION: {success_rate:.1f}% success rate below mission requirements"
        )

    print(f"\n" + "=" * 70)
    print(
        f"Data preparation results saved to: preparation_summary_with_validation.json"
    )
    print(f"Ready for Week 2: Classical Modeling Implementation")
    print(f"=" * 70)


if __name__ == "__main__":
    main()
