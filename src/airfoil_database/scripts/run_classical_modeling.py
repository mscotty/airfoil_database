# scripts/run_classical_modeling.py
"""
Week 2 Classical Modeling Execution Script
Supports HW4/HW5 document requirements and DoD mission objectives.
"""

import logging
import sys
from pathlib import Path
from airfoil_database.core.database import AirfoilDatabase
from airfoil_database.analysis.geometry_analyzer import GeometryAnalyzer
from airfoil_database.scripts.classical_modeling import ClassicalModelingPipeline


def setup_logging():
    """Setup logging configuration for classical modeling phase."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("classical_modeling.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def validate_preprocessed_database():
    """
    Validate that preprocessed database exists and contains required data.
    Supports document requirement for standardized input data.
    """
    try:
        db = AirfoilDatabase(db_name="airfoils_preprocessed.db", db_dir=".")

        # Check if database has airfoils
        airfoil_df = db.get_airfoil_dataframe()
        if len(airfoil_df) == 0:
            raise ValueError("Preprocessed database is empty")

        # Check if geometric analysis has been completed
        geometry_df = db.get_airfoil_geometry_dataframe()
        if len(geometry_df) == 0:
            logging.warning(
                "No geometric analysis found. Running geometric analysis first..."
            )
            return db, False  # Needs geometric analysis

        logging.info(
            f"Preprocessed database validated: {len(airfoil_df)} airfoils, {len(geometry_df)} with geometric analysis"
        )
        return db, True

    except Exception as e:
        logging.error(f"Preprocessed database validation failed: {e}")
        logging.info(
            "Please run Week 1 data preparation first to create airfoils_preprocessed.db"
        )
        return None, False


def run_geometric_analysis_if_needed(db):
    """Run geometric analysis if not already completed."""

    logging.info(
        "Running geometric analysis to extract features for classical modeling..."
    )
    analyzer = GeometryAnalyzer(db)

    # Use parallel processing for efficiency
    processed_count = analyzer.compute_geometry_metrics_parallel(
        num_processes=4, batch_size=50
    )

    logging.info(f"Geometric analysis completed for {processed_count} airfoils")
    return True


def main():
    """
    Execute complete classical modeling pipeline supporting document requirements.
    Generates Table 4 and associated figures for HW4/HW5 submission.
    """

    setup_logging()

    print("=" * 80)
    print("HOMEWORK 4/5: CLASSICAL MODELING FOR AIRFOIL GEOMETRIC SIMILARITY SEARCH")
    print("Mitchell Scott - Air Force Institute of Technology")
    print("Supporting DoD Mission: Rapid Airfoil Identification and Quality Assurance")
    print("=" * 80)

    # Step 1: Validate preprocessed database
    logging.info("Step 1: Validating preprocessed database...")
    db, has_geometry = validate_preprocessed_database()

    if db is None:
        print("❌ FAILED: Preprocessed database not available")
        print("Please run Week 1 data preparation first:")
        print("  python scripts/run_data_preparation.py")
        return False

    # Step 2: Run geometric analysis if needed
    if not has_geometry:
        logging.info("Step 2: Running geometric analysis...")
        success = run_geometric_analysis_if_needed(db)
        if not success:
            print("❌ FAILED: Geometric analysis could not be completed")
            return False
    else:
        logging.info("Step 2: Geometric analysis already completed")

    # Step 3: Initialize classical modeling pipeline
    logging.info("Step 3: Initializing classical modeling pipeline...")
    pipeline = ClassicalModelingPipeline(db)

    # Step 4: Extract geometric features (Table 3 support)
    logging.info("Step 4: Extracting geometric features for Table 3...")
    try:
        X, y, airfoil_names = pipeline.extract_geometric_features()
        logging.info(
            f"Extracted features: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes"
        )
    except Exception as e:
        logging.error(f"Feature extraction failed: {e}")
        return False

    # Step 5: Setup train/validation/test splits (Section II.D requirement)
    logging.info(
        "Step 5: Setting up train/validation/test splits to prevent overfitting..."
    )
    pipeline.setup_train_validate_split(X, y)

    # Step 6: Initialize classical models (Section II.D requirement)
    logging.info("Step 6: Initializing classical ML algorithms...")
    model_justifications = pipeline.setup_classical_models()

    print(f"\nInitialized {len(pipeline.models)} classical algorithms:")
    for model_name in pipeline.models.keys():
        print(f"  • {model_name}")

    # Step 7: Analyze input variable contributions (Section II.D requirement)
    logging.info("Step 7: Analyzing input variable contributions...")
    contributions = pipeline.analyze_input_variable_contributions()

    # Step 8: Train and evaluate models (Table 4 generation)
    logging.info("Step 8: Training and evaluating classical models for Table 4...")
    results = pipeline.train_and_evaluate_models()

    # Step 9: Generate comprehensive report (Document requirements)
    logging.info("Step 9: Generating comprehensive report for document submission...")

    # Create results directory
    output_dir = Path("classical_modeling_results")
    output_dir.mkdir(exist_ok=True)

    # Generate all required outputs
    report = pipeline.generate_comprehensive_report(str(output_dir))

    # Step 10: Mission criteria assessment (Section I requirements)
    mission_assessment = report["mission_assessment"]

    print(f"\n" + "=" * 80)
    print("CLASSICAL MODELING RESULTS SUMMARY")
    print("=" * 80)

    print(f"\n📊 PERFORMANCE METRICS:")
    print(
        f"  • Best performing algorithm: {mission_assessment.get('best_model', 'Unknown')}"
    )
    print(
        f"  • Achieved accuracy: {mission_assessment.get('accuracy_achieved', 0):.1f}%"
    )
    print(
        f"  • Achieved inference time: {mission_assessment.get('inference_time_achieved', 0):.3f}ms"
    )

    print(f"\n🎯 DOD MISSION CRITERIA:")
    meets_accuracy = mission_assessment.get("meets_95_percent_accuracy", False)
    meets_speed = mission_assessment.get("meets_10ms_inference", False)
    mission_ready = mission_assessment.get("supports_dod_mission", False)

    print(
        f"  • 95% accuracy requirement: {'✅ MET' if meets_accuracy else '❌ NOT MET'}"
    )
    print(
        f"  • <10ms inference requirement: {'✅ MET' if meets_speed else '❌ NOT MET'}"
    )
    print(f"  • Ready for DoD deployment: {'✅ YES' if mission_ready else '❌ NO'}")

    print(f"\n📁 DOCUMENT DELIVERABLES GENERATED:")
    print(
        f"  • Table 4 (Classical modeling results): {output_dir}/table4_classical_results.csv"
    )
    print(
        f"  • Performance comparison figure: {output_dir}/classical_performance_comparison.png"
    )
    print(f"  • Overfitting analysis figure: {output_dir}/overfitting_analysis.png")
    print(f"  • Comprehensive summary: {output_dir}/classical_modeling_summary.json")

    print(f"\n✅ CLASSICAL MODELING PHASE COMPLETE")
    print(f"Ready for Neural Network Modeling Phase (HW5 Table 5 requirements)")

    if mission_ready:
        print(f"\n🎖️  MISSION ASSESSMENT: Classical models demonstrate feasibility")
        print(
            f"   for DoD rapid airfoil identification and quality assurance applications"
        )
    else:
        print(f"\n⚠️  MISSION ASSESSMENT: Neural networks will be required to meet")
        print(f"   DoD performance criteria for operational deployment")

    print("=" * 80)

    return True


if __name__ == "__main__":
    import numpy as np

    success = main()

    if success:
        print("\n🎉 Classical modeling execution completed successfully!")
        print(
            "All document requirements for HW4/HW5 classical modeling section generated."
        )
        sys.exit(0)
    else:
        print("\n💥 Classical modeling execution failed!")
        print("Please check logs and resolve issues before proceeding.")
        sys.exit(1)
