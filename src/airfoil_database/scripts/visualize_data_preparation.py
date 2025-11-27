# scripts/visualize_data_preparation.py
import matplotlib.pyplot as plt
import numpy as np
import json
import random
from pathlib import Path
from typing import List, Tuple
import logging
from sqlmodel import Session, select
from airfoil_database.core.database import AirfoilDatabase
from airfoil_database.core.data_preparation import AirfoilDataPreparator
from airfoil_database.core.models import Airfoil


def create_point_density_visualization(db_path: str, output_dir: str = "figures"):
    """
    Create Plot 1: Point density comparison showing before/after resampling
    on the same plot for each individual airfoil. This addresses the "Variable
    Point Density" issue identified in the document.
    """
    # Initialize database and preparator
    db_dir = str(Path(db_path).parent)
    db_name = Path(db_path).name
    db = AirfoilDatabase(db_name=db_name, db_dir=db_dir)
    preparator = AirfoilDataPreparator(db, target_points=1000)

    # Get sample of diverse airfoils for visualization
    with Session(db.engine) as session:
        statement = select(Airfoil.name, Airfoil.pointcloud)
        all_airfoils = session.exec(statement).all()

    # Select airfoils with different point densities
    sample_airfoils = []
    point_counts = []

    for airfoil in all_airfoils:
        if airfoil.pointcloud:
            points = preparator.parse_pointcloud_from_string(airfoil.pointcloud)
            if len(points) > 0:
                point_counts.append((airfoil.name, len(points), points))

    # Sort by point count and select diverse examples (low, medium, high density)
    point_counts.sort(key=lambda x: x[1])
    indices = [
        0,
        len(point_counts) // 4,
        len(point_counts) // 2,
        3 * len(point_counts) // 4,
        len(point_counts) - 1,
    ]
    sample_airfoils = [point_counts[i] for i in indices if i < len(point_counts)][:4]

    # Create 2x2 subplot for individual airfoil comparisons
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Point Cloud Density Standardization: Before vs After Resampling\n"
        + "Supporting 1608-Class Deep Metric Learning Classification",
        fontsize=16,
        fontweight="bold",
    )

    axes_flat = axes.flatten()

    for idx, (name, original_count, original_points) in enumerate(sample_airfoils):
        ax = axes_flat[idx]

        # Get resampled points
        resampled_points = preparator.process_single_airfoil(name)

        # Plot original with larger dots for fewer points, smaller for more points
        dot_size_original = max(100 - original_count / 10, 5)
        ax.scatter(
            original_points[:, 0],
            original_points[:, 1],
            s=dot_size_original,
            alpha=0.6,
            color="blue",
            label=f"Original ({original_count} pts)",
            edgecolors="darkblue",
            linewidth=0.5,
        )

        # Plot resampled with uniform small dots
        if resampled_points is not None:
            ax.scatter(
                resampled_points[:, 0],
                resampled_points[:, 1],
                s=8,
                alpha=0.8,
                color="red",
                label="Resampled (1000 pts)",
                marker="s",
                edgecolors="darkred",
                linewidth=0.3,
            )

        ax.set_title(
            f"{name}\nBefore: {original_count} points → After: 1000 points",
            fontweight="bold",
            fontsize=12,
        )
        ax.set_xlabel("x/c")
        ax.set_ylabel("y/c")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.axis("equal")
        ax.set_xlim(-0.1, 1.1)

    plt.tight_layout()

    # Save figure
    Path(output_dir).mkdir(exist_ok=True)
    plt.savefig(
        f"{output_dir}/point_density_individual_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        f"{output_dir}/point_density_individual_comparison.pdf", bbox_inches="tight"
    )

    logging.info(f"Individual point density comparison saved to {output_dir}/")
    plt.show()
    plt.close()


def add_direction_arrows(ax, points: np.ndarray, color: str, arrow_spacing: int = 20):
    """Add arrows to show point traversal direction on airfoil."""
    if len(points) < arrow_spacing * 2:
        return

    for i in range(0, len(points) - arrow_spacing, arrow_spacing):
        start_point = points[i]
        end_point = points[i + arrow_spacing]

        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]

        # Make arrows more visible
        ax.arrow(
            start_point[0],
            start_point[1],
            dx,
            dy,
            head_width=0.025,
            head_length=0.015,
            fc=color,
            ec=color,
            alpha=0.9,
            length_includes_head=True,
            linewidth=1.5,
        )


def create_point_ordering_visualization(db_path: str, output_dir: str = "figures"):
    """
    Create Plot 2: Point ordering comparison with left subplot showing original
    ordering and right subplot showing standardized ordering. This addresses the
    "Inconsistent Point Ordering (clockwise vs. counterclockwise)" issue.
    """
    # Initialize database and preparator
    db_dir = str(Path(db_path).parent)
    db_name = Path(db_path).name
    db = AirfoilDatabase(db_name=db_name, db_dir=db_dir)
    preparator = AirfoilDataPreparator(db, target_points=1000)

    # Find airfoils with different ordering patterns
    sample_airfoils = []

    with Session(db.engine) as session:
        statement = select(Airfoil.name, Airfoil.pointcloud)
        airfoils = list(session.exec(statement).all())

        # Find examples with different characteristics
        for airfoil in airfoils[:100]:  # Check first 100
            if airfoil.pointcloud:
                points = preparator.parse_pointcloud_from_string(airfoil.pointcloud)
                if len(points) > 20:  # Need enough points to show ordering
                    normalized_points = preparator.normalize_airfoil_coordinates(points)
                    sample_airfoils.append((airfoil.name, normalized_points))
                    if len(sample_airfoils) >= 3:  # Get 3 examples
                        break

    # Create side-by-side comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(
        "Point Ordering Standardization: Original vs Consistent Counterclockwise\n"
        + "Ensuring Geometric Consistency for Deep Metric Learning",
        fontsize=16,
        fontweight="bold",
    )

    # Left plot: Original ordering
    for idx, (name, original_points) in enumerate(sample_airfoils):
        color = f"C{idx}"

        # Plot original airfoil with line connecting points to show order
        ax1.plot(
            original_points[:, 0],
            original_points[:, 1],
            "o-",
            color=color,
            linewidth=2,
            markersize=4,
            alpha=0.8,
            label=f"{name}",
        )

        # Add direction arrows to show traversal
        add_direction_arrows(
            ax1, original_points, color, arrow_spacing=max(len(original_points) // 8, 3)
        )

        # Mark start point with square
        ax1.plot(
            original_points[0, 0],
            original_points[0, 1],
            "s",
            color=color,
            markersize=10,
            markeredgecolor="black",
            markeredgewidth=2,
            label=f"{name} start",
        )

    ax1.set_title(
        "Before: Original Point Ordering\n(Inconsistent traversal patterns)",
        fontweight="bold",
        fontsize=14,
    )
    ax1.set_xlabel("x/c")
    ax1.set_ylabel("y/c")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.axis("equal")

    # Right plot: Standardized ordering
    for idx, (name, _) in enumerate(sample_airfoils):
        color = f"C{idx}"

        # Get standardized points
        standardized_points = preparator.process_single_airfoil(name)
        if standardized_points is not None:
            # Subsample for clearer visualization of ordering
            subsample_indices = np.linspace(
                0, len(standardized_points) - 1, 150, dtype=int
            )
            display_points = standardized_points[subsample_indices]

            ax2.plot(
                display_points[:, 0],
                display_points[:, 1],
                "o-",
                color=color,
                linewidth=2,
                markersize=4,
                alpha=0.8,
                label=f"{name}",
            )

            # Add direction arrows showing consistent counterclockwise ordering
            add_direction_arrows(ax2, display_points, color, arrow_spacing=8)

            # Mark start point (trailing edge) with square
            ax2.plot(
                display_points[0, 0],
                display_points[0, 1],
                "s",
                color=color,
                markersize=10,
                markeredgecolor="black",
                markeredgewidth=2,
                label=f"{name} start (TE)",
            )

    ax2.set_title(
        "After: Consistent Counterclockwise Ordering\n(All start from trailing edge)",
        fontweight="bold",
        fontsize=14,
    )
    ax2.set_xlabel("x/c")
    ax2.set_ylabel("y/c")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.axis("equal")

    plt.tight_layout()

    # Save figure
    Path(output_dir).mkdir(exist_ok=True)
    plt.savefig(
        f"{output_dir}/point_ordering_before_after_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        f"{output_dir}/point_ordering_before_after_comparison.pdf", bbox_inches="tight"
    )

    logging.info(f"Point ordering before/after comparison saved to {output_dir}/")
    plt.show()
    plt.close()


def create_data_distribution_analysis(db_path: str, output_dir: str = "figures"):
    """
    Create analysis plots showing point count distribution and ordering statistics
    from the original dataset, supporting the document's data understanding section.
    """
    db_dir = str(Path(db_path).parent)
    db_name = Path(db_path).name
    db = AirfoilDatabase(db_name=db_name, db_dir=db_dir)
    preparator = AirfoilDataPreparator(db)

    # Collect statistics
    point_counts = []
    ordering_stats = {"clockwise": 0, "counterclockwise": 0, "unknown": 0}

    with Session(db.engine) as session:
        statement = select(Airfoil)
        airfoils = session.exec(statement).all()

        for airfoil in airfoils:
            if airfoil.pointcloud:
                points = preparator.parse_pointcloud_from_string(airfoil.pointcloud)
                if len(points) > 0:
                    point_counts.append(len(points))
                    normalized_points = preparator.normalize_airfoil_coordinates(points)
                    ordering = preparator.detect_point_ordering(normalized_points)
                    ordering_stats[ordering] += 1

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Point count distribution
    ax1.hist(point_counts, bins=50, alpha=0.7, color="blue", edgecolor="black")
    ax1.axvline(
        1000, color="red", linestyle="--", linewidth=2, label="Target: 1000 points"
    )
    ax1.axvline(
        np.mean(point_counts),
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(point_counts):.0f} points",
    )
    ax1.set_xlabel("Number of Points")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Distribution of Point Counts\n(Before Standardization)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Ordering statistics
    labels = list(ordering_stats.keys())
    values = list(ordering_stats.values())
    colors = ["red", "green", "orange"]

    ax2.pie(values, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90)
    ax2.set_title("Point Ordering Distribution\n(Clockwise vs Counterclockwise)")

    plt.tight_layout()

    # Save figure
    Path(output_dir).mkdir(exist_ok=True)
    plt.savefig(
        f"{output_dir}/data_distribution_analysis.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(f"{output_dir}/data_distribution_analysis.pdf", bbox_inches="tight")
    plt.show()
    plt.close()

    # Print summary statistics
    print(f"\n=== Point Distribution Summary ===")
    print(f"Total airfoils processed: {len(point_counts)}")
    print(f"Average points per airfoil: {np.mean(point_counts):.1f}")
    print(f"Median points per airfoil: {np.median(point_counts):.1f}")
    print(f"Min points: {min(point_counts)}")
    print(f"Max points: {max(point_counts)}")
    print(f"Standard deviation: {np.std(point_counts):.1f}")
    print(f"\nOrdering Statistics:")
    for ordering, count in ordering_stats.items():
        percentage = (count / sum(ordering_stats.values())) * 100
        print(f"  {ordering}: {count} airfoils ({percentage:.1f}%)")


def create_targeted_data_preparation_figures(db_path: str, output_dir: str = "figures"):
    """
    Main function to create both targeted visualizations for HW5 documentation.
    This addresses the specific data preparation challenges identified in the document:
    - Variable Point Density
    - Inconsistent Point Ordering (clockwise vs. counterclockwise)

    These visualizations directly support the Method > Data Preparation section
    requirement for "up to 2 before/after figures showing the results of your
    data preparation" (Source: DASC 522 Homework4 Scott.docx)
    """
    logging.info("Creating targeted data preparation figures...")

    # Create both targeted visualizations
    create_point_density_visualization(db_path, output_dir)
    create_point_ordering_visualization(db_path, output_dir)

    # Also create the distribution analysis for additional context
    create_data_distribution_analysis(db_path, output_dir)

    print(f"\n=== Targeted Data Preparation Figures Created ===")
    print(f"Figures saved to: {output_dir}/")
    print(f"  - point_density_comparison.png/pdf")
    print(f"  - point_ordering_comparison.png/pdf")
    print(f"  - data_distribution_analysis.png/pdf")
    print(
        f"\nThese figures address the specific challenges identified in your project:"
    )
    print(f"  1. Variable Point Density → Standardized 1000-point resampling")
    print(f"  2. Inconsistent Point Ordering → Consistent counterclockwise traversal")

    logging.info("All targeted data preparation figures completed successfully")


def create_before_after_summary_figure(db_path: str, output_dir: str = "figures"):
    """
    Create a comprehensive summary figure showing the complete data preparation pipeline.
    This provides a single overview figure that could be used in the document's
    Data Preparation section.
    """
    db_dir = str(Path(db_path).parent)
    db_name = Path(db_path).name
    db = AirfoilDatabase(db_name=db_name, db_dir=db_dir)
    preparator = AirfoilDataPreparator(db, target_points=1000)

    # Get sample airfoils
    with Session(db.engine) as session:
        statement = select(Airfoil.name, Airfoil.pointcloud)
        all_airfoils = list(session.exec(statement).all())

    # Select 3 representative examples
    sample_names = []
    for airfoil in all_airfoils[:50]:  # Check first 50
        if airfoil.pointcloud:
            points = preparator.parse_pointcloud_from_string(airfoil.pointcloud)
            if len(points) > 0:
                sample_names.append(airfoil.name)
                if len(sample_names) >= 3:
                    break

    # Create comprehensive figure
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(
        "Airfoil Data Preparation Pipeline for Deep Metric Learning\n"
        + "Supporting DoD Mission: Rapid Airfoil Identification",
        fontsize=16,
        fontweight="bold",
    )

    for idx, airfoil_name in enumerate(sample_names):
        # Get original and processed data
        airfoil_data = db.get_airfoil_data(airfoil_name)
        if not airfoil_data:
            continue

        original_points = preparator.parse_pointcloud_from_string(airfoil_data[1])
        processed_points = preparator.process_single_airfoil(airfoil_name)

        # Before plot (left column)
        ax_before = axes[idx, 0]
        if original_points is not None and len(original_points) > 0:
            # Size dots based on point density for emphasis
            dot_size = max(100 - len(original_points) / 10, 5)
            ax_before.scatter(
                original_points[:, 0],
                original_points[:, 1],
                s=dot_size,
                alpha=0.7,
                c=f"C{idx}",
            )

            # Add direction indicators if enough points
            if len(original_points) > 20:
                add_direction_arrows(
                    ax_before,
                    original_points,
                    f"C{idx}",
                    arrow_spacing=max(len(original_points) // 10, 5),
                )

            ax_before.set_title(
                f"Before: {airfoil_name}\n({len(original_points)} pts, variable density)",
                fontweight="bold",
            )

        # After plot (right column)
        ax_after = axes[idx, 1]
        if processed_points is not None:
            ax_after.scatter(
                processed_points[:, 0],
                processed_points[:, 1],
                s=15,
                alpha=0.8,
                c=f"C{idx}",
            )

            # Add standardized direction indicators
            subsample = processed_points[::50]  # Every 50th point for clarity
            add_direction_arrows(ax_after, subsample, f"C{idx}", arrow_spacing=2)

            ax_after.set_title(
                f"After: {airfoil_name}\n(1000 pts, standardized)", fontweight="bold"
            )

        # Format both subplots
        for ax in [ax_before, ax_after]:
            ax.set_xlabel("x/c")
            ax.set_ylabel("y/c")
            ax.grid(True, alpha=0.3)
            ax.axis("equal")
            ax.set_xlim(-0.1, 1.1)

    plt.tight_layout()

    # Save comprehensive figure
    Path(output_dir).mkdir(exist_ok=True)
    plt.savefig(
        f"{output_dir}/data_preparation_pipeline_summary.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        f"{output_dir}/data_preparation_pipeline_summary.pdf", bbox_inches="tight"
    )
    plt.show()
    plt.close()

    logging.info(f"Data preparation pipeline summary saved to {output_dir}/")


# Legacy function for backward compatibility
def create_before_after_visualization(db_path: str, output_dir: str = "figures"):
    """
    Legacy function - calls the new targeted visualization approach.
    This maintains compatibility with existing scripts while providing
    the improved targeted visualizations required for HW5.
    """
    logging.info("Calling improved targeted data preparation visualizations...")
    create_targeted_data_preparation_figures(db_path, output_dir)


def create_point_distribution_analysis(db_path: str, output_dir: str = "figures"):
    """
    Maintained for backward compatibility - calls the improved distribution analysis.
    """
    create_data_distribution_analysis(db_path, output_dir)


# Main execution for testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    else:
        db_path = "/airfoil_database/airfoils.db"  # Default path from your project

    print("Creating targeted data preparation visualizations...")
    print("These support HW5 requirements for Data Preparation section figures.")

    create_targeted_data_preparation_figures(db_path)
    create_before_after_summary_figure(db_path)

    print("\nVisualization creation complete!")
    print("Figures address the specific data challenges identified in the document:")
    print("  1. Variable Point Density → Standardized to 1000 points")
    print("  2. Inconsistent Point Ordering → Consistent counterclockwise traversal")
    print("  3. Support for 1608-class deep metric learning classification")
