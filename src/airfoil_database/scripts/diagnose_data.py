# scripts/diagnose_data.py
"""
Diagnostic script to identify data quality issues in airfoil geometric features.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def diagnose_airfoil_data(db):
    """Comprehensive data quality diagnostics."""

    print("=" * 80)
    print("AIRFOIL DATA QUALITY DIAGNOSTICS")
    print("=" * 80)

    # Get raw geometric data (BEFORE scaling)
    geometry_df = db.get_airfoil_geometry_dataframe()

    feature_names = [
        "max_thickness",
        "max_camber",
        "leading_edge_radius",
        "trailing_edge_angle",
        "thickness_to_chord_ratio",
        "chord_length",
        "aspect_ratio",
    ]

    print(f"\n📊 Dataset Overview:")
    print(f"   Total airfoils: {len(geometry_df)}")
    print(f"   Unique airfoils: {geometry_df['name'].nunique()}")
    print(f"   Features: {len(feature_names)}")

    # Check 1: Missing values
    print(f"\n🔍 Check 1: Missing Values")
    missing = geometry_df[feature_names].isnull().sum()
    if missing.any():
        print(f"   ⚠️  WARNING: Missing values detected!")
        print(missing[missing > 0])
    else:
        print(f"   ✅ No missing values")

    # Check 2: Constant features (no variance)
    print(f"\n🔍 Check 2: Feature Variance")
    X_raw = geometry_df[feature_names].values
    valid_mask = ~np.any(np.isnan(X_raw), axis=1)
    X_clean = X_raw[valid_mask]

    for i, feature in enumerate(feature_names):
        values = X_clean[:, i]
        unique_vals = len(np.unique(values))
        variance = np.var(values)
        print(f"   {feature}:")
        print(f"      Unique values: {unique_vals}")
        print(f"      Range: [{values.min():.4f}, {values.max():.4f}]")
        print(f"      Variance: {variance:.6f}")

        if unique_vals == 1:
            print(f"      ❌ PROBLEM: Feature is constant (all same value)!")
        elif unique_vals < 10:
            print(f"      ⚠️  WARNING: Very few unique values")
        elif variance < 1e-10:
            print(f"      ❌ PROBLEM: Near-zero variance!")

    # Check 3: Feature correlations (raw data)
    print(f"\n🔍 Check 3: Feature Correlations (Raw Data)")
    corr_matrix = pd.DataFrame(X_clean, columns=feature_names).corr()

    perfect_corr = []
    high_corr = []

    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.99:
                perfect_corr.append((feature_names[i], feature_names[j], corr_val))
            elif abs(corr_val) > 0.8:
                high_corr.append((feature_names[i], feature_names[j], corr_val))

    if perfect_corr:
        print(f"   ❌ CRITICAL: Perfect/near-perfect correlations detected!")
        for f1, f2, corr in perfect_corr:
            print(f"      {f1} <-> {f2}: r={corr:.4f}")
            print(f"         → These features are redundant or mathematically derived")

    if high_corr:
        print(f"   ⚠️  High correlations (0.8-0.99):")
        for f1, f2, corr in high_corr:
            print(f"      {f1} <-> {f2}: r={corr:.4f}")

    if not perfect_corr and not high_corr:
        print(f"   ✅ No concerning correlations")

    # Check 4: Duplicate samples
    print(f"\n🔍 Check 4: Duplicate Feature Vectors")
    # Round to avoid floating point issues
    X_rounded = np.round(X_clean, decimals=6)
    unique_samples = np.unique(X_rounded, axis=0)
    n_duplicates = len(X_clean) - len(unique_samples)

    if n_duplicates > 0:
        print(f"   ⚠️  WARNING: {n_duplicates} duplicate feature vectors found")
        print(f"   This means {n_duplicates} airfoils are geometrically identical")
    else:
        print(f"   ✅ All feature vectors are unique")

    # Check 5: Sample some actual values
    print(f"\n🔍 Check 5: Sample of Raw Feature Values")
    sample_df = geometry_df[["name"] + feature_names].head(10)
    print(sample_df.to_string(index=False))

    # Check 6: Distance distribution
    print(f"\n🔍 Check 6: Pairwise Distance Distribution")
    from sklearn.metrics.pairwise import euclidean_distances
    from sklearn.preprocessing import StandardScaler

    # Sample 100 random airfoils for efficiency
    if len(X_clean) > 100:
        sample_indices = np.random.choice(len(X_clean), 100, replace=False)
        X_sample = X_clean[sample_indices]
    else:
        X_sample = X_clean

    # Raw distances
    distances_raw = euclidean_distances(X_sample)
    distances_raw = distances_raw[np.triu_indices_from(distances_raw, k=1)]

    # Scaled distances
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sample)
    distances_scaled = euclidean_distances(X_scaled)
    distances_scaled = distances_scaled[np.triu_indices_from(distances_scaled, k=1)]

    print(f"   Raw distances:")
    print(f"      Min: {distances_raw.min():.6f}")
    print(f"      Mean: {distances_raw.mean():.6f}")
    print(f"      Max: {distances_raw.max():.6f}")
    print(f"      Std: {distances_raw.std():.6f}")

    print(f"   Scaled distances:")
    print(f"      Min: {distances_scaled.min():.6f}")
    print(f"      Mean: {distances_scaled.mean():.6f}")
    print(f"      Max: {distances_scaled.max():.6f}")
    print(f"      Std: {distances_scaled.std():.6f}")

    if distances_raw.std() < 1e-6:
        print(f"      ❌ CRITICAL: All airfoils are nearly identical!")
    elif distances_scaled.std() < 0.5:
        print(f"      ⚠️  WARNING: Low diversity in feature space")

    # Check 7: Nearest neighbor test
    print(f"\n🔍 Check 7: Nearest Neighbor Sanity Check")
    from sklearn.neighbors import NearestNeighbors

    # Pick first airfoil as query
    query_name = geometry_df.iloc[0]["name"]
    query_features = X_clean[0:1]

    # Find 5 nearest neighbors in raw space
    nn = NearestNeighbors(
        n_neighbors=6, metric="euclidean"
    )  # 6 because first is itself
    nn.fit(X_clean)
    distances, indices = nn.kneighbors(query_features)

    print(f"   Query airfoil: {query_name}")
    print(f"   5 Nearest neighbors (raw features):")
    for i, (idx, dist) in enumerate(zip(indices[0][1:], distances[0][1:])):
        neighbor_name = geometry_df.iloc[idx]["name"]
        print(f"      {i+1}. {neighbor_name} (distance: {dist:.6f})")

    if distances[0][1] < 1e-6:
        print(f"      ❌ PROBLEM: Nearest neighbor is identical (distance ≈ 0)!")

    # Create visualizations
    print(f"\n📊 Generating diagnostic visualizations...")
    create_diagnostic_plots(X_clean, feature_names, corr_matrix)

    # Final assessment
    print(f"\n" + "=" * 80)
    print(f"DIAGNOSTIC SUMMARY")
    print(f"=" * 80)

    issues = []
    if perfect_corr:
        issues.append("❌ Perfect feature correlations detected")
    if distances_raw.std() < 1e-6:
        issues.append("❌ All airfoils appear nearly identical")
    if n_duplicates > len(X_clean) * 0.1:
        issues.append(f"⚠️  High duplicate rate ({n_duplicates}/{len(X_clean)})")

    if issues:
        print(f"🚨 CRITICAL ISSUES DETECTED:")
        for issue in issues:
            print(f"   {issue}")
        print(f"\nRECOMMENDATIONS:")
        print(f"   1. Check geometric feature calculation code")
        print(f"   2. Verify features are measuring different properties")
        print(f"   3. Remove redundant features (perfect correlations)")
        print(f"   4. Consider using raw airfoil coordinates instead")
    else:
        print(f"✅ Data quality appears acceptable")
        print(f"   Issue may be in modeling approach")

    print(f"=" * 80)


def create_diagnostic_plots(X, feature_names, corr_matrix):
    """Create diagnostic visualizations."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Feature distributions
    ax1 = axes[0, 0]
    df = pd.DataFrame(X, columns=feature_names)
    df.boxplot(ax=ax1, rot=45)
    ax1.set_title("Feature Distributions (Raw Values)")
    ax1.set_ylabel("Value")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Correlation heatmap
    ax2 = axes[0, 1]
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        center=0,
        ax=ax2,
        square=True,
        cbar_kws={"label": "Correlation"},
    )
    ax2.set_title("Feature Correlation Matrix")

    # Plot 3: Pairwise scatter (first 3 features)
    ax3 = axes[1, 0]
    if len(feature_names) >= 2:
        ax3.scatter(X[:, 0], X[:, 1], alpha=0.5, s=20)
        ax3.set_xlabel(feature_names[0])
        ax3.set_ylabel(feature_names[1])
        ax3.set_title(f"{feature_names[0]} vs {feature_names[1]}")
        ax3.grid(True, alpha=0.3)

    # Plot 4: Distance histogram
    ax4 = axes[1, 1]
    from sklearn.metrics.pairwise import euclidean_distances

    # Sample for efficiency
    if len(X) > 200:
        sample_idx = np.random.choice(len(X), 200, replace=False)
        X_sample = X[sample_idx]
    else:
        X_sample = X

    distances = euclidean_distances(X_sample)
    distances = distances[np.triu_indices_from(distances, k=1)]

    ax4.hist(distances, bins=50, alpha=0.7, edgecolor="black")
    ax4.set_xlabel("Euclidean Distance")
    ax4.set_ylabel("Frequency")
    ax4.set_title("Pairwise Distance Distribution")
    ax4.axvline(distances.mean(), color="red", linestyle="--", label="Mean")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("diagnostic_plots.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"   Saved diagnostic plots to: diagnostic_plots.png")


def main():
    """Run diagnostics."""
    from airfoil_database.core.database import AirfoilDatabase

    try:
        db = AirfoilDatabase(db_name="airfoils_preprocessed.db", db_dir=".")
        diagnose_airfoil_data(db)
    except Exception as e:
        logging.error(f"Failed to load database: {e}")
        return


if __name__ == "__main__":
    main()
