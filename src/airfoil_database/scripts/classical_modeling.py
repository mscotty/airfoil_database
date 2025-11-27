# scripts/classical_modeling.py
"""
Enhanced similarity search using distribution-based features.
Leverages thickness_distribution and camber_distribution from database.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.decomposition import PCA
import time
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.interpolate import interp1d


class DistributionBasedSimilaritySearch:
    """
    Similarity search using full thickness and camber distributions.
    This provides MUCH better discrimination than scalar features alone.
    """

    def __init__(self, preprocessed_db):
        self.db = preprocessed_db
        self.models = {}
        self.results = {}
        self.feature_names = []
        self.scaler = StandardScaler()
        self.X_scaled = None
        self.y = None
        self.airfoil_names = None

    def extract_distribution_features(self, n_samples=20):
        """
        Extract features from thickness and camber distributions.
        Samples distributions at fixed chord positions for feature vectors.

        Args:
            n_samples: Number of points to sample from each distribution
        """
        geometry_df = self.db.get_airfoil_geometry_dataframe()

        logging.info("Extracting distribution-based features...")

        feature_vectors = []
        valid_airfoils = []
        valid_ids = []

        # Create unique IDs
        unique_names = geometry_df["name"].unique()
        name_to_id = {name: idx for idx, name in enumerate(unique_names)}

        for idx, row in geometry_df.iterrows():
            name = row["name"]

            # Get distributions
            thickness_dist = row.get("thickness_distribution", "")
            camber_dist = row.get("camber_distribution", "")
            normalized_chord = row.get("normalized_chord", "")

            # Skip if distributions are missing
            if not thickness_dist or not camber_dist or not normalized_chord:
                logging.debug(f"Skipping {name}: missing distribution data")
                continue

            try:
                # Parse distributions from comma-separated strings
                thickness = np.array([float(x) for x in thickness_dist.split(",")])
                camber = np.array([float(x) for x in camber_dist.split(",")])
                chord = np.array([float(x) for x in normalized_chord.split(",")])

                # Validate data
                if len(thickness) < 3 or len(camber) < 3 or len(chord) < 3:
                    logging.debug(f"Skipping {name}: insufficient distribution points")
                    continue

                # Interpolate to fixed number of samples for consistent feature vectors
                # Sample at evenly spaced chord positions from 0 to 1
                target_chord = np.linspace(0, 1, n_samples)

                # Interpolate thickness distribution
                thickness_interp = interp1d(
                    chord,
                    thickness,
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate",
                )
                thickness_sampled = thickness_interp(target_chord)

                # Interpolate camber distribution
                camber_interp = interp1d(
                    chord,
                    camber,
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate",
                )
                camber_sampled = camber_interp(target_chord)

                # Add scalar features for additional context
                max_thickness = row["max_thickness"]
                max_camber = row["max_camber"]
                leading_edge_radius = row["leading_edge_radius"]
                trailing_edge_angle = row["trailing_edge_angle"]

                # Find positions of max thickness and max camber
                max_t_pos = target_chord[np.argmax(thickness_sampled)]
                max_c_pos = target_chord[np.argmax(np.abs(camber_sampled))]

                # Combine all features into a single vector:
                # - Sampled thickness distribution (n_samples values)
                # - Sampled camber distribution (n_samples values)
                # - Scalar features (6 values)
                # Total: 2*n_samples + 6 features
                feature_vector = np.concatenate(
                    [
                        thickness_sampled,
                        camber_sampled,
                        [
                            max_thickness,
                            max_camber,
                            leading_edge_radius,
                            trailing_edge_angle,
                            max_t_pos,
                            max_c_pos,
                        ],
                    ]
                )

                feature_vectors.append(feature_vector)
                valid_airfoils.append(name)
                valid_ids.append(name_to_id[name])

            except Exception as e:
                logging.debug(f"Error processing {name}: {e}")
                continue

        if len(feature_vectors) == 0:
            raise ValueError("No valid distribution features could be extracted!")

        X = np.array(feature_vectors)
        y = np.array(valid_ids)

        # Create feature names for documentation
        self.feature_names = (
            [f"thickness_at_{i/(n_samples-1):.2f}" for i in range(n_samples)]
            + [f"camber_at_{i/(n_samples-1):.2f}" for i in range(n_samples)]
            + [
                "max_thickness",
                "max_camber",
                "leading_edge_radius",
                "trailing_edge_angle",
                "max_thickness_position",
                "max_camber_position",
            ]
        )

        logging.info(f"Extracted {len(self.feature_names)} distribution-based features")
        logging.info(f"  - {n_samples} thickness distribution samples")
        logging.info(f"  - {n_samples} camber distribution samples")
        logging.info(f"  - 6 scalar geometric features")
        logging.info(f"Dataset: {len(X)} samples, {len(np.unique(y))} unique airfoils")

        return X, y, np.array(valid_airfoils)

    def analyze_feature_quality(self):
        """Analyze the quality of extracted features."""

        print("\n" + "=" * 80)
        print("FEATURE QUALITY ANALYSIS")
        print("=" * 80)

        # Check variance of features
        feature_vars = np.var(self.X_scaled, axis=0)

        print(f"\nFeature Statistics:")
        print(f"  Total features: {len(self.feature_names)}")
        print(f"  Features with variance > 0.1: {np.sum(feature_vars > 0.1)}")
        print(f"  Features with variance < 0.01: {np.sum(feature_vars < 0.01)}")

        # Identify low-variance features
        low_var_indices = np.where(feature_vars < 0.01)[0]
        if len(low_var_indices) > 0:
            print(f"\n⚠️  Low variance features:")
            for idx in low_var_indices[:10]:  # Show first 10
                print(f"    {self.feature_names[idx]}: var={feature_vars[idx]:.6f}")

        # Check for perfect correlations
        if len(self.feature_names) <= 50:  # Only for manageable sizes
            corr_matrix = np.corrcoef(self.X_scaled.T)
            perfect_corr = []
            for i in range(len(self.feature_names)):
                for j in range(i + 1, len(self.feature_names)):
                    if abs(corr_matrix[i, j]) > 0.99:
                        perfect_corr.append((i, j, corr_matrix[i, j]))

            if perfect_corr:
                print(f"\n⚠️  Near-perfect correlations detected:")
                for i, j, corr in perfect_corr[:5]:  # Show first 5
                    print(
                        f"    {self.feature_names[i]} <-> {self.feature_names[j]}: r={corr:.4f}"
                    )

        # Check distance distribution
        from sklearn.metrics.pairwise import euclidean_distances

        # Sample for efficiency
        if len(self.X_scaled) > 200:
            sample_idx = np.random.choice(len(self.X_scaled), 200, replace=False)
            X_sample = self.X_scaled[sample_idx]
        else:
            X_sample = self.X_scaled

        distances = euclidean_distances(X_sample)
        distances = distances[np.triu_indices_from(distances, k=1)]

        print(f"\nPairwise Distance Distribution:")
        print(f"  Min: {distances.min():.4f}")
        print(f"  Mean: {distances.mean():.4f}")
        print(f"  Median: {np.median(distances):.4f}")
        print(f"  Max: {distances.max():.4f}")
        print(f"  Std: {distances.std():.4f}")

        if distances.std() < 0.5:
            print(f"  ⚠️  WARNING: Low distance variance suggests poor discrimination")
        else:
            print(f"  ✅ Good distance distribution for similarity search")

        print("=" * 80)

    def setup_similarity_search_models(self):
        """Setup similarity search models."""

        self.models = {
            "KNN_Euclidean_5": {
                "model": NearestNeighbors(
                    n_neighbors=5, metric="euclidean", algorithm="auto"
                ),
                "description": "Euclidean distance on distribution features",
            },
            "KNN_Euclidean_10": {
                "model": NearestNeighbors(
                    n_neighbors=10, metric="euclidean", algorithm="auto"
                ),
                "description": "Extended neighborhood search",
            },
            "KNN_Manhattan_5": {
                "model": NearestNeighbors(
                    n_neighbors=5, metric="manhattan", algorithm="auto"
                ),
                "description": "Manhattan (L1) distance",
            },
            "KNN_Cosine_5": {
                "model": NearestNeighbors(
                    n_neighbors=5, metric="cosine", algorithm="brute"
                ),
                "description": "Cosine similarity (shape-based)",
            },
        }

        logging.info(f"Initialized {len(self.models)} similarity search models")
        return self.models

    def evaluate_similarity_search_robustness(
        self, k_values=[1, 5, 10], noise_level=0.01
    ):
        """
        Evaluate retrieval by adding noise to queries and searching the full database.
        This simulates real-world usage: "Can we identify this airfoil if the scan is imperfect?"
        """
        n_samples = len(self.y)
        logging.info(f"Evaluating robustness with noise_level={noise_level}...")

        # 1. Fit the model on the FULL dataset (The Haystack contains the Needles)
        # We want to see if we can retrieve the specific ID from the full DB

        results = {}

        # Add random noise to creating 'simulated queries'
        # We assume X_scaled is standardized (mean 0, std 1), so noise_level is in std deviations.
        np.random.seed(42)  # For reproducibility
        noise = np.random.normal(0, noise_level, self.X_scaled.shape)
        X_queries = self.X_scaled + noise

        for model_name, model_config in self.models.items():
            model = model_config["model"]

            # Fit on the clean, full database
            model.fit(self.X_scaled)

            start_time = time.time()
            # Query with the NOISY versions
            distances, indices = model.kneighbors(X_queries, n_neighbors=max(k_values))
            inference_time_total = (time.time() - start_time) * 1000
            mean_inference_time = inference_time_total / n_samples

            # Calculate Metrics
            reciprocal_ranks = []
            top_k_hits = {k: 0 for k in k_values}

            for i in range(n_samples):
                true_id = self.y[i]
                predicted_indices = indices[i]
                predicted_ids = self.y[predicted_indices]

                # Check hits
                if true_id in predicted_ids:
                    rank = np.where(predicted_ids == true_id)[0][0] + 1
                    reciprocal_ranks.append(1.0 / rank)
                else:
                    reciprocal_ranks.append(0.0)

                for k in k_values:
                    if true_id in predicted_ids[:k]:
                        top_k_hits[k] += 1

            self.results[model_name] = {
                "mean_reciprocal_rank": np.mean(reciprocal_ranks),
                "top_k_accuracy": {
                    k: (hits / n_samples) * 100 for k, hits in top_k_hits.items()
                },
                "mean_inference_time_ms": mean_inference_time,
                "description": model_config["description"],
            }

            logging.info(
                f"{model_name}: MRR={np.mean(reciprocal_ranks):.3f}, "
                f"Top-1={self.results[model_name]['top_k_accuracy'][1]:.1f}%"
            )

        return self.results

    def evaluate_baseline_methods(self):
        """Evaluate baseline methods."""

        n_airfoils = len(np.unique(self.y))

        self.results["Random_Selection"] = {
            "mean_reciprocal_rank": 1.0 / n_airfoils,
            "top_k_accuracy": {
                1: (1.0 / n_airfoils) * 100,
                5: min((5.0 / n_airfoils) * 100, 100.0),
                10: min((10.0 / n_airfoils) * 100, 100.0),
            },
            "mean_inference_time_ms": 0.001,
            "description": f"Random selection baseline",
        }

        logging.info(
            f"Random baseline: Top-1={self.results['Random_Selection']['top_k_accuracy'][1]:.3f}%"
        )

        return self.results

    def assess_dod_mission_criteria(self):
        """Assess against DoD mission requirements."""

        valid_models = {
            name: metrics
            for name, metrics in self.results.items()
            if not name.startswith("Random_")
        }

        if not valid_models:
            return {"error": "No valid models"}

        best_name = max(valid_models.items(), key=lambda x: x[1]["top_k_accuracy"][1])[
            0
        ]
        best_metrics = valid_models[best_name]

        mission_assessment = {
            "best_model": best_name,
            "top1_accuracy": best_metrics["top_k_accuracy"][1],
            "top5_accuracy": best_metrics["top_k_accuracy"][5],
            "top10_accuracy": best_metrics["top_k_accuracy"][10],
            "mrr": best_metrics["mean_reciprocal_rank"],
            "inference_time": best_metrics["mean_inference_time_ms"],
            "meets_95_percent": best_metrics["top_k_accuracy"][1] >= 95.0,
            "meets_10ms": best_metrics["mean_inference_time_ms"] <= 10.0,
            "mission_ready": (
                best_metrics["top_k_accuracy"][1] >= 95.0
                and best_metrics["mean_inference_time_ms"] <= 10.0
            ),
        }

        print(f"\n" + "=" * 80)
        print(f"DoD MISSION CRITERIA ASSESSMENT")
        print("=" * 80)
        print(f"Best performing model: {mission_assessment['best_model']}")
        print(f"")
        print(f"📊 Retrieval Accuracy:")
        print(
            f"   Top-1:  {mission_assessment['top1_accuracy']:.1f}% {'✅' if mission_assessment['top1_accuracy'] >= 95 else '❌'} (Target: 95%)"
        )
        print(f"   Top-5:  {mission_assessment['top5_accuracy']:.1f}%")
        print(f"   Top-10: {mission_assessment['top10_accuracy']:.1f}%")
        print(f"")
        print(f"⚡ Performance:")
        print(f"   Mean Reciprocal Rank: {mission_assessment['mrr']:.3f}")
        print(
            f"   Inference Time: {mission_assessment['inference_time']:.3f}ms {'✅' if mission_assessment['meets_10ms'] else '❌'} (Target: <10ms)"
        )
        print(f"")
        print(
            f"🎯 Mission Status: {'✅ READY FOR DEPLOYMENT' if mission_assessment['mission_ready'] else '❌ NEEDS IMPROVEMENT'}"
        )
        print("=" * 80)

        return mission_assessment

    def generate_table4(self, output_dir="results"):
        """Generate Table 4 results."""

        table_data = []

        sorted_results = sorted(
            [(n, m) for n, m in self.results.items() if not n.startswith("Random_")],
            key=lambda x: x[1]["top_k_accuracy"][1],
            reverse=True,
        )

        random_results = [
            (n, m) for n, m in self.results.items() if n.startswith("Random_")
        ]
        all_results = sorted_results + random_results

        for name, metrics in all_results:
            table_data.append(
                {
                    "Algorithm": name.replace("_", " "),
                    "Top-1 (%)": f"{metrics['top_k_accuracy'][1]:.1f}",
                    "Top-5 (%)": f"{metrics['top_k_accuracy'][5]:.1f}",
                    "Top-10 (%)": f"{metrics['top_k_accuracy'][10]:.1f}",
                    "MRR": f"{metrics['mean_reciprocal_rank']:.3f}",
                    "Time (ms)": f"{metrics['mean_inference_time_ms']:.3f}",
                }
            )

        df = pd.DataFrame(table_data)

        Path(output_dir).mkdir(exist_ok=True)
        df.to_csv(Path(output_dir) / "table4_similarity_results.csv", index=False)

        print("\nTable 4. Similarity Search Results (Distribution-Based Features)")
        print("=" * 100)
        print(df.to_string(index=False))
        print("=" * 100)

        return df

    def create_visualizations(self, output_dir="results"):
        """Create performance visualizations."""

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Extract data (exclude random baseline from most plots)
        names = [
            n.replace("_", " ")
            for n in self.results.keys()
            if not n.startswith("Random_")
        ]
        top1 = [
            m["top_k_accuracy"][1]
            for n, m in self.results.items()
            if not n.startswith("Random_")
        ]
        top5 = [
            m["top_k_accuracy"][5]
            for n, m in self.results.items()
            if not n.startswith("Random_")
        ]
        top10 = [
            m["top_k_accuracy"][10]
            for n, m in self.results.items()
            if not n.startswith("Random_")
        ]
        mrr = [
            m["mean_reciprocal_rank"]
            for n, m in self.results.items()
            if not n.startswith("Random_")
        ]
        times = [
            m["mean_inference_time_ms"]
            for n, m in self.results.items()
            if not n.startswith("Random_")
        ]

        # Plot 1: Top-K Accuracy Comparison
        x = np.arange(len(names))
        width = 0.25
        ax1.bar(x - width, top1, width, label="Top-1", alpha=0.8, color="steelblue")
        ax1.bar(x, top5, width, label="Top-5", alpha=0.8, color="coral")
        ax1.bar(x + width, top10, width, label="Top-10", alpha=0.8, color="lightgreen")
        ax1.axhline(
            95, color="red", linestyle="--", linewidth=2, label="95% Target", alpha=0.7
        )
        ax1.set_ylabel("Retrieval Accuracy (%)", fontweight="bold", fontsize=11)
        ax1.set_title("Top-K Retrieval Performance", fontsize=12, fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=45, ha="right")
        ax1.legend(loc="lower right")
        ax1.grid(True, alpha=0.3, axis="y")

        # Plot 2: Mean Reciprocal Rank
        bars = ax2.bar(names, mrr, alpha=0.8, color="mediumpurple", edgecolor="black")
        ax2.set_ylabel("Mean Reciprocal Rank", fontweight="bold", fontsize=11)
        ax2.set_title(
            "Ranking Quality (Higher = Better)", fontsize=12, fontweight="bold"
        )
        ax2.set_xticklabels(names, rotation=45, ha="right")
        ax2.set_ylim(0, 1.0)
        ax2.grid(True, alpha=0.3, axis="y")
        for bar, val in zip(bars, mrr):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.3f}",
                ha="center",
                fontsize=10,
                fontweight="bold",
            )

        # Plot 3: Inference Time
        bars = ax3.bar(names, times, alpha=0.8, color="darkorange", edgecolor="black")
        ax3.axhline(
            10, color="red", linestyle="--", linewidth=2, label="10ms Target", alpha=0.7
        )
        ax3.set_ylabel("Inference Time (ms)", fontweight="bold", fontsize=11)
        ax3.set_title("Query Response Time", fontsize=12, fontweight="bold")
        ax3.set_xticklabels(names, rotation=45, ha="right")
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis="y")

        # Plot 4: Accuracy vs Speed Trade-off
        scatter = ax4.scatter(
            times,
            top1,
            s=200,
            alpha=0.7,
            c=mrr,
            cmap="viridis",
            edgecolors="black",
            linewidth=1.5,
        )
        for i, name in enumerate(names):
            ax4.annotate(
                name,
                (times[i], top1[i]),
                fontsize=9,
                ha="right",
                xytext=(-5, 5),
                textcoords="offset points",
            )
        ax4.axhline(95, color="red", linestyle="--", alpha=0.5, linewidth=2)
        ax4.axvline(10, color="red", linestyle="--", alpha=0.5, linewidth=2)

        # Add colored regions
        ax4.fill_between(
            [0, 10], 95, 100, alpha=0.1, color="green", label="Meets Both Targets"
        )

        ax4.set_xlabel("Inference Time (ms)", fontweight="bold", fontsize=11)
        ax4.set_ylabel("Top-1 Accuracy (%)", fontweight="bold", fontsize=11)
        ax4.set_title("Speed vs Accuracy Trade-off", fontsize=12, fontweight="bold")
        ax4.grid(True, alpha=0.3)
        ax4.legend()

        # Add colorbar for MRR
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label("Mean Reciprocal Rank", fontweight="bold")

        plt.suptitle(
            "Airfoil Geometric Similarity Search: Distribution-Based Features\n"
            + "DoD Mission: Rapid Component Identification & Quality Assurance",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        plt.savefig(
            Path(output_dir) / "performance_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        logging.info(f"Visualizations saved to {output_dir}")

    def generate_comprehensive_report(self, output_dir="results"):
        """Generate complete report."""

        Path(output_dir).mkdir(exist_ok=True)

        table4 = self.generate_table4(output_dir)
        assessment = self.assess_dod_mission_criteria()
        self.create_visualizations(output_dir)

        # Save summary
        import json

        summary = {
            "mission_assessment": assessment,
            "feature_engineering": {
                "approach": "Distribution-based features",
                "features_used": len(self.feature_names),
                "feature_breakdown": {
                    "thickness_distribution_samples": sum(
                        1 for f in self.feature_names if "thickness_at" in f
                    ),
                    "camber_distribution_samples": sum(
                        1 for f in self.feature_names if "camber_at" in f
                    ),
                    "scalar_features": 6,
                },
                "advantages": [
                    "Captures full airfoil shape information",
                    "Much better discrimination than scalar features alone",
                    "Robust to variations in airfoil complexity",
                ],
            },
            "results": {
                k: {
                    "top1_accuracy": v["top_k_accuracy"][1],
                    "top5_accuracy": v["top_k_accuracy"][5],
                    "mrr": v["mean_reciprocal_rank"],
                    "inference_time_ms": v["mean_inference_time_ms"],
                }
                for k, v in self.results.items()
            },
        }

        with open(Path(output_dir) / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"\n✅ Comprehensive report generated:")
        print(f"   📄 Table 4: {output_dir}/table4_similarity_results.csv")
        print(f"   📊 Visualizations: {output_dir}/performance_analysis.png")
        print(f"   📋 Summary: {output_dir}/summary.json")

        return summary


def main():
    """Execute distribution-based similarity search pipeline."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 80)
    print("AIRFOIL GEOMETRIC SIMILARITY SEARCH")
    print("Enhanced with Distribution-Based Features")
    print("=" * 80)

    from airfoil_database.core.database import AirfoilDatabase

    try:
        db = AirfoilDatabase(db_name="airfoils_preprocessed.db", db_dir=".")
        logging.info("Connected to database")
    except Exception as e:
        logging.error(f"Database connection failed: {e}")
        return

    # Initialize pipeline
    pipeline = DistributionBasedSimilaritySearch(db)

    # Extract distribution-based features
    print("\n📊 Extracting distribution-based features...")
    X, y, names = pipeline.extract_distribution_features(n_samples=20)

    # Scale features
    pipeline.X_scaled = pipeline.scaler.fit_transform(X)
    pipeline.y = y
    pipeline.airfoil_names = names

    # Analyze feature quality
    pipeline.analyze_feature_quality()

    # Setup models
    pipeline.setup_similarity_search_models()

    # Evaluate
    pipeline.evaluate_similarity_search_robustness()
    pipeline.evaluate_baseline_methods()

    # Generate report
    pipeline.generate_comprehensive_report("classical_modeling_results")

    print("\n" + "=" * 80)
    print("✅ SIMILARITY SEARCH ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
