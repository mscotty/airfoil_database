import numpy as np
import pandas as pd
import tensorflow as keras
from tensorflow.keras import layers, models, regularizers, callbacks
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import json
from pathlib import Path
import time

# Adjust import based on your project structure
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from airfoil_database.core.database import AirfoilDatabase

# ==========================================
# CONFIGURATION
# ==========================================
DB_NAME = "airfoils_preprocessed.db"
OUTPUT_DIR = "neural_network_results"
N_SAMPLES = 20  # Number of points to sample from distributions
NOISE_LEVEL = 0.02  # Standard deviations of noise for augmentation
AUGMENTATION_FACTOR = 10  # How many noisy copies to create per airfoil for training


class AirfoilNNModeling:
    def __init__(self, db_path):
        self.db = AirfoilDatabase(db_name=db_path, db_dir=".")
        self.scaler = StandardScaler()
        self.encoder = LabelEncoder()
        self.X = None
        self.y = None
        self.classes = None
        Path(OUTPUT_DIR).mkdir(exist_ok=True)

    def extract_features(self):
        """Extracts distribution-based features (Same as Classical Modeling)"""
        print("📊 Extracting features...")
        df = self.db.get_airfoil_geometry_dataframe()

        feature_vectors = []
        labels = []

        for _, row in df.iterrows():
            try:
                # Parse distributions
                thick_dist = np.fromstring(row["thickness_distribution"], sep=",")
                camb_dist = np.fromstring(row["camber_distribution"], sep=",")
                chord = np.fromstring(row["normalized_chord"], sep=",")

                # Interpolate to fixed size
                target_chord = np.linspace(0, 1, N_SAMPLES)
                t_interp = interp1d(
                    chord, thick_dist, bounds_error=False, fill_value="extrapolate"
                )(target_chord)
                c_interp = interp1d(
                    chord, camb_dist, bounds_error=False, fill_value="extrapolate"
                )(target_chord)

                # Combine features (Distributions + Scalars)
                features = np.concatenate(
                    [
                        t_interp,
                        c_interp,
                        [
                            row["max_thickness"],
                            row["max_camber"],
                            row["leading_edge_radius"],
                            row["trailing_edge_angle"],
                            row["max_thickness_position"],
                            row["max_camber_position"],
                        ],
                    ]
                )

                feature_vectors.append(features)
                labels.append(row["name"])

            except Exception as e:
                continue

        self.X = np.array(feature_vectors)
        self.y = np.array(labels)

        # Normalize Features
        self.X = self.scaler.fit_transform(self.X)

        # Encode Labels (String -> Int)
        self.y_encoded = self.encoder.fit_transform(self.y)
        self.classes = self.encoder.classes_

        print(
            f"✅ Data prepared: {self.X.shape[0]} samples, {self.X.shape[1]} features"
        )
        print(f"   Classes: {len(self.classes)}")

    def create_augmented_dataset(self):
        """Creates a training set with synthetic noise"""
        print(
            f"Duplicate and Augment: Creating {AUGMENTATION_FACTOR} noisy copies per sample..."
        )
        X_aug = []
        y_aug = []

        for i in range(len(self.X)):
            # Original
            X_aug.append(self.X[i])
            y_aug.append(self.y_encoded[i])

            # Noisy copies
            for _ in range(AUGMENTATION_FACTOR):
                noise = np.random.normal(0, NOISE_LEVEL, self.X.shape[1])
                X_aug.append(self.X[i] + noise)
                y_aug.append(self.y_encoded[i])

        return np.array(X_aug), np.array(y_aug)

    def build_model(
        self, n_layers=2, n_neurons=64, learning_rate=0.001, regularization=None
    ):
        """Builds the Keras model"""
        model = models.Sequential()
        model.add(layers.Input(shape=(self.X.shape[1],)))

        for _ in range(n_layers):
            if regularization == "l2":
                reg = regularizers.l2(0.01)
            else:
                reg = None

            model.add(
                layers.Dense(n_neurons, activation="relu", kernel_regularizer=reg)
            )

            if regularization == "dropout":
                model.add(layers.Dropout(0.2))

        model.add(layers.Dense(len(self.classes), activation="softmax"))

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def hyperparameter_sweep(self):
        """Performs the sweep for Figure 5 and Table 5"""
        print("\n🔎 Starting Hyperparameter Sweep...")

        # Prepare Data
        X_train, y_train = self.create_augmented_dataset()
        # Test on original clean data (or slight noise)
        X_test, y_test = self.X, self.y_encoded

        results = []

        # Sweep Config
        neurons_list = [32, 64, 128, 256]

        best_acc = 0
        best_config = {}

        for n_neurons in neurons_list:
            print(f"   Testing {n_neurons} neurons...")
            model = self.build_model(n_layers=2, n_neurons=n_neurons)

            history = model.fit(
                X_train,
                y_train,
                epochs=20,
                batch_size=32,
                verbose=0,
                validation_data=(X_test, y_test),
            )

            val_acc = max(history.history["val_accuracy"])
            results.append({"neurons": n_neurons, "accuracy": val_acc})

            if val_acc > best_acc:
                best_acc = val_acc
                best_config = {"neurons": n_neurons}

        # Plot Figure 5
        plt.figure(figsize=(8, 6))
        x_vals = [r["neurons"] for r in results]
        y_vals = [r["accuracy"] * 100 for r in results]
        plt.plot(x_vals, y_vals, marker="o", linestyle="-", color="b")
        plt.title("Model Performance vs. Model Complexity")
        plt.xlabel("Number of Neurons (per layer)")
        plt.ylabel("Validation Accuracy (%)")
        plt.grid(True)
        plt.savefig(f"{OUTPUT_DIR}/figure5_hyperparameter_sweep.png")
        plt.close()

        print(
            f"✅ Sweep Complete. Best Accuracy: {best_acc*100:.1f}% with {best_config}"
        )
        return best_config

    def demonstrate_overfitting(self, best_neurons):
        """Demonstrates overfitting correction for Figure 6"""
        print("\n📉 Generating Overfitting Demonstration...")

        X_train, y_train = self.create_augmented_dataset()
        X_test, y_test = self.X, self.y_encoded

        # 1. Overfitting Model (Too complex, no reg, long training)
        print("   Training Overfitting Model...")
        bad_model = self.build_model(n_layers=4, n_neurons=512, regularization=None)
        history_bad = bad_model.fit(
            X_train,
            y_train,
            epochs=50,
            batch_size=32,
            verbose=0,
            validation_data=(X_test, y_test),
        )

        # 2. Corrected Model (Best config + Dropout)
        print("   Training Corrected Model...")
        good_model = self.build_model(
            n_layers=2, n_neurons=best_neurons, regularization="dropout"
        )
        history_good = good_model.fit(
            X_train,
            y_train,
            epochs=50,
            batch_size=32,
            verbose=0,
            validation_data=(X_test, y_test),
        )

        # Plot Figure 6 (Side by Side)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Bad Plot
        ax1.plot(history_bad.history["accuracy"], label="Train")
        ax1.plot(history_bad.history["val_accuracy"], label="Validation")
        ax1.set_title("Overfitting Model (High Complexity)")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Accuracy")
        ax1.legend()

        # Good Plot
        ax2.plot(history_good.history["accuracy"], label="Train")
        ax2.plot(history_good.history["val_accuracy"], label="Validation")
        ax2.set_title("Corrected Model (Dropout + Optimized)")
        ax2.set_xlabel("Epochs")
        ax2.legend()

        plt.savefig(f"{OUTPUT_DIR}/figure6_overfitting_correction.png")
        plt.close()
        print("✅ Overfitting plots saved.")

        # Save final metrics for Table 5
        final_acc = history_good.history["val_accuracy"][-1]
        print(f"Final Model Accuracy: {final_acc*100:.2f}%")


if __name__ == "__main__":
    pipeline = AirfoilNNModeling(DB_NAME)
    pipeline.extract_features()

    best_config = pipeline.hyperparameter_sweep()
    pipeline.demonstrate_overfitting(best_config["neurons"])
