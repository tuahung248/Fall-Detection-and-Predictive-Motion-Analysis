import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import glob
import warnings
import json
import re
from datetime import datetime

warnings.filterwarnings("ignore")


class EnhancedAccelerometerAnalyzer:
    def __init__(self, data_dir="accel_logs/data_files"):
        self.data_dir = data_dir
        self.processed_data = []
        self.features_df = None
        self.gps_data = []

    def parse_csv_metadata(self, filepath):
        metadata = {}
        try:
            with open(filepath, "r") as f:
                lines = f.readlines()

            for line in lines:
                if line.startswith("# ") and ":" in line:
                    key_value = line[2:].strip()
                    if ": " in key_value:
                        key, value = key_value.split(": ", 1)
                        metadata[key.lower().replace(" ", "_")] = value
                elif line.startswith("# Data"):
                    break

        except Exception as e:
            print(f"Error parsing metadata from {filepath}: {e}")

        return metadata

    def load_accelerometer_data(self, filepath):
        try:
            with open(filepath, "r") as f:
                lines = f.readlines()

            data_start = 0
            for i, line in enumerate(lines):
                if "timestamp" in line and ("ax" in line or "x" in line):
                    data_start = i
                    break

            df = pd.read_csv(filepath, skiprows=data_start)

            if "timestamp" in df.columns:
                df["timestamp_utc"] = pd.to_datetime(df["timestamp"])
            elif "timestamp_utc" in df.columns:
                df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])

            column_mapping = {
                "ax": "x",
                "ay": "y",
                "az": "z",
                "acc_x": "x",
                "acc_y": "y",
                "acc_z": "z",
            }
            df = df.rename(columns=column_mapping)

            if "a_mag" not in df.columns and all(
                col in df.columns for col in ["x", "y", "z"]
            ):
                df["a_mag"] = np.sqrt(df["x"] ** 2 + df["y"] ** 2 + df["z"] ** 2)

            return df

        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None

    def extract_advanced_features(self, df):
        features = {}

        for axis in ["x", "y", "z"]:
            if axis in df.columns:
                data = df[axis].values
                features[f"{axis}_mean"] = np.mean(data)
                features[f"{axis}_std"] = np.std(data)
                features[f"{axis}_var"] = np.var(data)
                features[f"{axis}_min"] = np.min(data)
                features[f"{axis}_max"] = np.max(data)
                features[f"{axis}_range"] = np.max(data) - np.min(data)
                features[f"{axis}_skew"] = stats.skew(data)
                features[f"{axis}_kurtosis"] = stats.kurtosis(data)
                features[f"{axis}_rms"] = np.sqrt(np.mean(data**2))
                features[f"{axis}_q25"] = np.percentile(data, 25)
                features[f"{axis}_q75"] = np.percentile(data, 75)
                features[f"{axis}_iqr"] = (
                    features[f"{axis}_q75"] - features[f"{axis}_q25"]
                )

                zero_crossings = np.where(np.diff(np.signbit(data)))[0]
                features[f"{axis}_zero_crossings"] = len(zero_crossings)

                from scipy.signal import find_peaks

                peaks, _ = find_peaks(np.abs(data))
                features[f"{axis}_peak_count"] = len(peaks)

        if "a_mag" in df.columns:
            mag_data = df["a_mag"].values
        else:
            mag_data = np.sqrt(df["x"] ** 2 + df["y"] ** 2 + df["z"] ** 2)

        features["magnitude_mean"] = np.mean(mag_data)
        features["magnitude_std"] = np.std(mag_data)
        features["magnitude_var"] = np.var(mag_data)
        features["magnitude_min"] = np.min(mag_data)
        features["magnitude_max"] = np.max(mag_data)
        features["magnitude_range"] = np.max(mag_data) - np.min(mag_data)
        features["magnitude_skew"] = stats.skew(mag_data)
        features["magnitude_kurtosis"] = stats.kurtosis(mag_data)

        baseline_deviation = np.abs(mag_data - 1.0)
        features["baseline_deviation_mean"] = np.mean(baseline_deviation)
        features["baseline_deviation_std"] = np.std(baseline_deviation)

        features["movement_score"] = (
            features["magnitude_var"]
            + (features["baseline_deviation_mean"] * 2)
            + (features["magnitude_range"] * 0.5)
        )

        if all(col in df.columns for col in ["x", "y", "z"]):
            features["xy_corr"] = np.corrcoef(df["x"], df["y"])[0, 1]
            features["xz_corr"] = np.corrcoef(df["x"], df["z"])[0, 1]
            features["yz_corr"] = np.corrcoef(df["y"], df["z"])[0, 1]

        if len(df) > 1:
            features["sample_count"] = len(df)
            features["duration_seconds"] = len(df) * 0.1

        return features

    def load_and_process_all_data(self):
        csv_files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        print(f"Found {len(csv_files)} CSV files")

        if len(csv_files) == 0:
            print(f"No CSV files found in {self.data_dir}")
            return False

        processed_count = 0

        for csv_file in csv_files:
            try:
                metadata = self.parse_csv_metadata(csv_file)

                df = self.load_accelerometer_data(csv_file)
                if df is None or len(df) < 5:
                    continue

                features = self.extract_advanced_features(df)

                filename = os.path.basename(csv_file)
                features["filename"] = filename
                features["filepath"] = csv_file

                filename_parts = filename.replace(".csv", "").split("_")
                if len(filename_parts) >= 3:
                    features["sequence_number"] = int(filename_parts[0])
                    features["timestamp"] = filename_parts[1]
                    features["detected_activity"] = "_".join(filename_parts[2:])

                for key, value in metadata.items():
                    if key == "activity":
                        features["activity_label"] = value
                    elif key == "alert":
                        features["alert_flag"] = value.lower() == "true"
                    elif key == "gps_latitude":
                        try:
                            features["gps_lat"] = float(value)
                        except:
                            features["gps_lat"] = None
                    elif key == "gps_longitude":
                        try:
                            features["gps_lon"] = float(value)
                        except:
                            features["gps_lon"] = None
                    elif key == "sequence":
                        features["sequence_id"] = value
                    elif key == "collection_time":
                        features["collection_duration"] = value

                if features.get("gps_lat") and features.get("gps_lon"):
                    self.gps_data.append(
                        {
                            "filename": filename,
                            "activity": features.get(
                                "activity_label",
                                features.get("detected_activity", "unknown"),
                            ),
                            "lat": features["gps_lat"],
                            "lon": features["gps_lon"],
                            "alert": features.get("alert_flag", False),
                        }
                    )

                self.processed_data.append(features)
                processed_count += 1

            except Exception as e:
                print(f"Error processing {csv_file}: {e}")
                continue

        if processed_count > 0:
            self.features_df = pd.DataFrame(self.processed_data)
            print(f"Successfully processed {processed_count} files")

            self.features_df["activity_clean"] = self.features_df.apply(
                lambda row: row.get(
                    "activity_label", row.get("detected_activity", "unknown")
                ),
                axis=1,
            )

            return True
        else:
            print("No files were successfully processed")
            return False

    def analyze_detection_accuracy(self):
        if self.features_df is None:
            print("No data loaded. Run load_and_process_all_data() first.")
            return

        print("\nDetection Algorithm Analysis")
        print("=" * 40)

        activity_counts = self.features_df["activity_clean"].value_counts()
        print(f"\nActivity Distribution:")
        for activity, count in activity_counts.items():
            percentage = (count / len(self.features_df)) * 100
            print(f"{activity:20}: {count:3d} samples ({percentage:5.1f}%)")

        if "alert_flag" in self.features_df.columns:
            alert_count = self.features_df["alert_flag"].sum()
            print(
                f"\nAlerts: {alert_count}/{len(self.features_df)} ({alert_count/len(self.features_df)*100:.1f}%)"
            )

            if alert_count > 0:
                alert_activities = self.features_df[
                    self.features_df["alert_flag"] == True
                ]["activity_clean"].value_counts()
                print("Alert activities:")
                for activity, count in alert_activities.items():
                    print(f"  {activity}: {count} alerts")

        print(f"\nActivity Stats:")
        print("-" * 40)

        key_features = [
            "magnitude_mean",
            "magnitude_std",
            "movement_score",
            "baseline_deviation_mean",
        ]

        for activity in activity_counts.index:
            activity_data = self.features_df[
                self.features_df["activity_clean"] == activity
            ]
            print(f"\n{activity.upper().replace('_', ' ')} (n={len(activity_data)}):")

            for feature in key_features:
                if feature in activity_data.columns:
                    mean_val = activity_data[feature].mean()
                    std_val = activity_data[feature].std()
                    print(f"  {feature:25}: {mean_val:6.4f} ± {std_val:6.4f}")

    def analyze_gps_patterns(self):
        if not self.gps_data:
            print("No GPS data available")
            return

        print(f"\nGPS Analysis")
        print("-" * 20)

        gps_df = pd.DataFrame(self.gps_data)
        print(f"GPS samples: {len(gps_df)}")

        location_activities = gps_df.groupby(["lat", "lon"])["activity"].value_counts()
        print(f"\nLocations:")
        for (lat, lon), activities in location_activities.groupby(level=[0, 1]):
            print(f"({lat:.6f}, {lon:.6f}):")
            for activity, count in activities.items():
                print(f"  {activity}: {count}")

        alert_locations = gps_df[gps_df["alert"] == True]
        if len(alert_locations) > 0:
            print(f"\nAlert locations:")
            for _, row in alert_locations.iterrows():
                print(f"  {row['activity']} at ({row['lat']:.6f}, {row['lon']:.6f})")

    def create_enhanced_visualizations(self):
        print(f"\nCreating visualizations...")
        os.makedirs("accel_logs/analysis", exist_ok=True)

        self.plot_activity_detection_performance()
        self.plot_feature_importance()
        if self.gps_data:
            self.plot_gps_activity_map()
        self.plot_enhanced_time_series()
        self.perform_classification_analysis()

    def plot_activity_detection_performance(self):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        activity_counts = self.features_df["activity_clean"].value_counts()
        colors = plt.cm.Set3(np.linspace(0, 1, len(activity_counts)))
        ax1.pie(
            activity_counts.values,
            labels=activity_counts.index,
            autopct="%1.1f%%",
            colors=colors,
        )
        ax1.set_title("Activity Distribution")

        sns.boxplot(
            data=self.features_df, x="activity_clean", y="movement_score", ax=ax2
        )
        ax2.set_title("Movement Score by Activity")
        ax2.tick_params(axis="x", rotation=45)

        magnitude_features = ["magnitude_mean", "magnitude_std", "magnitude_range"]
        activity_means = self.features_df.groupby("activity_clean")[
            magnitude_features
        ].mean()

        x = np.arange(len(activity_means.index))
        width = 0.25

        for i, feature in enumerate(magnitude_features):
            ax3.bar(x + i * width, activity_means[feature], width, label=feature)

        ax3.set_xlabel("Activity")
        ax3.set_ylabel("Magnitude (g)")
        ax3.set_title("Magnitude Features")
        ax3.set_xticks(x + width)
        ax3.set_xticklabels(activity_means.index, rotation=45)
        ax3.legend()

        if "alert_flag" in self.features_df.columns:
            alert_by_activity = self.features_df.groupby("activity_clean")[
                "alert_flag"
            ].agg(["sum", "count"])
            alert_rate = alert_by_activity["sum"] / alert_by_activity["count"] * 100

            ax4.bar(alert_rate.index, alert_rate.values, color="red", alpha=0.7)
            ax4.set_title("Alert Rate (%)")
            ax4.set_ylabel("Alert Rate (%)")
            ax4.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(
            "accel_logs/analysis/detection_performance.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
        print("  Detection performance saved")

    def plot_feature_importance(self):
        excluded_columns = [
            "filename",
            "filepath",
            "activity_clean",
            "detected_activity",
            "activity_label",
            "alert_flag",
            "timestamp",
            "gps_lat",
            "gps_lon",
            "sequence_id",
            "collection_duration",
            "sequence_number",
        ]

        feature_columns = []
        for col in self.features_df.columns:
            if col not in excluded_columns:
                try:
                    pd.to_numeric(self.features_df[col], errors="raise")
                    feature_columns.append(col)
                except (ValueError, TypeError):
                    continue

        X = self.features_df[feature_columns].fillna(0)
        y = self.features_df["activity_clean"]

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)

        importance = pd.DataFrame(
            {"feature": feature_columns, "importance": rf.feature_importances_}
        ).sort_values("importance", ascending=False)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        top_features = importance.head(20)
        ax1.barh(range(len(top_features)), top_features["importance"])
        ax1.set_yticks(range(len(top_features)))
        ax1.set_yticklabels(top_features["feature"])
        ax1.set_xlabel("Importance")
        ax1.set_title("Top Features")
        ax1.invert_yaxis()

        feature_categories = {
            "magnitude": [f for f in feature_columns if "magnitude" in f],
            "x_axis": [f for f in feature_columns if f.startswith("x_")],
            "y_axis": [f for f in feature_columns if f.startswith("y_")],
            "z_axis": [f for f in feature_columns if f.startswith("z_")],
            "correlation": [f for f in feature_columns if "corr" in f],
            "movement": [
                f for f in feature_columns if "movement" in f or "baseline" in f
            ],
        }

        category_importance = {}
        for category, features in feature_categories.items():
            category_features = [
                f for f in features if f in importance["feature"].values
            ]
            if category_features:
                category_importance[category] = importance[
                    importance["feature"].isin(category_features)
                ]["importance"].sum()

        if category_importance:
            ax2.pie(
                category_importance.values(),
                labels=category_importance.keys(),
                autopct="%1.1f%%",
            )
            ax2.set_title("Feature Categories")

        plt.tight_layout()
        plt.savefig(
            "accel_logs/analysis/feature_importance.png", dpi=150, bbox_inches="tight"
        )
        plt.close()
        print("  Feature importance saved")

    def plot_gps_activity_map(self):
        if not self.gps_data:
            return

        gps_df = pd.DataFrame(self.gps_data)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        activities = gps_df["activity"].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(activities)))

        for i, activity in enumerate(activities):
            activity_data = gps_df[gps_df["activity"] == activity]
            ax1.scatter(
                activity_data["lon"],
                activity_data["lat"],
                c=[colors[i]],
                label=activity,
                alpha=0.7,
                s=100,
            )

        ax1.set_xlabel("Longitude")
        ax1.set_ylabel("Latitude")
        ax1.set_title("Activity Locations")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        alert_data = gps_df[gps_df["alert"] == True]
        if len(alert_data) > 0:
            ax2.scatter(
                gps_df["lon"],
                gps_df["lat"],
                c="lightblue",
                alpha=0.5,
                s=50,
                label="Normal",
            )
            ax2.scatter(
                alert_data["lon"],
                alert_data["lat"],
                c="red",
                s=200,
                marker="X",
                label="Alerts",
            )
            ax2.set_xlabel("Longitude")
            ax2.set_ylabel("Latitude")
            ax2.set_title("Alert Locations")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            "accel_logs/analysis/gps_activity_map.png", dpi=150, bbox_inches="tight"
        )
        plt.close()
        print("  GPS map saved")

    def plot_enhanced_time_series(self):
        activities = self.features_df["activity_clean"].unique()
        samples_per_activity = min(
            3, self.features_df["activity_clean"].value_counts().min()
        )

        selected_files = []
        for activity in activities:
            activity_files = self.features_df[
                self.features_df["activity_clean"] == activity
            ]["filepath"].tolist()
            selected_files.extend(activity_files[:samples_per_activity])

        fig, axes = plt.subplots(
            len(selected_files), 1, figsize=(15, 3 * len(selected_files))
        )
        if len(selected_files) == 1:
            axes = [axes]

        for i, filepath in enumerate(selected_files[:12]):
            df = self.load_accelerometer_data(filepath)
            if df is None:
                continue

            filename = os.path.basename(filepath)
            activity = self.features_df[self.features_df["filepath"] == filepath][
                "activity_clean"
            ].iloc[0]

            axes[i].plot(df["x"], label="X", alpha=0.7, linewidth=1)
            axes[i].plot(df["y"], label="Y", alpha=0.7, linewidth=1)
            axes[i].plot(df["z"], label="Z", alpha=0.7, linewidth=1)

            if "a_mag" in df.columns:
                axes[i].plot(df["a_mag"], label="Mag", color="black", linewidth=2)

            axes[i].axhline(y=1, color="gray", linestyle="--", alpha=0.5, label="1g")
            axes[i].axhline(y=2.5, color="red", linestyle="--", alpha=0.7, label="Fall")

            axes[i].set_title(f'{activity.replace("_", " ").title()}: {filename}')
            axes[i].set_ylabel("Acceleration (g)")
            axes[i].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            axes[i].grid(True, alpha=0.3)

        if len(selected_files) > 0:
            axes[-1].set_xlabel("Sample")

        plt.suptitle("Time Series Analysis", fontsize=16)
        plt.tight_layout()
        plt.savefig(
            "accel_logs/analysis/enhanced_time_series.png", dpi=150, bbox_inches="tight"
        )
        plt.close()
        print("  Time series saved")

    def perform_classification_analysis(self):
        excluded_columns = [
            "filename",
            "filepath",
            "activity_clean",
            "detected_activity",
            "activity_label",
            "alert_flag",
            "timestamp",
            "gps_lat",
            "gps_lon",
            "sequence_id",
            "collection_duration",
            "sequence_number",
        ]

        feature_columns = []
        for col in self.features_df.columns:
            if col not in excluded_columns:
                try:
                    pd.to_numeric(self.features_df[col], errors="raise")
                    feature_columns.append(col)
                except (ValueError, TypeError):
                    continue

        X = self.features_df[feature_columns].fillna(0)
        y = self.features_df["activity_clean"]

        min_class_count = y.value_counts().min()
        if min_class_count < 2:
            print(f"Warning: Small classes ({min_class_count}). Using random split.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train_scaled, y_train)

        y_pred = rf.predict(X_test_scaled)

        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=rf.classes_,
            yticklabels=rf.classes_,
        )
        plt.title("Classification Results")
        plt.ylabel("True")
        plt.xlabel("Predicted")
        plt.tight_layout()
        plt.savefig(
            "accel_logs/analysis/classification_confusion_matrix.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        report = classification_report(y_test, y_pred)
        with open("accel_logs/analysis/classification_report.txt", "w") as f:
            f.write("Classification Analysis\n")
            f.write("=" * 25 + "\n\n")
            f.write("Random Forest Results:\n\n")
            f.write(report)
            f.write(f"\n\nAccuracy: {rf.score(X_test_scaled, y_test):.4f}")

        print("  Classification done")
        print(f"  Accuracy: {rf.score(X_test_scaled, y_test):.4f}")

    def generate_comprehensive_report(self):
        if self.features_df is None:
            print("No data available")
            return

        report_file = "accel_logs/analysis/comprehensive_analysis_report.txt"

        with open(report_file, "w") as f:
            f.write("Accelerometer Analysis Report\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("Data Overview\n")
            f.write("-" * 15 + "\n")
            f.write(f"Total samples: {len(self.features_df)}\n")
            f.write(f"Activities: {len(self.features_df['activity_clean'].unique())}\n")
            f.write(f"GPS samples: {len([d for d in self.gps_data])}\n")

            if "alert_flag" in self.features_df.columns:
                alert_count = self.features_df["alert_flag"].sum()
                f.write(
                    f"Alerts: {alert_count} ({alert_count/len(self.features_df)*100:.1f}%)\n"
                )

            f.write("\nActivity Distribution:\n")
            activity_counts = self.features_df["activity_clean"].value_counts()
            for activity, count in activity_counts.items():
                percentage = (count / len(self.features_df)) * 100
                f.write(f"  {activity:20}: {count:3d} ({percentage:5.1f}%)\n")

            f.write(f"\nDetection Performance\n")
            f.write("-" * 20 + "\n")

            key_features = [
                "magnitude_mean",
                "magnitude_std",
                "movement_score",
                "baseline_deviation_mean",
            ]
            f.write("Activity Stats:\n\n")

            for activity in activity_counts.index:
                activity_data = self.features_df[
                    self.features_df["activity_clean"] == activity
                ]
                f.write(
                    f"{activity.upper().replace('_', ' ')} (n={len(activity_data)}):\n"
                )

                for feature in key_features:
                    if feature in activity_data.columns:
                        mean_val = activity_data[feature].mean()
                        std_val = activity_data[feature].std()
                        f.write(f"  {feature:25}: {mean_val:7.4f} ± {std_val:7.4f}\n")
                f.write("\n")

            if self.gps_data:
                f.write(f"GPS Analysis\n")
                f.write("-" * 12 + "\n")
                gps_df = pd.DataFrame(self.gps_data)
                f.write(f"GPS samples: {len(gps_df)}\n")
                f.write(f"Locations: {len(gps_df[['lat', 'lon']].drop_duplicates())}\n")

                location_counts = (
                    gps_df.groupby(["lat", "lon"]).size().sort_values(ascending=False)
                )
                f.write("Top locations:\n")
                for (lat, lon), count in location_counts.head().items():
                    f.write(f"  ({lat:.6f}, {lon:.6f}): {count}\n")

                alert_locations = gps_df[gps_df["alert"] == True]
                if len(alert_locations) > 0:
                    f.write(f"\nAlert locations ({len(alert_locations)}):\n")
                    for _, row in alert_locations.iterrows():
                        f.write(
                            f"  {row['activity']} at ({row['lat']:.6f}, {row['lon']:.6f})\n"
                        )

            f.write(f"\nRecommendations\n")
            f.write("-" * 15 + "\n")

            still_samples = len(
                self.features_df[self.features_df["activity_clean"] == "still"]
            )
            total_samples = len(self.features_df)

            if still_samples / total_samples > 0.8:
                f.write("• Too much 'still' activity:\n")
                f.write("  - Adjust motion thresholds\n")
                f.write("  - Increase sampling rate\n")
                f.write("  - Calibrate sensitivity\n\n")

            if "alert_flag" in self.features_df.columns:
                alert_rate = self.features_df["alert_flag"].sum() / len(
                    self.features_df
                )
                if alert_rate > 0.1:
                    f.write("• High alert rate:\n")
                    f.write("  - Adjust fall threshold\n")
                    f.write("  - Add multi-stage validation\n")
                    f.write("  - Use context for alerts\n\n")
                elif alert_rate == 0:
                    f.write("• No alerts triggered:\n")
                    f.write("  - Test fall detection\n")
                    f.write("  - Lower thresholds\n")
                    f.write("  - Check alert system\n\n")

            f.write("• General improvements:\n")
            f.write("  - Better GPS signal\n")
            f.write("  - Longer collection windows\n")
            f.write("  - Ground truth labeling\n")

        print(f"Report saved: {report_file}")

    def run_complete_analysis(self):
        print("Starting analysis...")

        if not self.load_and_process_all_data():
            print("Failed to load data.")
            return False

        print("Analyzing detection...")
        self.analyze_detection_accuracy()

        print("Analyzing GPS...")
        self.analyze_gps_patterns()

        print("Creating visuals...")
        self.create_enhanced_visualizations()

        print("Generating report...")
        self.generate_comprehensive_report()

        print("\nAnalysis complete!")
        print("Results in: accel_logs/analysis/")
        print("- detection_performance.png")
        print("- feature_importance.png")
        print("- enhanced_time_series.png")
        print("- classification_confusion_matrix.png")
        print("- comprehensive_analysis_report.txt")

        if self.gps_data:
            print("- gps_activity_map.png")

        return True

    def compare_with_ground_truth(self, ground_truth_file=None):
        if ground_truth_file and os.path.exists(ground_truth_file):
            print(f"\nComparing with ground truth: {ground_truth_file}")

            gt_df = pd.read_csv(ground_truth_file)

            if "filename" in gt_df.columns and "filename" in self.features_df.columns:
                comparison_df = self.features_df.merge(
                    gt_df[["filename", "true_activity"]], on="filename", how="inner"
                )

                if len(comparison_df) > 0:
                    accuracy = (
                        comparison_df["activity_clean"]
                        == comparison_df["true_activity"]
                    ).mean()
                    print(f"Detection accuracy: {accuracy:.2%}")

                    from sklearn.metrics import classification_report, confusion_matrix

                    cm = confusion_matrix(
                        comparison_df["true_activity"], comparison_df["activity_clean"]
                    )
                    report = classification_report(
                        comparison_df["true_activity"], comparison_df["activity_clean"]
                    )

                    print(f"\n{report}")

                    os.makedirs("accel_logs/analysis", exist_ok=True)

                    plt.figure(figsize=(10, 8))
                    sns.heatmap(
                        cm,
                        annot=True,
                        fmt="d",
                        cmap="Blues",
                        xticklabels=sorted(comparison_df["activity_clean"].unique()),
                        yticklabels=sorted(comparison_df["true_activity"].unique()),
                    )
                    plt.title("Ground Truth vs Detection")
                    plt.ylabel("True")
                    plt.xlabel("Detected")
                    plt.tight_layout()
                    plt.savefig(
                        "accel_logs/analysis/ground_truth_comparison.png",
                        dpi=150,
                        bbox_inches="tight",
                    )
                    plt.close()

                    print("Ground truth comparison saved")
                else:
                    print("No matching files found")
            else:
                print("Ground truth file needs 'filename' and 'true_activity' columns")
        else:
            print("No ground truth file found")
            print(
                "Create CSV with 'filename' and 'true_activity' columns for comparison"
            )

    def export_features_for_ml(
        self, output_file="accel_logs/analysis/features_for_ml.csv"
    ):
        if self.features_df is None:
            print("No data available")
            return

        excluded_columns = [
            "filename",
            "filepath",
            "activity_clean",
            "detected_activity",
            "activity_label",
            "timestamp",
            "sequence_id",
            "collection_duration",
            "sequence_number",
        ]

        feature_columns = []
        for col in self.features_df.columns:
            if col not in excluded_columns:
                try:
                    pd.to_numeric(self.features_df[col], errors="raise")
                    feature_columns.append(col)
                except (ValueError, TypeError):
                    continue

        export_df = self.features_df[feature_columns + ["activity_clean"]].copy()
        export_df = export_df.fillna(0)

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        export_df.to_csv(output_file, index=False)

        print(f"Features exported: {output_file}")
        print(f"Shape: {export_df.shape}")
        print(f"Features: {len(feature_columns)}")
        print(f"Activities: {export_df['activity_clean'].unique()}")


def main():
    print("Accelerometer Data Analyzer")
    print("=" * 27)

    analyzer = EnhancedAccelerometerAnalyzer()

    if not os.path.exists(analyzer.data_dir):
        print(f"Data directory not found: {analyzer.data_dir}")
        print("Collect some data first.")
        return

    if analyzer.run_complete_analysis():
        print("\nExporting features...")
        analyzer.export_features_for_ml()

        ground_truth_file = "accel_logs/ground_truth.csv"
        if os.path.exists(ground_truth_file):
            analyzer.compare_with_ground_truth(ground_truth_file)
        else:
            analyzer.compare_with_ground_truth()

    else:
        print("Analysis failed. Check data files.")


def main():
    print("Enhanced Accelerometer Data Analyzer")
    analyzer = EnhancedAccelerometerAnalyzer()
    if not os.path.exists(analyzer.data_dir):
        print(f"Data directory not found: {analyzer.data_dir}")
        print("Please ensure you have collected some accelerometer data first.")
        return
    if analyzer.run_complete_analysis():
        print("\nExporting features for ML...")
        analyzer.export_features_for_ml()
        ground_truth_file = "accel_logs/ground_truth.csv"
        if os.path.exists(ground_truth_file):
            analyzer.compare_with_ground_truth(ground_truth_file)
        else:
            analyzer.compare_with_ground_truth()

    else:
        print("Analysis failed. Please check your data files.")


if __name__ == "__main__":
    main()