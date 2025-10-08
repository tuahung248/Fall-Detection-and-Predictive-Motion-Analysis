import os
import time
import pandas as pd
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import glob
from datetime import datetime


class AccelerometerFileHandler(FileSystemEventHandler):
    def __init__(self, data_dir, ground_truth_file):
        self.data_dir = data_dir
        self.ground_truth_file = ground_truth_file

    def on_created(self, event):
        if event.is_directory:
            return

        if event.src_path.endswith(".csv"):
            self.add_to_ground_truth(event.src_path)

    def extract_activity_from_filename(self, filepath):
        filename = os.path.basename(filepath)
        parts = filename.replace(".csv", "").split("_")

        if len(parts) >= 3:
            activity = "_".join(parts[2:])
            return activity
        return "unknown"

    def add_to_ground_truth(self, filepath):
        filename = os.path.basename(filepath)
        activity = self.extract_activity_from_filename(filepath)
        time.sleep(1)

        try:
            if os.path.exists(self.ground_truth_file):
                gt_df = pd.read_csv(self.ground_truth_file)
            else:
                gt_df = pd.DataFrame(columns=["filename", "true_activity"])
            if filename not in gt_df["filename"].values:
                new_row = pd.DataFrame(
                    {"filename": [filename], "true_activity": [activity]}
                )
                gt_df = pd.concat([gt_df, new_row], ignore_index=True)
                gt_df.to_csv(self.ground_truth_file, index=False)

                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] Added to ground truth: {filename} -> {activity}"
                )

        except Exception as e:
            print(f"Error adding {filename} to ground truth: {e}")


class AccelerometerPipeline:
    def __init__(
        self,
        data_dir="accel_logs/data_files",
        ground_truth_file="accel_logs/ground_truth.csv",
    ):
        self.data_dir = data_dir
        self.ground_truth_file = ground_truth_file
        self.observer = None

    def initialize_ground_truth(self):
        """Initialize ground truth file with existing CSV files"""
        if not os.path.exists(self.data_dir):
            print(f"Data directory {self.data_dir} doesn't exist")
            return

        csv_files = glob.glob(os.path.join(self.data_dir, "*.csv"))

        if os.path.exists(self.ground_truth_file):
            gt_df = pd.read_csv(self.ground_truth_file)
            existing_files = set(gt_df["filename"].values)
        else:
            gt_df = pd.DataFrame(columns=["filename", "true_activity"])
            existing_files = set()

        new_entries = []
        for csv_file in csv_files:
            filename = os.path.basename(csv_file)
            if filename not in existing_files:
                parts = filename.replace(".csv", "").split("_")
                if len(parts) >= 3:
                    activity = "_".join(parts[2:])
                else:
                    activity = "unknown"

                new_entries.append({"filename": filename, "true_activity": activity})

        if new_entries:
            new_df = pd.DataFrame(new_entries)
            gt_df = pd.concat([gt_df, new_df], ignore_index=True)
            gt_df.to_csv(self.ground_truth_file, index=False)
            print(f"Initialized ground truth with {len(new_entries)} files")
        else:
            print("Ground truth file is up to date")

    def start_monitoring(self):
        """Start monitoring for new files"""
        print(f"Starting pipeline monitoring...")
        print(f"Data directory: {self.data_dir}")
        print(f"Ground truth file: {self.ground_truth_file}")
        print("Press Ctrl+C to stop")
        self.initialize_ground_truth()
        event_handler = AccelerometerFileHandler(self.data_dir, self.ground_truth_file)
        self.observer = Observer()
        self.observer.schedule(event_handler, self.data_dir, recursive=False)

        try:
            self.observer.start()
            print(f"\n Pipeline is running! Watching for new CSV files...")

            while True:
                time.sleep(1)

        except KeyboardInterrupt:
            print("\n Stopping pipeline...")

        finally:
            if self.observer:
                self.observer.stop()
                self.observer.join()
            print("Pipeline stopped")

    def run_analysis(self):
        print("\n🔄 Running analysis...")

        try:
            from data_analyzer import EnhancedAccelerometerAnalyzer

            analyzer = EnhancedAccelerometerAnalyzer(self.data_dir)
            if analyzer.run_complete_analysis():
                print("Analysis completed successfully")

                # Export features
                analyzer.export_features_for_ml()

                # Compare with ground truth
                if os.path.exists(self.ground_truth_file):
                    analyzer.compare_with_ground_truth(self.ground_truth_file)

                return True
            else:
                print("Analysis failed")
                return False

        except Exception as e:
            print(f"Error running analysis: {e}")
            return False

    def manual_sync(self):
        print("Manual sync - updating ground truth with all existing files...")
        self.initialize_ground_truth()
        print("Manual sync completed")


def main():
    print("Accelerometer Data Pipeline")
    print("=" * 27)

    pipeline = AccelerometerPipeline()

    print("\nOptions:")
    print("1. Start monitoring (auto-add new files to ground truth)")
    print("2. Run analysis once")
    print("3. Manual sync (update ground truth with existing files)")
    print("4. Start monitoring + auto-analysis")

    choice = input("\nSelect option (1-4): ").strip()

    if choice == "1":
        pipeline.start_monitoring()

    elif choice == "2":
        pipeline.run_analysis()

    elif choice == "3":
        pipeline.manual_sync()

    elif choice == "4":
        print("Starting monitoring with auto-analysis...")
        pipeline.initialize_ground_truth()

        event_handler = AccelerometerFileHandler(
            pipeline.data_dir, pipeline.ground_truth_file
        )
        observer = Observer()
        observer.schedule(event_handler, pipeline.data_dir, recursive=False)
        observer.start()

        print("Pipeline running with auto-analysis!")
        print("New files will be added to ground truth automatically")

        try:
            last_file_count = 0

            while True:
                csv_files = glob.glob(os.path.join(pipeline.data_dir, "*.csv"))
                current_count = len(csv_files)
                if current_count > last_file_count:
                    print(f"\n Detected {current_count - last_file_count} new files")
                    time.sleep(2) 
                    pipeline.run_analysis()
                    last_file_count = current_count

                time.sleep(10)

        except KeyboardInterrupt:
            print("\n Stopping pipeline...")

        finally:
            observer.stop()
            observer.join()
            print("Pipeline stopped")

    else:
        print("Invalid option")


if __name__ == "__main__":
    main()
