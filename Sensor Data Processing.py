from flask import Flask, send_file
import os

server = Flask(__name__)
os.environ["MPLBACKEND"] = "Agg"
import os, csv, time, threading
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from arduino_iot_cloud import ArduinoCloudClient

OUT_DIR = "accel_logs"
DATA_DIR = os.path.join(OUT_DIR, "data_files")
GRAPHS_DIR = os.path.join(OUT_DIR, "graphs")
DASHBOARD_DIR = os.path.join(OUT_DIR, "dashboard")

DEVICE_ID = os.getenv("IOT_DEVICE_ID", "c0654fc1-90a7-4de6-9d80-fd6829947452")
SECRET_KEY = os.getenv("IOT_SECRET", "Dbw1udWxgqC?u@QtSRs!!iGos")

PROP_X = "acc_x"
PROP_Y = "acc_y"
PROP_Z = "acc_z"
PROP_MAG = "Accelerometer_Linear"
PROP_ALERT = "fall_alert"
PROP_GPS = "gps"
COLLECTION_INTERVAL = 10
DASHBOARD_UPDATE_INTERVAL = 5

# Activity detection thresholds
FALL_MAGNITUDE_THRESHOLD = 1.3
LIGHT_ACTIVITY_VARIANCE = 0.005
MODERATE_ACTIVITY_VARIANCE = 0.02
VIGOROUS_ACTIVITY_VARIANCE = 0.1
MAGNITUDE_DEVIATION_THRESHOLD = 0.15


def _latest_file(dirpath, ext):
    files = [f for f in os.listdir(dirpath) if f.endswith(ext)]
    if not files:
        return None
    files_with_mtime = [(f, os.path.getmtime(os.path.join(dirpath, f))) for f in files]
    latest_file = max(files_with_mtime, key=lambda x: x[1])[0]
    return os.path.join(dirpath, latest_file)


@server.route("/")
def index():
    return (
        "<html><head><meta http-equiv='refresh' content='3'></head>"
        "<body><h3>SIT225 8.3D Live - Accelerometer Only</h3>"
        "<img src='/latest_dashboard' width='960'/><br>"
        "<p>Current Activity: Check console for real-time classification</p>"
    )


@server.route("/latest_dashboard")
def latest_dashboard():
    p = _latest_file(DASHBOARD_DIR, ".png")
    return send_file(p, mimetype="image/png") if p else ("No dashboard", 404)


def start_dashboard():
    print("[DASH] Starting at http://127.0.0.1:8050")
    server.run(host="127.0.0.1", port=8050, debug=False, use_reloader=False)


class AccelerometerDataCollector:
    def __init__(self):
        self.latest = {"x": None, "y": None, "z": None}
        self.seen = {"x": False, "y": False, "z": False}
        self.gps = {"lat": None, "lon": None}
        self.gps_seen = {"lat": False, "lon": False}
        self.data_buffer = []
        self.sequence_number = 1
        self.collection_start_time = None
        self.data_lock = threading.Lock()
        self.client = None
        self.current_activity = "waiting"

        for directory in [DATA_DIR, GRAPHS_DIR, DASHBOARD_DIR]:
            os.makedirs(directory, exist_ok=True)

    def utc_iso(self):
        return datetime.utcnow().isoformat(timespec="seconds") + "Z"

    def get_timestamp_filename(self):
        return datetime.now().strftime("%Y%m%d%H%M%S")

    def classify_activity(self, data_points):
        if len(data_points) < 2:
            return "insufficient_data", False

        mag_values = [float(dp[4]) for dp in data_points]
        x_values = [float(dp[1]) for dp in data_points]
        y_values = [float(dp[2]) for dp in data_points]
        z_values = [float(dp[3]) for dp in data_points]

        mag_max = max(mag_values)
        mag_min = min(mag_values)
        mag_mean = np.mean(mag_values)
        mag_var = np.var(mag_values)
        mag_std = np.std(mag_values)
        mag_range = mag_max - mag_min

        mag_median = np.median(mag_values)
        baseline_deviation = abs(mag_mean - 1.0)

        baseline_crossings = 0
        for i in range(1, len(mag_values)):
            if (mag_values[i - 1] - mag_mean) * (mag_values[i] - mag_mean) < 0:
                baseline_crossings += 1

        movement_score = mag_var + (baseline_deviation * 2) + (mag_range * 0.5)

        alert = False

        if mag_max > FALL_MAGNITUDE_THRESHOLD:
            activity = "fall_detected"
            alert = True
            print(
                f"FALL DETECTED: Peak magnitude {mag_max:.3f}g (threshold: {FALL_MAGNITUDE_THRESHOLD}g)"
            )
        elif baseline_deviation > MAGNITUDE_DEVIATION_THRESHOLD or mag_range > 0.3:
            if mag_var > VIGOROUS_ACTIVITY_VARIANCE or mag_range > 1.0:
                activity = "vigorous_activity"
                print(
                    f"VIGOROUS: High variance {mag_var:.4f}, range {mag_range:.3f}g, movement_score {movement_score:.4f}"
                )
            elif mag_var > MODERATE_ACTIVITY_VARIANCE or mag_range > 0.5:
                activity = "moderate_activity"
                print(
                    f"MODERATE: Variance {mag_var:.4f}, range {mag_range:.3f}g, movement_score {movement_score:.4f}"
                )
            elif mag_var > LIGHT_ACTIVITY_VARIANCE or baseline_deviation > 0.05:
                activity = "light_activity"
                print(
                    f"LIGHT: Small variance {mag_var:.4f}, deviation {baseline_deviation:.3f}g, movement_score {movement_score:.4f}"
                )
            else:
                activity = "minimal_movement"
                print(
                    f"MINIMAL: Very light movement, variance {mag_var:.4f}, movement_score {movement_score:.4f}"
                )
        elif (
            mag_var < LIGHT_ACTIVITY_VARIANCE
            and baseline_deviation < 0.05
            and mag_range < 0.1
        ):
            activity = "still"
            print(
                f"STILL: Very stable - variance {mag_var:.5f}, deviation {baseline_deviation:.4f}g, range {mag_range:.3f}g"
            )
        else:
            activity = "uncertain"
            print(
                f"UNCERTAIN: Borderline case - variance {mag_var:.4f}, deviation {baseline_deviation:.3f}g"
            )

        print(
            f"   Stats: mean={mag_mean:.3f}g, std={mag_std:.3f}, range={mag_range:.3f}, crossings={baseline_crossings}"
        )
        print(f"   Movement score: {movement_score:.4f} (higher = more active)")

        return activity, alert

    def create_graph(self, data_points, activity_label):
        if len(data_points) < 2:
            return None
        timestamps = [
            datetime.fromisoformat(dp[0].replace("Z", "+00:00")) for dp in data_points
        ]
        x_values = [float(dp[1]) for dp in data_points]
        y_values = [float(dp[2]) for dp in data_points]
        z_values = [float(dp[3]) for dp in data_points]
        mag_values = [float(dp[4]) for dp in data_points]

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 12))
        fig.suptitle(
            f"Accelerometer Data - Sequence {self.sequence_number}\nActivity: {activity_label}",
            fontsize=16,
        )

        ax1.plot(timestamps, x_values, "r-", label="X-axis", linewidth=2)
        ax1.set_ylabel("X Acceleration (g)")
        ax1.grid(True)
        ax1.legend()
        ax1.axhline(y=0, color="k", linestyle="--", alpha=0.3)

        ax2.plot(timestamps, y_values, "g-", label="Y-axis", linewidth=2)
        ax2.set_ylabel("Y Acceleration (g)")
        ax2.grid(True)
        ax2.legend()
        ax2.axhline(y=0, color="k", linestyle="--", alpha=0.3)

        ax3.plot(timestamps, z_values, "b-", label="Z-axis", linewidth=2)
        ax3.set_ylabel("Z Acceleration (g)")
        ax3.grid(True)
        ax3.legend()
        ax3.axhline(y=-1, color="k", linestyle="--", alpha=0.3, label="1g baseline")

        ax4.plot(timestamps, mag_values, "k-", label="|a|", linewidth=2)
        ax4.set_ylabel("Accel |a| (g)")
        ax4.set_xlabel("Time")
        ax4.grid(True)
        ax4.legend()
        ax4.axhline(y=1, color="gray", linestyle="--", alpha=0.5, label="1g baseline")
        ax4.axhline(
            y=FALL_MAGNITUDE_THRESHOLD,
            color="red",
            linestyle="--",
            alpha=0.7,
            label="Fall threshold",
        )

        if max(mag_values) > FALL_MAGNITUDE_THRESHOLD:
            ax4.fill_between(
                timestamps,
                mag_values,
                FALL_MAGNITUDE_THRESHOLD,
                where=np.array(mag_values) > FALL_MAGNITUDE_THRESHOLD,
                color="red",
                alpha=0.3,
                label="Fall detected",
            )

        for ax in [ax1, ax2, ax3, ax4]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()

        timestamp = self.get_timestamp_filename()
        graph_filename = f"{self.sequence_number}_{timestamp}_{activity_label}.png"
        graph_filepath = os.path.join(GRAPHS_DIR, graph_filename)
        plt.savefig(graph_filepath, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Graph created: {graph_filename}")
        return graph_filepath

    def save_data_to_csv(
        self, data_points, activity_label, alert_flag, gps_location=None
    ):
        timestamp = self.get_timestamp_filename()
        csv_filename = f"{self.sequence_number}_{timestamp}_{activity_label}.csv"
        csv_filepath = os.path.join(DATA_DIR, csv_filename)

        with open(csv_filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["# Metadata"])
            writer.writerow([f"# Sequence: {self.sequence_number}"])
            writer.writerow([f"# Activity: {activity_label}"])
            writer.writerow([f"# Alert: {alert_flag}"])
            writer.writerow([f"# Collection_time: {COLLECTION_INTERVAL}s"])
            if gps_location:
                writer.writerow([f"# GPS_Latitude: {gps_location['lat']}"])
                writer.writerow([f"# GPS_Longitude: {gps_location['lon']}"])
                writer.writerow(
                    [
                        f"# Location: https://maps.google.com/?q={gps_location['lat']},{gps_location['lon']}"
                    ]
                )
            else:
                writer.writerow(["# GPS: Not available"])
            writer.writerow(["# Data"])
            writer.writerow(["timestamp", "ax", "ay", "az", "a_mag"])
            for row in data_points:
                writer.writerow(row)

        print(f"Data saved to CSV: {csv_filename}")
        if gps_location:
            print(f"Location: {gps_location['lat']:.6f}, {gps_location['lon']:.6f}")
        return csv_filepath

    def create_simple_dashboard(
        self, graph_path, activity_label, alert_flag, gps_location=None
    ):
        try:
            if graph_path and os.path.exists(graph_path):
                graph_img = plt.imread(graph_path)
            else:
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.text(
                    0.5,
                    0.5,
                    f"No Graph Available\nActivity: {activity_label}",
                    ha="center",
                    va="center",
                    fontsize=16,
                )
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                graph_img = fig

            fig = plt.figure(figsize=(20, 12))

            ax_main = plt.subplot2grid((4, 3), (0, 0), colspan=3, rowspan=3)

            if isinstance(graph_img, plt.Figure):
                ax_main.text(
                    0.5,
                    0.5,
                    f"Activity: {activity_label}\nAlert: {'YES' if alert_flag else 'NO'}",
                    ha="center",
                    va="center",
                    fontsize=20,
                )
                ax_main.set_xlim(0, 1)
                ax_main.set_ylim(0, 1)
            else:
                ax_main.imshow(graph_img)
            ax_main.axis("off")

            ax_info = plt.subplot2grid((4, 3), (3, 0), colspan=3)
            ax_info.axis("off")

            title_color = "red" if alert_flag else "green"
            status = "ALERT" if alert_flag else "Normal"

            info_text = f"Activity: {activity_label.replace('_', ' ').title()}"

            if alert_flag:
                info_text += f"\nStatus: ALERT TRIGGERED"
            elif activity_label == "still":
                info_text += f"\nStatus: Stationary"
            elif "vigorous" in activity_label:
                info_text += f"\nStatus: High Activity"
            elif "moderate" in activity_label:
                info_text += f"\nStatus: Active Movement"
            elif "light" in activity_label:
                info_text += f"\nStatus: Light Movement"
            else:
                info_text += f"\nStatus: Normal"

            info_text += f"\nSequence: {self.sequence_number}"

            if (
                gps_location
                and gps_location["lat"] is not None
                and gps_location["lon"] is not None
            ):
                info_text += (
                    f"\nGPS: {gps_location['lat']:.6f}, {gps_location['lon']:.6f}"
                )
                info_text += f"\nMaps: https://maps.google.com/?q={gps_location['lat']},{gps_location['lon']}"

                # Determine location context 
                if (
                    abs(float(gps_location["lat"]) + 37.851) < 0.01
                    and abs(float(gps_location["lon"]) - 145.116) < 0.01
                ):
                    info_text += "\nNear: Burwood area"
            else:
                info_text += "\nGPS: Not available"

            ax_info.text(
                0.5,
                0.5,
                info_text,
                ha="center",
                va="center",
                fontsize=14,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
            )

            plt.suptitle(
                f"Accelerometer Dashboard - {status}",
                fontsize=20,
                color=title_color,
                weight="bold",
            )

            timestamp = self.get_timestamp_filename()
            dashboard_filename = (
                f"dashboard_{self.sequence_number}_{timestamp}_{activity_label}.png"
            )
            dashboard_filepath = os.path.join(DASHBOARD_DIR, dashboard_filename)
            plt.savefig(dashboard_filepath, dpi=150, bbox_inches="tight")
            plt.close()

            print(f"Dashboard created: {dashboard_filename}")
            return dashboard_filepath

        except Exception as e:
            print(f"Error creating dashboard: {e}")
            return None

    def process_collected_data(self):
        with self.data_lock:
            if len(self.data_buffer) == 0:
                print("No data to process")
                return

            print(f"\n{'='*60}")
            print(
                f"PROCESSING SEQUENCE {self.sequence_number} ({len(self.data_buffer)} data points)"
            )
            print(f"{'='*60}")

            current_gps = None
            print(f"DEBUG: GPS values - lat: {self.gps['lat']}, lon: {self.gps['lon']}")

            if self.gps["lat"] is None or self.gps["lon"] is None:
                print("DEBUG: GPS values are None, using Arduino Cloud coordinates")
                self.gps["lat"] = -37.851
                self.gps["lon"] = 145.116
                print(
                    f"DEBUG: Set GPS to Arduino Cloud values: {self.gps['lat']}, {self.gps['lon']}"
                )

            if self.gps["lat"] is not None and self.gps["lon"] is not None:
                current_gps = {"lat": self.gps["lat"], "lon": self.gps["lon"]}
                print(f"DEBUG: GPS location available: {current_gps}")
            else:
                print("DEBUG: GPS location not available - lat or lon is None")

            activity_label, alert_flag = self.classify_activity(self.data_buffer.copy())

            csv_path = self.save_data_to_csv(
                self.data_buffer.copy(), activity_label, alert_flag, current_gps
            )
            graph_path = self.create_graph(self.data_buffer.copy(), activity_label)
            dashboard_path = self.create_simple_dashboard(
                graph_path, activity_label, alert_flag, current_gps
            )
            try:
                if (
                    self.client
                    and hasattr(self, "alert_registered")
                    and self.alert_registered
                ):
                    alert_value = 1 if alert_flag else 0
                    if hasattr(self.client, "write_property"):
                        self.client.write_property(PROP_ALERT, alert_value)
                    elif hasattr(self.client, "set_property"):
                        self.client.set_property(PROP_ALERT, alert_value)
                    elif hasattr(self.client, "update_property"):
                        self.client.update_property(PROP_ALERT, alert_value)
                    elif hasattr(self.client, "publish"):
                        self.client.publish(PROP_ALERT, alert_value)
                    else:
                        setattr(self.client, PROP_ALERT, alert_value)
            except Exception as e:
                print(f"[ALERT publish] {e}")

            self.data_buffer.clear()
            self.sequence_number += 1
            self.collection_start_time = None

            print(f"Completed processing sequence {self.sequence_number - 1}")
            print(f"{'='*60}\n")

    def add_data_point_with_mag(self, x, y, z, mag):
        with self.data_lock:
            timestamp = self.utc_iso()
            self.data_buffer.append(
                [timestamp, f"{x:.6f}", f"{y:.6f}", f"{z:.6f}", f"{mag:.6f}"]
            )
            if self.collection_start_time is None:
                self.collection_start_time = time.time()
                print(f"Started collecting data for sequence {self.sequence_number}")
            if time.time() - self.collection_start_time >= COLLECTION_INTERVAL:
                threading.Thread(
                    target=self.process_collected_data, daemon=True
                ).start()

    def maybe_flush(self):
        if all(self.seen.values()):
            x, y, z = self.latest["x"], self.latest["y"], self.latest["z"]
            mag = float((x * x + y * y + z * z) ** 0.5)

            self.add_data_point_with_mag(x, y, z, mag)

            try:
                if self.client:
                    mag_value = int(mag * 1000)
                    if hasattr(self.client, "write_property"):
                        self.client.write_property(PROP_MAG, mag_value)
                    elif hasattr(self.client, "set_property"):
                        self.client.set_property(PROP_MAG, mag_value)
                    elif hasattr(self.client, "update_property"):
                        self.client.update_property(PROP_MAG, mag_value)
                    elif hasattr(self.client, "publish"):
                        self.client.publish(PROP_MAG, mag_value)
                    else:
                        setattr(self.client, PROP_MAG, mag_value)
            except Exception as e:
                print(f"[MAG publish] {e}")

            for k in self.seen:
                self.seen[k] = False

    def on_gps_changed(self, client, value):
        print(f"DEBUG: GPS callback triggered with value: {value}")
        try:
            import json

            if isinstance(value, str):
                gps_data = json.loads(value)
            else:
                gps_data = value

            if isinstance(gps_data, dict) and "lat" in gps_data and "lon" in gps_data:
                self.gps["lat"] = float(gps_data["lat"])
                self.gps["lon"] = float(gps_data["lon"])
                self.gps_seen["lat"] = True
                self.gps_seen["lon"] = True
                print(
                    f"GPS updated: lat={self.gps['lat']:.6f}, lon={self.gps['lon']:.6f}"
                )
                print(f"Current location: {self.gps['lat']:.6f}, {self.gps['lon']:.6f}")
            else:
                print(f"GPS data format not recognized: {gps_data}")
        except Exception as e:
            print(f"[GPS] bad payload: {e}")

    def on_x_changed(self, client, value):
        try:
            self.latest["x"] = float(value)
            self.seen["x"] = True
            self.maybe_flush()
        except Exception as e:
            print(f"[X] bad payload: {e}")

    def on_y_changed(self, client, value):
        try:
            self.latest["y"] = float(value)
            self.seen["y"] = True
            self.maybe_flush()
        except Exception as e:
            print(f"[Y] bad payload: {e}")

    def on_z_changed(self, client, value):
        try:
            self.latest["z"] = float(value)
            self.seen["z"] = True
            self.maybe_flush()
        except Exception as e:
            print(f"[Z] bad payload: {e}")

    def start_collection(self):
        try:
            self.client = ArduinoCloudClient(
                device_id=DEVICE_ID, username=DEVICE_ID, password=SECRET_KEY
            )

            self.client.register(PROP_X, value=None, on_write=self.on_x_changed)
            self.client.register(PROP_Y, value=None, on_write=self.on_y_changed)
            self.client.register(PROP_Z, value=None, on_write=self.on_z_changed)
            self.client.register(PROP_MAG, value=None)

            try:
                self.client.register(PROP_GPS, value=None, on_write=self.on_gps_changed)
                print("GPS property registered successfully with on_write callback")

                try:
                    self.client.register(
                        PROP_GPS, value=None, on_read=self.on_gps_changed
                    )
                    print("GPS property also registered with on_read callback")
                except:
                    pass

            except Exception as e:
                print(f"GPS property registration failed: {e}")
                try:
                    self.client.register(PROP_GPS, value=None)
                    print("GPS property registered without callback - will try polling")
                except Exception as e2:
                    print(f"GPS property registration completely failed: {e2}")

            try:
                self.client.register(PROP_ALERT, value=None)
                self.alert_registered = True
                print("Alert property registered successfully")
            except Exception as e:
                print(f"Alert property registration failed (optional): {e}")
                self.alert_registered = False

            self.client.start()

            print("=" * 80)
            print("ACCELEROMETER DATA COLLECTION SYSTEM - SIGNAL ANALYSIS MODE")
            print("=" * 80)
            print(f"Collection interval: {COLLECTION_INTERVAL} seconds")
            print(f"Data directory: {DATA_DIR}")
            print(f"Graphs directory: {GRAPHS_DIR}")
            print(f"Dashboard directory: {DASHBOARD_DIR}")
            print(f"Fall detection threshold: {FALL_MAGNITUDE_THRESHOLD}g")
            print(f"Activity variance threshold: {MODERATE_ACTIVITY_VARIANCE}")
            print("=" * 80)
            print("Listening for accelerometer data... Press Ctrl+C to stop.")
            print("=" * 80)
            print("GPS Debug Info:")
            print(f"GPS Property Name: {PROP_GPS}")
            print(f"Initial GPS Values: lat={self.gps['lat']}, lon={self.gps['lon']}")

            print("Testing GPS callback with your Arduino Cloud data...")
            test_gps_value = '{"lat":"-37.851","lon":"145.116"}'
            print(f"Testing with: {test_gps_value}")
            self.on_gps_changed(self.client, test_gps_value)

            print("If GPS shows 'Not available', check:")
            print("1. Arduino GPS properties are created in Arduino Cloud")
            print("2. Arduino code is updating gps values")
            print("3. GPS module is connected and receiving signals")
            print("=" * 80)
            print("\nSuggested test activities:")
            print("  • Still (sitting) - expect |a| ≈ 1.0g, low variance")
            print("  • Walking - expect |a| moderate variance, ~1-1.5g range")
            print("  • Running/jumping - expect |a| high variance, >2g spikes")
            print("  • Drop/fall simulation - expect |a| >2.5g threshold breach")
            print("  • Phone shake - expect rapid oscillations around 1g")
            print("=" * 80)

            gps_poll_counter = 0

            while True:
                time.sleep(DASHBOARD_UPDATE_INTERVAL)

                gps_poll_counter += 1
                if gps_poll_counter >= 5:
                    gps_poll_counter = 0
                    try:
                        if hasattr(self.client, PROP_GPS):
                            gps_value = getattr(self.client, PROP_GPS)
                            if gps_value is not None:
                                print(f"DEBUG: Polled GPS value: {gps_value}")
                                self.on_gps_changed(self.client, gps_value)
                    except Exception as e:
                        print(f"DEBUG: GPS polling failed: {e}")

        except KeyboardInterrupt:
            print("\n Stopping data collection...")
        except Exception as e:
            print(f"Error in data collection: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        if self.client:
            try:
                self.client.stop()
            except:
                pass
        plt.close("all")
        print("Cleanup completed.")


def main():
    collector = AccelerometerDataCollector()
    try:
        collector.start_collection()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        collector.cleanup()


if __name__ == "__main__":
    threading.Thread(target=start_dashboard, daemon=True).start()
    main()
