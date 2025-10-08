# Sensor-Data-Processing-and-Analysis-Pipeline
# Accelerometer Activity Detection & Analysis

A real-time accelerometer data analysis system that detects and classifies human activities, including fall detection, using machine learning techniques. This project processes time-series accelerometer data to identify activity patterns, generate alerts, and provide insights into movement behavior.

## Overview

This system ingests accelerometer data (typically from mobile devices or wearable sensors) and performs sophisticated signal processing to classify activities into categories like still, light activity, moderate activity, vigorous activity, and fall detection. The pipeline includes data collection, feature engineering, machine learning classification, and comprehensive reporting.

### Key Features

- **Real-time Activity Classification**: Identifies six distinct activity states using Random Forest models
- **Fall Detection**: Specialized detection for sudden impact events with GPS location tracking
- **Feature Engineering**: Extracts magnitude, standard deviation, movement scores, and baseline deviations
- **GPS Integration**: Associates location data with detected activities and alerts
- **Comprehensive Analytics**: Generates detailed reports with statistical summaries and performance metrics

## Project Requirements

This project is designed for data scientists and researchers working with time-series sensor data, particularly in the domains of human activity recognition, fall detection systems, or health monitoring applications.

### Prerequisites

- Python 3.7 or higher
- Basic understanding of time-series analysis
- Familiarity with machine learning concepts
- Knowledge of signal processing fundamentals

## Dependencies

The project relies on the following Python libraries:

```bash
numpy>=1.19.0
pandas>=1.2.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
seaborn>=0.11.0
```

Install all dependencies using:

```bash
pip install -r requirements.txt
```

## Getting Started

### Data Format

The system expects accelerometer data in CSV format with the following structure:

```
timestamp,x,y,z,latitude,longitude
2025-09-29 02:15:30,0.002,-0.012,1.003,-37.851,145.116
2025-09-29 02:15:31,-0.001,0.008,0.998,-37.851,145.116
```

**Required columns:**
- `timestamp`: ISO format datetime string
- `x`, `y`, `z`: Accelerometer readings in g-force units
- `latitude`, `longitude`: GPS coordinates (optional but recommended for fall alerts)

### Initial Setup

1. Place your accelerometer data files in the `data/` directory
2. Configure analysis parameters in `src/config.py` if needed
3. Train the classification model or use the pre-trained model

## How to Run the Application

### Basic Analysis Pipeline

Run the complete analysis pipeline on your data:

```bash
python src/main.py --input data/accelerometer_data.csv
```

This will:
1. Load and validate your accelerometer data
2. Engineer features from the raw signals
3. Classify activities using the trained model
4. Generate comprehensive reports in the `analysis/` directory

### Training a New Model

If you have labeled training data, train a new classification model:

```bash
python src/train_model.py --training-data data/labeled_data.csv --output models/
```

The training script will:
- Extract features from labeled accelerometer sequences
- Train a Random Forest classifier with cross-validation
- Save the model and feature scaler for inference
- Generate a classification report with performance metrics

### Generating Reports

Create detailed analysis reports from processed data:

```bash
python src/generate_report.py --data analysis/processed_data.csv
```

## Architecture & Workflow

### Data Processing Pipeline

The system follows a multi-stage pipeline:

1. **Data Ingestion**: Raw accelerometer readings are loaded and validated
2. **Feature Extraction**: Time-domain features are computed over sliding windows
3. **Classification**: Machine learning model predicts activity type
4. **Alert Generation**: Fall events trigger immediate alerts with GPS data
5. **Report Generation**: Statistical summaries and visualizations are created

### Feature Engineering

The system computes several key features for activity classification:

**Magnitude Calculation**: The overall acceleration magnitude is computed as the Euclidean norm of the three axes, providing a rotation-invariant measure of movement intensity.

```python
magnitude = np.sqrt(x**2 + y**2 + z**2)
```

**Movement Score**: A composite metric combining magnitude standard deviation and mean baseline deviation, normalized to quantify activity intensity on a consistent scale.

```python
movement_score = (magnitude_std * 0.6) + (baseline_deviation_mean * 0.4)
```

**Baseline Deviation**: The absolute difference between observed magnitude and the expected resting magnitude (typically 1.0 g), indicating deviation from stillness.

## Code Examples

### Loading and Processing Accelerometer Data

Process raw sensor data to extract meaningful features:

```python
data = pd.read_csv('data/accelerometer_data.csv')
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Calculate magnitude
data['magnitude'] = np.sqrt(
    data['x']**2 + data['y']**2 + data['z']**2
)

# Compute rolling statistics
window_size = 10
data['magnitude_std'] = data['magnitude'].rolling(
    window=window_size, 
    min_periods=1
).std()

data['magnitude_mean'] = data['magnitude'].rolling(
    window=window_size,
    min_periods=1
).mean()
```

### Activity Classification

Classify activities using the trained model:

```python
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load pre-trained model
model = joblib.load('models/activity_classifier.pkl')
scaler = joblib.load('models/feature_scaler.pkl')

# Prepare features
features = ['magnitude_mean', 'magnitude_std', 
            'movement_score', 'baseline_deviation_mean']
X = data[features]
X_scaled = scaler.transform(X)

# Predict activities
predictions = model.predict(X_scaled)
data['activity'] = predictions
```

### Fall Detection Logic

Identify potential fall events based on signal characteristics:

```python
# Define thresholds
FALL_MAGNITUDE_THRESHOLD = 2.5  # g-force
FALL_STD_THRESHOLD = 0.8
POST_FALL_STILLNESS = 0.05

# Detect falls
fall_mask = (
    (data['magnitude'] > FALL_MAGNITUDE_THRESHOLD) &
    (data['magnitude_std'] > FALL_STD_THRESHOLD)
)

# Verify with post-impact stillness
for idx in data[fall_mask].index:
    if idx + 5 < len(data):
        post_impact = data.loc[idx+1:idx+5, 'magnitude_std'].mean()
        if post_impact < POST_FALL_STILLNESS:
            data.loc[idx, 'fall_detected'] = True
```

### GPS Alert Integration

Generate location-based alerts for critical events:

```python
alerts = data[data['activity'] == 'fall_detected'].copy()

if not alerts.empty:
    for idx, row in alerts.iterrows():
        alert_info = {
            'timestamp': row['timestamp'],
            'location': (row['latitude'], row['longitude']),
            'magnitude': row['magnitude'],
            'activity': row['activity']
        }
        print(f"ALERT: Fall detected at {alert_info['location']}")
        # Trigger notification system here
```

## Understanding the Results

### Classification Report

The system achieves excellent performance on the test dataset, with the Random Forest model showing perfect classification accuracy (1.0000) for the "still" activity class. This high performance is typical when there's clear separation between activity states in the feature space.

### Activity Distribution

From the sample analysis report, the data shows:
- **Still (77.1%)**: Dominant class representing stationary periods
- **Vigorous Activity (8.3%)**: High-intensity movements
- **Moderate Activity (6.2%)**: Medium-intensity movements
- **Fall Detected (4.2%)**: Critical events requiring immediate attention
- **Light Activity (2.1%)**: Low-intensity movements
- **Uncertain (2.1%)**: Ambiguous patterns requiring review

### Statistical Insights

Each activity class shows distinct statistical signatures:

- **Still**: Low magnitude variation (std ≈ 0.0015) and minimal movement scores
- **Vigorous Activity**: High magnitude variation (std ≈ 0.41) and elevated movement scores
- **Fall Detection**: Moderate magnitude variation with characteristic impact patterns

These patterns enable reliable classification even with relatively simple features.

## Customization & Extension

### Adjusting Detection Thresholds

Modify thresholds in `src/config.py` to tune sensitivity:

```python
ACTIVITY_THRESHOLDS = {
    'still': {'movement_score': 0.02},
    'light': {'movement_score': 0.15},
    'moderate': {'movement_score': 0.50},
    'vigorous': {'movement_score': 1.0}
}
```

### Adding New Features

Extend the feature set for improved classification:

```python
# Frequency domain features
from scipy.fft import fft

def extract_frequency_features(signal, sampling_rate=100):
    fft_vals = np.abs(fft(signal))
    dominant_freq = np.argmax(fft_vals[:len(fft_vals)//2])
    return {
        'dominant_frequency': dominant_freq * sampling_rate / len(signal),
        'spectral_energy': np.sum(fft_vals**2)
    }
```

## Troubleshooting

**Low classification accuracy**: Ensure your data has sufficient variety across activity classes and consider collecting more labeled training samples.

**GPS data missing**: The system functions without GPS but fall alerts won't include location information. Verify that your data source includes latitude and longitude columns.

**High false positive rate for falls**: Adjust the `FALL_MAGNITUDE_THRESHOLD` and `FALL_STD_THRESHOLD` parameters to reduce sensitivity, or increase the post-impact stillness verification window.

## Future Enhancements

Consider these extensions to improve the system:

- **Deep Learning Models**: Implement LSTM or CNN architectures for temporal pattern recognition
- **Multi-sensor Fusion**: Integrate gyroscope and magnetometer data for richer context
- **Online Learning**: Enable model updates with new labeled data without full retraining
- **Real-time Visualization**: Add live dashboard for monitoring activity streams
- **Personalized Thresholds**: Adapt detection parameters based on individual user baselines

## Contributing

This project welcomes contributions from the data science and health monitoring communities. Whether you're improving the classification algorithms, adding new features, or enhancing the documentation, your input helps make activity detection more accurate and accessible.

Focus areas for contribution include optimizing feature engineering pipelines, implementing additional machine learning models, and expanding the test coverage with diverse accelerometer datasets.

---

*Ready to analyze movement patterns in your sensor data? Start by exploring the sample analysis reports in the `analysis/` directory to see what insights await in your own datasets.*
