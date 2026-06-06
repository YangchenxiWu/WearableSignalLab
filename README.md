# WearableSignalLab

This mini-project demonstrates basic IMU time-series processing and baseline
human activity classification. It is intended as a technical bridge project for
MSc applications in Human Technology, Sports Technology, and Human Movement
Analytics, not as a full research contribution.

## Why It Matters

Wearable-based movement monitoring often starts with noisy accelerometer and
gyroscope signals. A credible first step is to show that raw sensor windows can
be loaded, filtered, summarized with movement features, and evaluated with a
simple classifier. This repository keeps that workflow small and reproducible.

## Dataset

The first version uses the UCI Human Activity Recognition Using Smartphones
Dataset. The dataset contains accelerometer and gyroscope inertial signals from
smartphone sensors, with predefined train/test splits and six activity labels.

Source:
https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

Citation:

Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz.
"A Public Domain Dataset for Human Activity Recognition Using Smartphones."
ESANN 2013.

Large raw data files are not committed. Run the preparation script to download
and extract the dataset into `data/UCI HAR Dataset/`.

## Pipeline

1. Download and prepare UCI HAR data.
2. Load accelerometer and gyroscope inertial signal windows.
3. Apply a Butterworth low-pass filter to a raw accelerometer segment.
4. Compute accelerometer and gyroscope magnitude signals.
5. Extract time-domain and frequency-domain features.
6. Train a baseline Random Forest classifier.
7. Save metrics, reports, feature tables, and diagnostic figures.

## Reproduce

Create an environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the full pipeline:

```bash
python3 scripts/01_prepare_data.py
python3 scripts/02_filter_and_features.py
python3 scripts/03_train_baseline.py
python3 scripts/04_make_figures.py
```

Expected outputs:

```text
outputs/features.csv
outputs/classification_report.txt
outputs/metrics.json
figures/raw_vs_filtered_signal.png
figures/signal_magnitude_or_peak_detection.png
figures/feature_distribution_by_activity.png
figures/confusion_matrix.png
```

## Example Outputs

The repository generates:

- Raw vs filtered accelerometer signal figure.
- Accelerometer magnitude and peak detection figure.
- Feature distribution by activity figure.
- Confusion matrix for the baseline classifier.
- CSV, JSON, and text outputs under `outputs/`.

## Visual Summary

![Raw vs filtered signal](figures/raw_vs_filtered_signal.png)

![Confusion matrix](figures/confusion_matrix.png)

## Results Summary

In the current first version, the Random Forest baseline produced:

- Accuracy: 0.9009
- Macro F1-score: 0.8980
- Train windows: 7,352
- Test windows: 2,947
- Extracted features: 74

These values are saved in `outputs/metrics.json` and should be treated as
reproducible baseline results, not as a claim of model novelty.

## Limitations

- This is a baseline educational pipeline, not a publication-grade model.
- UCI HAR is already segmented into fixed windows, so this first version does
  not reconstruct a full continuous raw recording.
- The classifier is intentionally simple and does not use deep learning.
- Sensor placement, participant variability, and dataset collection conditions
  limit generalization to real sports settings.

## Future Direction

This project can be extended toward sports technology and human movement
analytics by adding raw continuous wearable datasets, sport-specific movement
labels, subject-wise validation, richer frequency features, and more careful
error analysis.
