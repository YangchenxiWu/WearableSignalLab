# Polar Sample Data

This directory contains a coordinate-free sample derived from a self-collected
Polar field-training session export.

Raw Polar exports may contain identifiable location data. Users should
anonymize or avoid committing raw files.

The committed sample file, `polar_session_sample.csv`, keeps only derived
session time-series fields:

- `elapsed_seconds`
- `heart_rate_bpm`
- `speed_kmh`
- `distance_m`

The sample does not include GPS coordinates, route geometry, raw IMU, raw ECG,
or raw accelerometer data. Speed and distance are treated as wearable/GNSS-
derived session diagnostics from the Polar export.

To process a private local export without committing it:

```bash
python3 scripts/05_process_polar_session.py --input-csv path/to/polar_export.CSV
```

To create a coordinate-free sample from a private Polar CSV:

```bash
python3 scripts/05_process_polar_session.py \
  --input-csv path/to/polar_export.CSV \
  --write-sanitized-sample data/polar_sample/polar_session_sample.csv
```
