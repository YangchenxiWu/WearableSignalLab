# Polar Sample Data

This directory contains a coordinate-free sample derived from a self-collected
Polar field-training session export.

Raw Polar exports may contain identifiable location data. Users should
anonymize or avoid committing raw files.

Expected private input files may include:

- Polar training CSV
- Polar training TCX
- Polar training FIT
- Polar route GPX

The current processing script reads Polar CSV directly and can optionally parse
GPX route files for derived distance, speed, and elevation diagnostics. TCX/FIT
support is not required for the committed sample.

Polar exports may include fields such as:

- Session metadata, including sport type, date, duration, distance, calories,
  HRmax, HRrest, and other device-derived fields.
- Time-series heart rate.
- GNSS-derived speed, distance, route coordinates, and elevation, depending on
  the export type.
- Device/session fields such as cadence, pace, temperature, or power when
  available.

The committed sample file, `polar_session_sample.csv`, keeps only derived
session time-series fields:

- `elapsed_seconds`
- `heart_rate_bpm`
- `speed_kmh`
- `distance_m`

The sample does not include GPS coordinates, route geometry, raw IMU, raw ECG,
or raw accelerometer data. Speed and distance are treated as wearable/GNSS-
derived session diagnostics from the Polar export.

Privacy guidance:

- Do not commit raw GPX, TCX, FIT, or CSV files if they contain identifiable
  route coordinates, names, addresses, or private training locations.
- Prefer committing coordinate-free derived samples such as
  `polar_session_sample.csv`.
- If route information is needed for private analysis, keep it local and share
  only derived summaries, metrics JSON, config records, and figures that do not
  expose coordinates.
- Root-level private Polar exports are ignored by this repository's
  `.gitignore`.

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

The configured field-session diagnostics are controlled by:

```text
config/polar_config.yaml
```

Derived outputs can be shared even when the raw Polar export remains private,
provided the outputs do not expose identifiable location data.
