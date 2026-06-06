"""Process a Polar field-training session export.

The Polar extension works with wearable-derived heart-rate and optional
GNSS/session fields. It does not treat Polar Flow exports as raw IMU, ECG, or
accelerometer data unless those fields are explicitly present in future files.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "polar_config.yaml"
DATA_DIR = ROOT / "data" / "polar_sample"
OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = ROOT / "figures"
MPL_CONFIG_DIR = ROOT / ".mplconfig"
XDG_CACHE_DIR = ROOT / ".cache"
MPL_CONFIG_DIR.mkdir(exist_ok=True)
XDG_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt


DEFAULT_SAMPLE = DATA_DIR / "polar_session_sample.csv"
REQUIRED_CONFIG_KEYS = {
    "hr_max",
    "hr_rest",
    "hr_zone_method",
    "hr_zones",
    "rolling_window_seconds",
    "recovery_window_seconds",
    "speed_smoothing_seconds",
    "lag_correlation_max_seconds",
    "lag_correlation_step_seconds",
}


@dataclass
class SessionData:
    samples: pd.DataFrame
    metadata: dict[str, str]
    source_fields: list[str]


def parse_scalar(value: str) -> object:
    text = value.strip()
    if not text:
        return ""
    if text[0] == text[-1] and text.startswith(("'", '"')):
        return text[1:-1]
    if text.startswith("[") and text.endswith("]"):
        items = [item.strip() for item in text[1:-1].split(",") if item.strip()]
        return [parse_scalar(item) for item in items]
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def load_simple_yaml(path: Path) -> dict[str, object]:
    """Load the repository's small YAML config without adding a dependency."""

    if not path.exists():
        raise FileNotFoundError(f"Polar config not found: {path}")

    config: dict[str, object] = {}
    current_map: dict[str, object] | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip(" "))
            if ":" not in line:
                raise ValueError(f"Invalid YAML line {line_number}: {raw_line.rstrip()}")
            key, value = line.strip().split(":", 1)
            if indent == 0:
                if value.strip():
                    config[key] = parse_scalar(value)
                    current_map = None
                else:
                    config[key] = {}
                    current_map = config[key]  # type: ignore[assignment]
            elif indent == 2 and current_map is not None:
                current_map[key] = parse_scalar(value)
            else:
                raise ValueError(
                    f"Unsupported YAML nesting at line {line_number}: {raw_line.rstrip()}"
                )
    return config


def validate_config(config: dict[str, object]) -> dict[str, object]:
    missing = sorted(REQUIRED_CONFIG_KEYS - set(config))
    if missing:
        raise ValueError(f"Missing required Polar config keys: {', '.join(missing)}")

    method = str(config["hr_zone_method"])
    if method not in {"percent_hrmax", "percent_hrr"}:
        raise ValueError("hr_zone_method must be 'percent_hrmax' or 'percent_hrr'.")

    hr_zones = config["hr_zones"]
    if not isinstance(hr_zones, dict) or not hr_zones:
        raise ValueError("hr_zones must be a mapping of zone names to [low, high].")
    for name, bounds in hr_zones.items():
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError(f"HR zone {name} must be a [low, high] list.")
        low, high = float(bounds[0]), float(bounds[1])
        if low < 0 or high <= low:
            raise ValueError(f"Invalid bounds for HR zone {name}: {bounds}")
        hr_zones[name] = [low, high]

    numeric_positive = [
        "hr_max",
        "hr_rest",
        "rolling_window_seconds",
        "recovery_window_seconds",
        "speed_smoothing_seconds",
        "lag_correlation_max_seconds",
        "lag_correlation_step_seconds",
    ]
    for key in numeric_positive:
        value = float(config[key])
        if value <= 0:
            raise ValueError(f"{key} must be positive.")
        config[key] = int(value) if value.is_integer() else value

    if float(config["hr_max"]) <= float(config["hr_rest"]):
        raise ValueError("hr_max must be greater than hr_rest.")
    return config


def normalize_name(name: str) -> str:
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def first_matching_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    normalized = {normalize_name(col): col for col in columns}
    for candidate in candidates:
        key = normalize_name(candidate)
        if key in normalized:
            return normalized[key]
    return None


def parse_duration_to_seconds(value: object) -> float | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    parts = text.split(":")
    try:
        numbers = [float(part) for part in parts]
    except ValueError:
        return None
    if len(numbers) == 3:
        return numbers[0] * 3600 + numbers[1] * 60 + numbers[2]
    if len(numbers) == 2:
        return numbers[0] * 60 + numbers[1]
    if len(numbers) == 1:
        return numbers[0]
    return None


def find_polar_timeseries_header(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig") as handle:
        for index, line in enumerate(handle):
            cols = [col.strip() for col in line.rstrip("\n").split(",")]
            if "Time" in cols and any("HR" in col for col in cols):
                return index
            if "elapsed_seconds" in cols:
                return index
    raise ValueError(f"Could not find a Polar time-series header in {path}")


def read_polar_metadata(path: Path, header_row: int) -> dict[str, str]:
    if header_row < 2:
        return {}
    header = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns.tolist()
    values = pd.read_csv(path, skiprows=1, nrows=1, header=None, encoding="utf-8-sig")
    row = values.iloc[0].tolist()
    return {
        str(key): str(value)
        for key, value in zip(header, row)
        if pd.notna(value) and str(value).strip()
    }


def load_polar_csv(path: Path) -> SessionData:
    header_row = find_polar_timeseries_header(path)
    metadata = read_polar_metadata(path, header_row)
    samples = pd.read_csv(path, skiprows=header_row, encoding="utf-8-sig")
    samples = samples.dropna(axis=1, how="all")
    source_fields = samples.columns.tolist()

    time_col = first_matching_column(samples.columns, ["Time"])
    if "elapsed_seconds" in samples.columns:
        samples["elapsed_seconds"] = pd.to_numeric(
            samples["elapsed_seconds"], errors="coerce"
        )
    elif time_col is None:
        raise ValueError("Polar CSV time-series data must include a Time column.")
    else:
        samples["elapsed_seconds"] = samples[time_col].map(parse_duration_to_seconds)
    if samples["elapsed_seconds"].isna().all():
        raise ValueError("Could not parse elapsed time values from the Polar CSV.")

    date = metadata.get("Date")
    start = metadata.get("Start time")
    if date and start:
        start_time = pd.to_datetime(f"{date} {start}", errors="coerce")
        if pd.notna(start_time):
            samples["timestamp"] = start_time + pd.to_timedelta(
                samples["elapsed_seconds"], unit="s"
            )

    column_map = {
        "heart_rate_bpm": ["HR (bpm)", "Heart rate", "HeartRate", "HR"],
        "speed_kmh": ["Speed (km/h)", "Speed"],
        "distance_m": ["Distances (m)", "Distance (m)", "DistanceMeters"],
        "altitude_m": ["Altitude (m)", "Elevation (m)", "Altitude", "Elevation"],
    }
    for target, candidates in column_map.items():
        col = (
            target
            if target in samples.columns
            else first_matching_column(samples.columns, candidates)
        )
        if col is not None and col in samples.columns:
            samples[target] = pd.to_numeric(samples[col], errors="coerce")
    if "heart_rate_bpm" in samples:
        samples.loc[samples["heart_rate_bpm"] <= 0, "heart_rate_bpm"] = np.nan

    keep = ["elapsed_seconds", "timestamp"]
    keep.extend([col for col in column_map if col in samples.columns])
    samples = samples[[col for col in keep if col in samples.columns]].copy()
    return SessionData(samples=samples, metadata=metadata, source_fields=source_fields)


def haversine_m(
    lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series
) -> pd.Series:
    radius_m = 6_371_000
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    d_phi = np.radians(lat2 - lat1)
    d_lambda = np.radians(lon2 - lon1)
    a = (
        np.sin(d_phi / 2) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(d_lambda / 2) ** 2
    )
    return 2 * radius_m * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def load_gpx_derived(path: Path) -> pd.DataFrame:
    tree = ET.parse(path)
    root = tree.getroot()
    ns = {"gpx": "http://www.topografix.com/GPX/1/1"}
    rows = []
    for point in root.findall(".//gpx:trkpt", ns):
        ele = point.find("gpx:ele", ns)
        timestamp = point.find("gpx:time", ns)
        rows.append(
            {
                "timestamp": timestamp.text if timestamp is not None else None,
                "latitude": float(point.attrib["lat"]),
                "longitude": float(point.attrib["lon"]),
                "altitude_m": float(ele.text) if ele is not None and ele.text else np.nan,
            }
        )
    if not rows:
        raise ValueError(f"No GPX track points found in {path}")

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df["elapsed_seconds"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
    df["segment_distance_m"] = haversine_m(
        df["latitude"].shift(),
        df["longitude"].shift(),
        df["latitude"],
        df["longitude"],
    ).fillna(0)
    df["distance_m"] = df["segment_distance_m"].cumsum()
    dt_hours = df["timestamp"].diff().dt.total_seconds().div(3600)
    df["speed_kmh"] = df["segment_distance_m"].div(1000).div(dt_hours).replace(
        [np.inf, -np.inf], np.nan
    )
    return df[["elapsed_seconds", "timestamp", "distance_m", "speed_kmh", "altitude_m"]]


def derive_sample(session: SessionData, output_path: Path) -> None:
    sample_cols = [
        "elapsed_seconds",
        "heart_rate_bpm",
        "speed_kmh",
        "distance_m",
        "altitude_m",
    ]
    sample = session.samples[[col for col in sample_cols if col in session.samples]].copy()
    if "elapsed_seconds" in sample:
        sample["elapsed_seconds"] = sample["elapsed_seconds"].round().astype("Int64")
    for col in ["heart_rate_bpm", "speed_kmh", "distance_m", "altitude_m"]:
        if col in sample:
            sample[col] = sample[col].round(3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output_path, index=False)


def infer_hrmax(metadata: dict[str, str], configured_hrmax: float) -> float:
    for key in ["HR max", "Max heart rate", "Maximum heart rate"]:
        if key in metadata:
            value = pd.to_numeric(metadata[key], errors="coerce")
            if pd.notna(value) and value > 0:
                return float(value)
    return float(configured_hrmax)


def sample_interval_seconds(df: pd.DataFrame) -> float:
    deltas = df["elapsed_seconds"].diff().dropna()
    positive = deltas[deltas > 0]
    if positive.empty:
        return 1.0
    return float(positive.median())


def window_to_samples(df: pd.DataFrame, seconds: float) -> int:
    return max(int(round(seconds / sample_interval_seconds(df))), 1)


def sample_weights_seconds(df: pd.DataFrame) -> pd.Series:
    if "elapsed_seconds" not in df or len(df) == 0:
        return pd.Series(dtype=float)
    deltas = df["elapsed_seconds"].diff().shift(-1)
    median_delta = sample_interval_seconds(df)
    return deltas.fillna(median_delta).clip(
        lower=0, upper=max(float(median_delta) * 5, 1)
    )


def hr_zone_scale(hr: pd.Series, config: dict[str, object]) -> pd.Series:
    hr_max = float(config["hr_max"])
    hr_rest = float(config["hr_rest"])
    method = str(config["hr_zone_method"])
    if method == "percent_hrmax":
        return hr / hr_max
    if method == "percent_hrr":
        return (hr - hr_rest) / (hr_max - hr_rest)
    raise ValueError(f"Unsupported HR zone method: {method}")


def compute_zone_seconds(df: pd.DataFrame, config: dict[str, object]) -> dict[str, float]:
    if "heart_rate_bpm" not in df:
        return {}
    weights = sample_weights_seconds(df)
    scaled_hr = hr_zone_scale(df["heart_rate_bpm"], config)
    zones = {}
    for label, bounds in dict(config["hr_zones"]).items():
        lower, upper = float(bounds[0]), float(bounds[1])
        mask = (
            scaled_hr.ge(lower)
            & scaled_hr.lt(upper)
            & df["heart_rate_bpm"].notna()
        )
        zones[str(label)] = float(weights[mask].sum())
    return zones


def elevation_gain_loss(series: pd.Series) -> tuple[float | None, float | None]:
    clean = series.dropna()
    if clean.empty:
        return None, None
    deltas = clean.diff().dropna()
    gain = deltas[deltas > 0].sum()
    loss = -deltas[deltas < 0].sum()
    return float(gain), float(loss)


def pearson_or_none(x: pd.Series, y: pd.Series) -> float | None:
    paired = pd.concat([x, y], axis=1).dropna()
    if len(paired) < 3:
        return None
    return float(paired.iloc[:, 0].corr(paired.iloc[:, 1]))


def timestamp_or_none(row: pd.Series) -> str | None:
    if "timestamp" not in row or pd.isna(row["timestamp"]):
        return None
    value = row["timestamp"]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def compute_hr_recovery(
    df: pd.DataFrame, recovery_window_seconds: float
) -> tuple[dict[str, object], list[str]]:
    if "heart_rate_bpm" not in df or df["heart_rate_bpm"].dropna().empty:
        return {"hr_recovery_computed": False}, ["HR recovery skipped: HR is unavailable."]

    hr_df = df.dropna(subset=["heart_rate_bpm", "elapsed_seconds"]).copy()
    if hr_df.empty:
        return {"hr_recovery_computed": False}, ["HR recovery skipped: no valid HR samples."]

    peak_idx = hr_df["heart_rate_bpm"].idxmax()
    peak = hr_df.loc[peak_idx]
    target_elapsed = float(peak["elapsed_seconds"]) + float(recovery_window_seconds)
    later = hr_df[hr_df["elapsed_seconds"] >= target_elapsed]
    metrics: dict[str, object] = {
        "hr_recovery_computed": True,
        "peak_hr": float(peak["heart_rate_bpm"]),
        "peak_hr_elapsed_seconds": float(peak["elapsed_seconds"]),
        "peak_hr_timestamp": timestamp_or_none(peak),
        "recovery_window_seconds": float(recovery_window_seconds),
    }
    warnings: list[str] = []

    if later.empty:
        metrics.update(
            {
                "hr_after_recovery_window": None,
                "hr_after_recovery_elapsed_seconds": None,
                "hr_after_recovery_timestamp": None,
                "hr_recovery_bpm": None,
                "hr_recovery_slope_bpm_per_s": None,
                "hr_recovery_slope_sign_convention": (
                    "Positive values indicate HR decrease per second after peak HR."
                ),
            }
        )
        warnings.append(
            "HR recovery window unavailable after peak HR; recovery metrics set to null."
        )
        return metrics, warnings

    recovery = later.iloc[0]
    hr_recovery_bpm = float(peak["heart_rate_bpm"] - recovery["heart_rate_bpm"])
    metrics.update(
        {
            "hr_after_recovery_window": float(recovery["heart_rate_bpm"]),
            "hr_after_recovery_elapsed_seconds": float(recovery["elapsed_seconds"]),
            "hr_after_recovery_timestamp": timestamp_or_none(recovery),
            "hr_recovery_bpm": hr_recovery_bpm,
            "hr_recovery_slope_bpm_per_s": hr_recovery_bpm
            / float(recovery_window_seconds),
            "hr_recovery_slope_sign_convention": (
                "Positive values indicate HR decrease per second after peak HR."
            ),
        }
    )
    return metrics, warnings


def smooth_speed(df: pd.DataFrame, smoothing_seconds: float) -> pd.Series:
    window = window_to_samples(df, smoothing_seconds)
    return df["speed_kmh"].rolling(window=window, min_periods=1, center=True).mean()


def compute_lagged_correlations(
    df: pd.DataFrame, config: dict[str, object]
) -> tuple[pd.DataFrame, dict[str, object], list[str]]:
    if not {"heart_rate_bpm", "speed_kmh"}.issubset(df.columns):
        return (
            pd.DataFrame(),
            {
                "hr_speed_coupling_computed": False,
                "hr_speed_coupling_note": "Skipped because HR or speed data are unavailable.",
            },
            ["HR-speed coupling skipped: HR or speed data are unavailable."],
        )

    paired = df[["elapsed_seconds", "heart_rate_bpm", "speed_kmh"]].dropna().copy()
    paired["elapsed_seconds"] = paired["elapsed_seconds"].astype(float)
    if len(paired) < 3:
        return (
            pd.DataFrame(),
            {
                "hr_speed_coupling_computed": False,
                "hr_speed_coupling_note": "Skipped because fewer than 3 paired samples are available.",
            },
            ["HR-speed coupling skipped: fewer than 3 paired samples."],
        )

    paired["speed_kmh_smoothed"] = smooth_speed(
        paired, float(config["speed_smoothing_seconds"])
    )
    base_corr = pearson_or_none(paired["heart_rate_bpm"], paired["speed_kmh_smoothed"])
    max_lag = int(config["lag_correlation_max_seconds"])
    step = int(config["lag_correlation_step_seconds"])
    if step <= 0:
        raise ValueError("lag_correlation_step_seconds must be positive.")

    rows = []
    for lag in range(0, max_lag + 1, step):
        lagged = pd.merge_asof(
            paired[["elapsed_seconds", "heart_rate_bpm"]].sort_values("elapsed_seconds"),
            paired[["elapsed_seconds", "speed_kmh_smoothed"]]
            .assign(elapsed_seconds=lambda x: x["elapsed_seconds"] + lag)
            .sort_values("elapsed_seconds"),
            on="elapsed_seconds",
            direction="nearest",
            tolerance=max(step / 2, sample_interval_seconds(paired)),
        )
        corr = pearson_or_none(lagged["heart_rate_bpm"], lagged["speed_kmh_smoothed"])
        rows.append(
            {
                "lag_seconds": lag,
                "hr_speed_correlation": corr,
                "paired_sample_count": int(lagged.dropna().shape[0]),
            }
        )

    lag_df = pd.DataFrame(rows)
    valid = lag_df.dropna(subset=["hr_speed_correlation"])
    if valid.empty:
        best_lag = None
        best_corr = None
    else:
        best_row = valid.loc[valid["hr_speed_correlation"].abs().idxmax()]
        best_lag = int(best_row["lag_seconds"])
        best_corr = float(best_row["hr_speed_correlation"])

    metrics = {
        "hr_speed_coupling_computed": True,
        "hr_speed_correlation": base_corr,
        "speed_smoothing_seconds": float(config["speed_smoothing_seconds"]),
        "lag_correlation_max_seconds": int(config["lag_correlation_max_seconds"]),
        "lag_correlation_step_seconds": int(config["lag_correlation_step_seconds"]),
        "best_lag_seconds": best_lag,
        "best_lagged_hr_speed_correlation": best_corr,
        "hr_speed_coupling_note": (
            "Lagged correlation is an exploratory diagnostic and does not establish causality."
        ),
    }
    return lag_df, metrics, []


def compute_metrics(
    df: pd.DataFrame, metadata: dict[str, str], config: dict[str, object]
) -> tuple[dict[str, object], pd.DataFrame]:
    warnings: list[str] = []
    duration_s = float(df["elapsed_seconds"].max() - df["elapsed_seconds"].min())
    metrics: dict[str, object] = {
        "duration_seconds": duration_s,
        "duration_minutes": duration_s / 60,
        "sample_count": int(len(df)),
        "hrmax_bpm_used_for_zones": float(config["hr_max"]),
        "hr_rest_bpm_used_for_zones": float(config["hr_rest"]),
        "hr_zone_method": str(config["hr_zone_method"]),
        "missing_timestamp_count": int(df["elapsed_seconds"].isna().sum()),
        "available_fields": [col for col in df.columns if col != "timestamp"],
        "polar_export_note": (
            "Processed wearable-derived heart-rate and GNSS/session fields; "
            "no raw IMU, raw ECG, or raw accelerometer data is claimed."
        ),
    }

    if "heart_rate_bpm" in df and df["heart_rate_bpm"].notna().any():
        hr = df["heart_rate_bpm"].dropna()
        recovery_metrics, recovery_warnings = compute_hr_recovery(
            df, float(config["recovery_window_seconds"])
        )
        warnings.extend(recovery_warnings)
        metrics.update(
            {
                "mean_hr_bpm": float(hr.mean()),
                "max_hr_bpm": float(hr.max()),
                "min_hr_bpm": float(hr.min()),
                "std_hr_bpm": float(hr.std()),
                "missing_hr_count": int(df["heart_rate_bpm"].isna().sum()),
                "rolling_window_seconds": float(config["rolling_window_seconds"]),
                "hr_zone_seconds": compute_zone_seconds(df, config),
                **recovery_metrics,
            }
        )
    else:
        warnings.append("Heart-rate diagnostics skipped: HR data are unavailable.")

    if "distance_m" in df and df["distance_m"].notna().any():
        metrics["total_distance_m"] = float(
            df["distance_m"].max() - df["distance_m"].min()
        )
    elif "Total distance (km)" in metadata:
        distance_km = pd.to_numeric(metadata["Total distance (km)"], errors="coerce")
        if pd.notna(distance_km):
            metrics["total_distance_m"] = float(distance_km * 1000)

    if "speed_kmh" in df and df["speed_kmh"].notna().any():
        metrics["mean_speed_kmh"] = float(df["speed_kmh"].mean())
        metrics["max_speed_kmh"] = float(df["speed_kmh"].max())

    if "altitude_m" in df and df["altitude_m"].notna().any():
        gain, loss = elevation_gain_loss(df["altitude_m"])
        metrics["elevation_gain_m"] = gain
        metrics["elevation_loss_m"] = loss

    lag_df, coupling_metrics, coupling_warnings = compute_lagged_correlations(
        df, config
    )
    metrics.update(coupling_metrics)
    warnings.extend(coupling_warnings)
    metrics["warnings"] = warnings
    return metrics, lag_df


def make_summary(metrics: dict[str, object]) -> pd.DataFrame:
    flat = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                flat[f"{key}.{subkey}"] = subvalue
        elif isinstance(value, list):
            flat[key] = ";".join(str(item) for item in value)
        else:
            flat[key] = value
    return pd.DataFrame([flat])


def plot_hr_timeseries(df: pd.DataFrame, config: dict[str, object]) -> None:
    target = FIGURE_DIR / "polar_hr_timeseries.png"
    if "heart_rate_bpm" not in df or df["heart_rate_bpm"].dropna().empty:
        target.unlink(missing_ok=True)
        return
    window = window_to_samples(df, float(config["rolling_window_seconds"]))
    rolling = df["heart_rate_bpm"].rolling(window=window, min_periods=1).mean()
    plt.figure(figsize=(9, 4))
    plt.plot(
        df["elapsed_seconds"] / 60,
        df["heart_rate_bpm"],
        label="Heart rate",
        alpha=0.55,
    )
    plt.plot(
        df["elapsed_seconds"] / 60,
        rolling,
        label=f"{config['rolling_window_seconds']} s rolling mean",
        linewidth=2,
    )
    plt.xlabel("Time (min)")
    plt.ylabel("Heart rate (bpm)")
    plt.title("Polar Heart-Rate Time Series")
    plt.legend()
    plt.tight_layout()
    plt.savefig(target, dpi=180)
    plt.close()


def plot_training_zones(zone_seconds: dict[str, float]) -> None:
    target = FIGURE_DIR / "polar_training_zones.png"
    if not zone_seconds:
        target.unlink(missing_ok=True)
        return
    labels = list(zone_seconds)
    minutes = [zone_seconds[label] / 60 for label in labels]
    plt.figure(figsize=(8, 4))
    plt.bar(labels, minutes, color="tab:blue")
    plt.xlabel("HR zone")
    plt.ylabel("Time (min)")
    plt.title("Polar Heart-Rate Zone Distribution")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(target, dpi=180)
    plt.close()


def plot_speed_timeseries(df: pd.DataFrame) -> None:
    target = FIGURE_DIR / "polar_speed_timeseries.png"
    if "speed_kmh" not in df or df["speed_kmh"].dropna().empty:
        target.unlink(missing_ok=True)
        return
    plt.figure(figsize=(9, 4))
    plt.plot(df["elapsed_seconds"] / 60, df["speed_kmh"], linewidth=1.4)
    plt.xlabel("Time (min)")
    plt.ylabel("Speed (km/h)")
    plt.title("Polar Speed Time Series")
    plt.tight_layout()
    plt.savefig(target, dpi=180)
    plt.close()


def plot_elevation_profile(df: pd.DataFrame) -> None:
    target = FIGURE_DIR / "polar_elevation_profile.png"
    if "altitude_m" not in df or df["altitude_m"].dropna().empty:
        target.unlink(missing_ok=True)
        return
    x = df["distance_m"] / 1000 if "distance_m" in df else df["elapsed_seconds"] / 60
    xlabel = "Distance (km)" if "distance_m" in df else "Time (min)"
    plt.figure(figsize=(9, 4))
    plt.plot(x, df["altitude_m"], linewidth=1.5)
    plt.xlabel(xlabel)
    plt.ylabel("Elevation (m)")
    plt.title("Polar Elevation Profile")
    plt.tight_layout()
    plt.savefig(target, dpi=180)
    plt.close()


def plot_hr_speed_relationship(df: pd.DataFrame, config: dict[str, object]) -> None:
    target = FIGURE_DIR / "polar_hr_speed_relationship.png"
    if not {"heart_rate_bpm", "speed_kmh"}.issubset(df.columns):
        target.unlink(missing_ok=True)
        return
    paired = df[["heart_rate_bpm", "speed_kmh"]].dropna().copy()
    if paired.empty:
        target.unlink(missing_ok=True)
        return
    paired["speed_kmh_smoothed"] = smooth_speed(
        paired.assign(elapsed_seconds=df.loc[paired.index, "elapsed_seconds"]),
        float(config["speed_smoothing_seconds"]),
    )
    plt.figure(figsize=(6, 5))
    plt.scatter(paired["speed_kmh_smoothed"], paired["heart_rate_bpm"], s=10, alpha=0.35)
    plt.xlabel("Smoothed speed (km/h)")
    plt.ylabel("Heart rate (bpm)")
    plt.title("Polar HR-Speed Relationship")
    plt.tight_layout()
    plt.savefig(target, dpi=180)
    plt.close()


def plot_lag_correlation(lag_df: pd.DataFrame) -> None:
    target = FIGURE_DIR / "polar_hr_speed_lag_correlation.png"
    if lag_df.empty or lag_df["hr_speed_correlation"].dropna().empty:
        target.unlink(missing_ok=True)
        return
    plt.figure(figsize=(7, 4))
    plt.plot(
        lag_df["lag_seconds"],
        lag_df["hr_speed_correlation"],
        marker="o",
        linewidth=1.6,
    )
    plt.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    plt.xlabel("Speed lag (s)")
    plt.ylabel("Pearson r")
    plt.title("Exploratory HR-Speed Lag Correlation")
    plt.tight_layout()
    plt.savefig(target, dpi=180)
    plt.close()


def save_outputs(
    df: pd.DataFrame,
    metrics: dict[str, object],
    lag_df: pd.DataFrame,
    config: dict[str, object],
) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(exist_ok=True)
    make_summary(metrics).to_csv(OUTPUT_DIR / "polar_session_summary.csv", index=False)
    (OUTPUT_DIR / "polar_session_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "polar_config_used.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    lag_path = OUTPUT_DIR / "polar_hr_speed_lag_correlation.csv"
    if lag_df.empty:
        lag_path.unlink(missing_ok=True)
    else:
        lag_df.to_csv(lag_path, index=False)

    plot_hr_timeseries(df, config)
    plot_training_zones(metrics.get("hr_zone_seconds", {}))
    plot_speed_timeseries(df)
    plot_elevation_profile(df)
    plot_hr_speed_relationship(df, config)
    plot_lag_correlation(lag_df)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_SAMPLE,
        help="Polar training CSV export or sanitized sample CSV.",
    )
    parser.add_argument(
        "--input-gpx",
        type=Path,
        default=None,
        help=(
            "Optional Polar GPX route export. Coordinates are used only for "
            "derived metrics."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="YAML configuration for Polar diagnostics.",
    )
    parser.add_argument(
        "--write-sanitized-sample",
        type=Path,
        default=None,
        help="Write a coordinate-free sample CSV derived from the input CSV.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if not args.input_csv.exists() and args.input_gpx is None:
        raise FileNotFoundError(
            "No Polar sample found. Add data/polar_sample/polar_session_sample.csv "
            "or pass --input-csv /path/to/polar.csv"
        )

    config = validate_config(load_simple_yaml(args.config))
    session = load_polar_csv(args.input_csv) if args.input_csv.exists() else None
    if session is not None and args.write_sanitized_sample:
        derive_sample(session, args.write_sanitized_sample)

    if session is not None:
        df = session.samples
        metadata = session.metadata
        source_fields = session.source_fields
    else:
        df = pd.DataFrame()
        metadata = {}
        source_fields = []

    if args.input_gpx is not None:
        if not args.input_gpx.exists():
            raise FileNotFoundError(f"GPX file not found: {args.input_gpx}")
        gpx_df = load_gpx_derived(args.input_gpx)
        if df.empty:
            df = gpx_df
        else:
            for col in ["distance_m", "speed_kmh", "altitude_m"]:
                if col not in df or df[col].dropna().empty:
                    df[col] = pd.merge_asof(
                        df.sort_values("elapsed_seconds"),
                        gpx_df[["elapsed_seconds", col]]
                        .dropna()
                        .sort_values("elapsed_seconds"),
                        on="elapsed_seconds",
                        direction="nearest",
                        tolerance=2,
                    )[col]
        source_fields.extend(
            ["GPX timestamp", "GPX latitude", "GPX longitude", "GPX elevation"]
        )

    if df.empty:
        raise ValueError("No Polar samples were loaded.")

    config["hr_max"] = infer_hrmax(metadata, float(config["hr_max"]))
    metrics, lag_df = compute_metrics(df, metadata, config)
    metrics["source_fields_detected"] = source_fields
    save_outputs(df, metrics, lag_df, config)

    print(f"Wrote summary: {OUTPUT_DIR / 'polar_session_summary.csv'}")
    print(f"Wrote metrics: {OUTPUT_DIR / 'polar_session_metrics.json'}")
    print(f"Wrote config audit: {OUTPUT_DIR / 'polar_config_used.json'}")
    print(f"Wrote Polar figures to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
