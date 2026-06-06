"""Filter IMU signals and extract baseline movement features."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, periodogram, sosfiltfilt


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "data" / "UCI HAR Dataset"
OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = ROOT / "figures"
FS = 50.0


def load_signal(split: str, signal_name: str) -> np.ndarray:
    path = DATASET_DIR / split / "Inertial Signals" / f"{signal_name}_{split}.txt"
    return np.loadtxt(path)


def load_labels() -> dict[int, str]:
    labels = {}
    with (DATASET_DIR / "activity_labels.txt").open("r", encoding="utf-8") as f:
        for line in f:
            idx, name = line.split()
            labels[int(idx)] = name
    return labels


def lowpass_filter(signal: np.ndarray, cutoff_hz: float = 5.0) -> np.ndarray:
    sos = butter(4, cutoff_hz, btype="lowpass", fs=FS, output="sos")
    return sosfiltfilt(sos, signal)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def dominant_frequency(x: np.ndarray) -> float:
    freqs, power = periodogram(x, fs=FS)
    if len(freqs) <= 1:
        return 0.0
    idx = int(np.argmax(power[1:]) + 1)
    return float(freqs[idx])


def spectral_energy(x: np.ndarray) -> float:
    _, power = periodogram(x, fs=FS)
    return float(np.sum(power))


def summarize(prefix: str, x: np.ndarray) -> dict[str, float]:
    peaks, _ = find_peaks(x)
    return {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_std": float(np.std(x)),
        f"{prefix}_min": float(np.min(x)),
        f"{prefix}_max": float(np.max(x)),
        f"{prefix}_range": float(np.max(x) - np.min(x)),
        f"{prefix}_rms": rms(x),
        f"{prefix}_peak_count": int(len(peaks)),
        f"{prefix}_dominant_freq": dominant_frequency(x),
        f"{prefix}_spectral_energy": spectral_energy(x),
    }


def load_split(split: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    signals = {
        "acc_x": load_signal(split, "total_acc_x"),
        "acc_y": load_signal(split, "total_acc_y"),
        "acc_z": load_signal(split, "total_acc_z"),
        "gyro_x": load_signal(split, "body_gyro_x"),
        "gyro_y": load_signal(split, "body_gyro_y"),
        "gyro_z": load_signal(split, "body_gyro_z"),
    }
    y = np.loadtxt(DATASET_DIR / split / f"y_{split}.txt", dtype=int)
    return signals, y


def extract_split_features(split: str, labels: dict[int, str]) -> pd.DataFrame:
    signals, y = load_split(split)
    rows = []
    n_windows = len(y)

    for i in range(n_windows):
        acc = np.vstack([signals["acc_x"][i], signals["acc_y"][i], signals["acc_z"][i]])
        gyro = np.vstack([signals["gyro_x"][i], signals["gyro_y"][i], signals["gyro_z"][i]])
        acc_mag = np.sqrt(np.sum(acc**2, axis=0))
        gyro_mag = np.sqrt(np.sum(gyro**2, axis=0))

        row: dict[str, object] = {
            "split": split,
            "window_id": i,
            "activity_id": int(y[i]),
            "activity": labels[int(y[i])],
        }

        for name in ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]:
            row.update(summarize(name, signals[name][i]))

        row.update(summarize("acc_mag", acc_mag))
        row.update(summarize("gyro_mag", gyro_mag))
        row["acc_sma"] = float(np.mean(np.abs(acc[0]) + np.abs(acc[1]) + np.abs(acc[2])))
        row["gyro_sma"] = float(np.mean(np.abs(gyro[0]) + np.abs(gyro[1]) + np.abs(gyro[2])))
        rows.append(row)

    return pd.DataFrame(rows)


def save_demo_signal(labels: dict[int, str]) -> None:
    signals, y = load_split("train")
    raw = signals["acc_x"][0]
    filtered = lowpass_filter(raw)
    acc_mag = np.sqrt(
        signals["acc_x"][0] ** 2 + signals["acc_y"][0] ** 2 + signals["acc_z"][0] ** 2
    )
    peaks, _ = find_peaks(acc_mag)

    demo = pd.DataFrame(
        {
            "sample_index": np.arange(len(raw)),
            "time_seconds": np.arange(len(raw)) / FS,
            "acc_x_raw": raw,
            "acc_x_filtered": filtered,
            "acc_magnitude": acc_mag,
            "is_peak": np.isin(np.arange(len(raw)), peaks).astype(int),
            "activity": labels[int(y[0])],
        }
    )
    demo.to_csv(OUTPUT_DIR / "demo_signal.csv", index=False)


def main() -> None:
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            "UCI HAR dataset not found. Run: python3 scripts/01_prepare_data.py"
        )

    OUTPUT_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(exist_ok=True)
    labels = load_labels()

    train = extract_split_features("train", labels)
    test = extract_split_features("test", labels)
    features = pd.concat([train, test], ignore_index=True)
    features.to_csv(OUTPUT_DIR / "features.csv", index=False)
    save_demo_signal(labels)

    summary = {
        "n_rows": int(len(features)),
        "n_features": int(len(features.columns) - 4),
        "splits": features["split"].value_counts().to_dict(),
        "activities": sorted(features["activity"].unique().tolist()),
    }
    (OUTPUT_DIR / "feature_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Wrote features: {OUTPUT_DIR / 'features.csv'}")


if __name__ == "__main__":
    main()
