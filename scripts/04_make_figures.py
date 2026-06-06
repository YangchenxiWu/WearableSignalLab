"""Create diagnostic figures for the wearable signal pipeline."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
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
import seaborn as sns


def save_raw_vs_filtered() -> None:
    demo = pd.read_csv(OUTPUT_DIR / "demo_signal.csv")
    plt.figure(figsize=(9, 4))
    plt.plot(demo["time_seconds"], demo["acc_x_raw"], label="Raw acc x", linewidth=1.2)
    plt.plot(
        demo["time_seconds"],
        demo["acc_x_filtered"],
        label="Butterworth low-pass",
        linewidth=1.8,
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration")
    plt.title("Raw vs Filtered Accelerometer Signal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "raw_vs_filtered_signal.png", dpi=180)
    plt.close()


def save_magnitude_peaks() -> None:
    demo = pd.read_csv(OUTPUT_DIR / "demo_signal.csv")
    peaks = demo[demo["is_peak"] == 1]
    plt.figure(figsize=(9, 4))
    plt.plot(
        demo["time_seconds"],
        demo["acc_magnitude"],
        label="Acceleration magnitude",
        linewidth=1.5,
    )
    plt.scatter(
        peaks["time_seconds"],
        peaks["acc_magnitude"],
        s=18,
        color="tab:red",
        label="Detected peaks",
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Magnitude")
    plt.title("Signal Magnitude and Peak Detection")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "signal_magnitude_or_peak_detection.png", dpi=180)
    plt.close()


def save_feature_distribution() -> None:
    features = pd.read_csv(OUTPUT_DIR / "features.csv")
    plt.figure(figsize=(10, 5))
    sns.boxplot(
        data=features,
        x="activity",
        y="acc_mag_rms",
        hue="activity",
        palette="Set2",
        legend=False,
    )
    plt.xlabel("Activity")
    plt.ylabel("Acceleration magnitude RMS")
    plt.title("Feature Distribution by Activity")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "feature_distribution_by_activity.png", dpi=180)
    plt.close()


def save_confusion_matrix() -> None:
    cm = pd.read_csv(OUTPUT_DIR / "confusion_matrix.csv", index_col=0)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "confusion_matrix.png", dpi=180)
    plt.close()


def main() -> None:
    FIGURE_DIR.mkdir(exist_ok=True)
    save_raw_vs_filtered()
    save_magnitude_peaks()
    save_feature_distribution()
    save_confusion_matrix()
    print(f"Wrote figures to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
