"""Download and lightly prepare the UCI HAR dataset."""

from __future__ import annotations

import csv
import shutil
import ssl
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATASET_DIR = DATA_DIR / "UCI HAR Dataset"
ZIP_PATH = DATA_DIR / "UCI_HAR_Dataset.zip"
UCI_URL = (
    "https://archive.ics.uci.edu/static/public/240/"
    "human+activity+recognition+using+smartphones.zip"
)


def download_dataset() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    if DATASET_DIR.exists():
        print(f"Dataset already exists: {DATASET_DIR}")
        return

    if not ZIP_PATH.exists():
        print(f"Downloading UCI HAR dataset to {ZIP_PATH}")
        try:
            urllib.request.urlretrieve(UCI_URL, ZIP_PATH)
        except ssl.SSLCertVerificationError:
            print("SSL certificate verification failed; retrying with an unverified context.")
            context = ssl._create_unverified_context()
            with urllib.request.urlopen(UCI_URL, context=context) as response:
                with ZIP_PATH.open("wb") as f:
                    shutil.copyfileobj(response, f)
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, ssl.SSLCertVerificationError):
                print("SSL certificate verification failed; retrying with an unverified context.")
                context = ssl._create_unverified_context()
                with urllib.request.urlopen(UCI_URL, context=context) as response:
                    with ZIP_PATH.open("wb") as f:
                        shutil.copyfileobj(response, f)
            else:
                raise

    print(f"Extracting {ZIP_PATH}")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(DATA_DIR)

    if not DATASET_DIR.exists():
        nested_zip = DATA_DIR / "UCI HAR Dataset.zip"
        if nested_zip.exists():
            with zipfile.ZipFile(nested_zip, "r") as zf:
                zf.extractall(DATA_DIR)

    if not DATASET_DIR.exists():
        raise FileNotFoundError("Could not find extracted 'UCI HAR Dataset' folder.")


def read_first_signal_row(path: Path) -> list[float]:
    with path.open("r", encoding="utf-8") as f:
        return [float(x) for x in f.readline().split()]


def write_sample_window() -> None:
    inertial = DATASET_DIR / "train" / "Inertial Signals"
    axes = {
        "total_acc_x": inertial / "total_acc_x_train.txt",
        "total_acc_y": inertial / "total_acc_y_train.txt",
        "total_acc_z": inertial / "total_acc_z_train.txt",
        "body_gyro_x": inertial / "body_gyro_x_train.txt",
        "body_gyro_y": inertial / "body_gyro_y_train.txt",
        "body_gyro_z": inertial / "body_gyro_z_train.txt",
    }
    signals = {name: read_first_signal_row(path) for name, path in axes.items()}

    sample_path = DATA_DIR / "sample_window.csv"
    with sample_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_index", *signals.keys()])
        for i in range(len(next(iter(signals.values())))):
            writer.writerow([i, *[signals[name][i] for name in signals]])

    print(f"Wrote sample window: {sample_path}")


def main() -> None:
    download_dataset()
    write_sample_window()


if __name__ == "__main__":
    main()
