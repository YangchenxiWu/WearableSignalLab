# Data

This project uses the UCI Human Activity Recognition Using Smartphones Dataset.

Download source:
https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

The scripts can download the dataset automatically:

```bash
python3 scripts/01_prepare_data.py
```

After extraction, the expected local folder is:

```text
data/UCI HAR Dataset/
```

Large raw dataset files are not committed to this repository. The preparation
script creates a small `data/sample_window.csv` file for quick inspection after
the dataset is available.

Dataset citation:

Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz.
"A Public Domain Dataset for Human Activity Recognition Using Smartphones."
21st European Symposium on Artificial Neural Networks, Computational Intelligence
and Machine Learning, ESANN 2013.
