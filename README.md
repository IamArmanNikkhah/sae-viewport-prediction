# Spatiotemporal Attentive Entropy (SAE) for Viewport Prediction

This repository contains the official implementation for the 6-week undergraduate research project focused on the paper: **"Spatiotemporal Attentive Entropy: A Geometry-Correct Cross-Entropy on S² for Calibrated, Low-Latency Viewport Prediction."**

## Core Objectives

Our goal is to implement the core components of the SAE paper, including:
1.  A data processing pipeline for 360° head-tracking data.
2.  A numerically stable, streaming estimator for the SAE metric.
3.  An LSTM-based prediction model that uses SAE as a feature.
4.  An evaluation suite to validate our results against the paper's findings.

---

## Initial Setup
1. Create a folder called **“SAE Viewport Prediction Folder”** on your desktop.
2. Open a terminal and navigate into that folder using the cd command on terminal.

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/IamArmanNikkhah/sae-viewport-prediction.git
cd sae-viewport-prediction
git checkout -b feature/week-5-training-pipeline origin/feature/week-5-training-pipeline
git branch
```
> This ensures you’re on the branch containing all code and files needed to run the full pipeline.

---

### 2. Create a Virtual Environment

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```
> If activation fails, ensure you don’t have PowerShell execution restrictions enabled.

---

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Download the Dataset

1. Visit [Zenodo Dataset Page](https://zenodo.org/records/10650505)
2. Download **Image_H.7z** manually.  
   *(Note: This file may take ~10+ minutes to download depending on your connection.)*
3. While downloading, create a folder named `data` inside the `sae-viewport-prediction` folder.
4. Extract the contents of **Image_H.7z** using 7-Zip or any unzip tool.
5. Move the extracted folder into the `data` folder.

Your directory structure should look like this:
```
SAE Viewport Prediction Folder/
└── sae-viewport-prediction/
    ├── src
    ├── data
    │   └── H                # Extracted folder from Image_H.7z
    │     └── SalMaps
    │   └── Scanpaths        # Move this folder out of H and into data
    ├── tests
    ├── scripts
    └── .vscode
```

---

## Run the Code (Train the Model)

**Mac/Linux/Windows:**
```bash
python scripts/process_all_scanpaths.py
python scripts/make_npy_from_cleaned.py
python scripts/generate_vmf_lut.py
```

After running the preprocessing steps, your directory should look like this:
```
SAE Viewport Prediction Folder/
└── sae-viewport-prediction/
    ├── src
    ├── data
    │   └── H
    │     └── SalMaps
    │   └── Processed
    │   └── Scanpaths
    ├── tests
    ├── scripts
    └── .vscode
```
```bash
python -m src.training.train
```

> The last command starts the **training process** using the dataset.
---

## Evaluate the Model

To evaluate the trained model and compute metrics (MAE and NLL):

**Mac/Linux/Windows:**
```bash
python -m src.evaluation.evaluate
```

> This will print the **Mean Angular Error (MAE)** and **Negative Log-Likelihood (NLL)** metrics for each time horizon.

---

## Summary

You’ve now successfully set up, trained, and evaluated the **SAE Viewport Prediction** model.  
This project demonstrates how to predict 360° video viewing directions while also estimating confidence in real time.



## Directory Structure

The repository is organized to keep code, data, and experiments separate and clean.
- **/data:** Holds the raw and processed datasets. (Note: Large data files will not be committed to Git).
- **/src:** All source code for the project, organized into modules.
- **/notebooks:** Jupyter notebooks for data exploration, visualization, and experiments.
- **/docs:** Project planning documents, like our project guide.

---
