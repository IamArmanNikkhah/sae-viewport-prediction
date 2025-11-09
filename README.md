# Spatiotemporal Attentive Entropy (SAE) for Viewport Prediction

This repository contains the official implementation for the 6-week undergraduate research project focused on the paper: **"Spatiotemporal Attentive Entropy: A Geometry-Correct Cross-Entropy on S² for Calibrated, Low-Latency Viewport Prediction."**

## Core Objectives

Our goal is to implement the core components of the SAE paper, including:
1.  A data processing pipeline for 360° head-tracking data.
2.  A numerically stable, streaming estimator for the SAE metric.
3.  An LSTM-based prediction model that uses SAE as a feature.
4.  An evaluation suite to validate our results against the paper's findings.

## Getting Started

### Prerequisites
- Python 3.8+
- A virtual environment tool (e.g., `venv`, `conda`)

### Installation & Setup
1.  **Fork & Clone:** Each team member should fork this repository and then clone their fork locally:
    ```bash
    git clone https://github.com/[YOUR_USERNAME]/sae-viewport-prediction.git
    cd sae-viewport-prediction
    ```
2.  **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Directory Structure

The repository is organized to keep code, data, and experiments separate and clean.
- **/data:** Holds the raw and processed datasets. (Note: Large data files will not be committed to Git).
- **/src:** All source code for the project, organized into modules.
- **/notebooks:** Jupyter notebooks for data exploration, visualization, and experiments.
- **/docs:** Project planning documents, like our project guide.

---
