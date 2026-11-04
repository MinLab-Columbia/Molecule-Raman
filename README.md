# Molecular Code of Biological Raman Spectroscopy

This repository provides the core computational implementations accompanying the study **"Molecular Code of Biological Raman Spectroscopy."**

## Overview

Raman spectroscopy enables label-free, nondestructive characterization of biological samples by probing their intrinsic molecular vibrations. However, the molecular interpretation of complex biological Raman spectra remains challenging due to extensive spectral overlap and contributions from multiple molecular species.

In this study, we integrate spatially paired Raman spectroscopy and matrix-assisted laser desorption/ionization mass spectrometry imaging (MALDI-MSI) to investigate the molecular information encoded in biological Raman spectra. By establishing computational relationships between Raman spectral features and mass spectrometry-derived molecular measurements, the framework enables bidirectional analysis of spectral and molecular information.

This repository provides Python implementations of two core computational components: **molecular-to-spectral decoding (Decoder)** and **spectral-to-molecular translation (Translator)**.

## Computational Framework

![Overview of the experimental and computational framework](Fig1.png)

**Figure 1.** Overview of the experimental workflow and computational framework integrating Raman spectroscopy and MALDI mass spectrometry imaging for molecular interpretation of biological Raman spectra.

### 1. Molecular-to-Spectral Decoding (Decoder)

The Decoder models the relationship between molecular abundance profiles measured by MALDI-MSI and their corresponding Raman spectral signatures.

Using paired molecular and spectral measurements, a regularized regression model is established to predict Raman spectral features from MALDI molecular features. This approach provides a computational framework for investigating molecular contributions to biological Raman signals.

- **Input:** MALDI molecular feature matrix
- **Output:** Predicted Raman spectral feature matrix

### 2. Spectral-to-Molecular Translation (Translator)

The Translator performs the reverse mapping, predicting molecular abundance profiles from Raman spectral measurements.

Using paired Raman–MALDI data, the model learns relationships between Raman spectral features and mass spectrometry-derived molecular features. This framework enables Raman-based prediction of molecular distributions and supports the investigation of molecular information across biological tissues.

- **Input:** Raman spectral feature matrix
- **Output:** Predicted MALDI molecular feature matrix

Both components use ridge regression for regularized multivariate prediction.

## Repository Structure

```text
Molecular-Code-of-Biological-Raman-Spectroscopy/
├── README.md
├── Fig1.png
├── decoder.py
├── translator.py
├── requirements.txt
└── .gitignore
```

- **`decoder.py`** — Molecular-to-spectral decoding using paired MALDI and Raman measurements.
- **`translator.py`** — Spectral-to-molecular translation using paired Raman and MALDI measurements.
- **`Fig1.png`** — Overview of the experimental and computational framework.
- **`requirements.txt`** — Python dependencies for the computational workflow.
- **`.gitignore`** — Excludes local datasets, caches, and other non-repository files.

## Installation

Clone this repository and install the required Python packages:

```bash
git clone https://github.com/USERNAME/Molecular-Code-of-Biological-Raman-Spectroscopy.git
cd Molecular-Code-of-Biological-Raman-Spectroscopy
pip install -r requirements.txt
```

Replace `USERNAME` with the GitHub account or organization hosting this repository.

## Input Data

The computational workflows accept preprocessed, spatially paired Raman and MALDI datasets stored in HDF5 (`.h5`) format.

Each input file should contain the following datasets:

| Dataset | Description |
| --- | --- |
| `maldi_matrix` | Molecular abundance matrix, shape `(n_pixels, n_maldi_features)` |
| `raman_matrix_median` | Raman spectral matrix, shape `(n_pixels, n_raman_features)` |
| `x_coords` | Optional spatial x-coordinates |
| `y_coords` | Optional spatial y-coordinates |

Each row of the Raman and MALDI matrices must correspond to the same spatially registered measurement location. The data should already be preprocessed and spatially aligned before applying the computational workflows. Training and testing files must use consistent feature definitions for the model inputs and targets.

## Usage

The Decoder and Translator scripts operate on separate training and testing datasets. Each model is fitted on the user-provided training data; no pretrained model is required.

**Molecular-to-spectral decoding:**

```bash
python decoder.py --train train_data.h5 --test test_data.h5
```

**Spectral-to-molecular translation:**

```bash
python translator.py --train train_data.h5 --test test_data.h5
```

Each workflow follows the same general computational procedure:

1. Load paired Raman and MALDI feature matrices.
2. Construct the corresponding input and target matrices.
3. Fit a ridge regression model using the training dataset.
4. Predict spectral or molecular features for the test dataset.
5. Evaluate prediction performance.

Users can apply these implementations to their own preprocessed, spatially paired Raman–MALDI datasets.

## Data Availability

The complete experimental Raman and MALDI datasets are not included in this repository due to their large size.

This repository provides the core computational implementations rather than the complete experimental data-processing and figure-generation pipeline. No pretrained models are distributed. Users can train and evaluate the models using their own preprocessed, spatially paired Raman–MALDI datasets.

For information regarding the availability of datasets supporting the manuscript, please refer to the Data Availability statement in the associated publication when available.

## Citation

If you use this code in your research, please cite:

*Molecular Code of Biological Raman Spectroscopy.*

Full citation information will be updated upon publication.
