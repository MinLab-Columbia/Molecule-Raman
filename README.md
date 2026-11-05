# Molecular Code of Biological Raman Spectroscopy

This repository contains the computational code associated with the study *Molecular Code of Biological Raman Spectroscopy*.

## Overview

Raman spectroscopy provides label-free measurements of molecular vibrations in biological specimens, but assigning overlapping Raman signals to individual molecular species remains challenging. In this study, spatially paired Raman measurements and matrix-assisted laser desorption/ionization mass spectrometry imaging (MALDI-MSI) are used to investigate the molecular information encoded in biological Raman spectra.

The computational framework examines the relationship between Raman spectral features and MALDI-derived molecular profiles in two directions: predicting Raman spectra from molecular measurements (**Decoder**) and predicting molecular profiles from Raman spectra (**Translator**).

## Experimental and computational framework

![Experimental and computational overview of paired Raman spectroscopy and MALDI-MSI](Fig1.png)

*Fig. 1 | Experimental workflow, Raman imaging instrumentation, and representative molecular and spectral images from biological tissues.*

## Computational methods

### Decoder: molecular-to-spectral mapping

The Decoder uses paired MALDI and Raman measurements to fit a ridge regression model that maps molecular abundance features to Raman spectral features. The fitted model can be applied to an independent dataset with the same molecular feature definitions to predict its Raman spectra.

- **Input:** MALDI molecular abundance matrix
- **Target:** Raman spectral intensity matrix

### Translator: spectral-to-molecular mapping

The Translator fits the reverse mapping, using Raman spectral features to predict MALDI molecular abundance features. The model can be evaluated on an independent, spatially paired Raman–MALDI dataset.

- **Input:** Raman spectral intensity matrix
- **Target:** MALDI molecular abundance matrix

Both scripts implement model fitting, prediction, and evaluation from user-supplied paired data. Preprocessing of raw measurements and spatial registration are not part of this repository.

## Repository contents

```text
Molecule-Raman/
├── README.md
├── Fig1.png
├── decoder.py
├── translator.py
├── requirements.txt
└── .gitignore
```

- `decoder.py`: MALDI-to-Raman regression workflow
- `translator.py`: Raman-to-MALDI regression workflow
- `Fig1.png`: Experimental and computational overview
- `requirements.txt`: Python package dependencies

## Installation

```bash
git clone https://github.com/MinLab-Columbia/Molecule-Raman.git
cd Molecule-Raman
pip install -r requirements.txt
```

## Input data

The scripts accept preprocessed, spatially paired Raman and MALDI measurements in HDF5 (`.h5`) files. Each file must contain:

| HDF5 dataset | Contents | Shape |
| --- | --- | --- |
| `maldi_matrix` | Molecular abundance measurements | `(n_pixels, n_maldi_features)` |
| `raman_matrix_median` | Raman spectral intensities | `(n_pixels, n_raman_features)` |

The optional `x_coords` and `y_coords` datasets may be included for spatial reference. Rows in the Raman and MALDI matrices must represent corresponding spatial locations. Training and test datasets must use the same feature identities and order within each modality.

## Usage

Use separate HDF5 files for model fitting and evaluation. Both files must contain paired Raman and MALDI matrices in the format described above.

**Molecular-to-spectral decoding**

```bash
python decoder.py --train path/to/train.h5 --test path/to/test.h5
```

**Spectral-to-molecular translation**

```bash
python translator.py --train path/to/train.h5 --test path/to/test.h5
```

Each script fits a ridge regression model on the training data and evaluates predictions against the corresponding measured features in the test data. No pretrained weights are required.

## Data availability

The complete experimental datasets are not hosted in this repository because of their size. No pretrained models are included. Users may run the code on their own preprocessed and spatially registered Raman–MALDI measurements. Availability of the data supporting the study is described in the manuscript's Data Availability statement.
