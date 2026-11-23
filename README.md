# Taiwan Precipitation Downscaling

**Application of a Diffusion-Based Deep Learning Model for Downscaling Precipitation Forecasts in Taiwan**

## Overview

This repository provides research code associated with a study on diffusion-based precipitation downscaling over Taiwan, conducted at the Department of Civil Engineering, National Chung Hsing University, Taiwan.

The study investigates the application of a diffusion-based generative model (DM) to enhance the spatial resolution of precipitation forecasts over Taiwan.

The framework performs 20-fold spatial downscaling, transforming coarse-resolution precipitation fields (0.25°) into high-resolution precipitation estimates (0.0125°, approximately 1.25 km).

The framework is further evaluated using GraphCast precipitation forecasts to investigate its applicability to high-resolution precipitation forecasting over Taiwan.

## Research Objectives

- Develop a diffusion-based precipitation downscaling framework for Taiwan.
- Reconstruct high-resolution precipitation patterns from coarse-resolution inputs.
- Evaluate downscaling performance against QPESUMS precipitation estimates.
- Compare the diffusion model with Quantile Mapping (QM).
- Investigate the application of diffusion-based downscaling to GraphCast precipitation forecasts.

## Dataset

- **Reference dataset:** QPESUMS precipitation estimates
- **Training period:** 2015–2023
- **Testing period:** 2024
- **Input resolution:** 0.25°
- **Target resolution:** 0.0125°
- **Precipitation accumulation:** 12 hours

QPESUMS precipitation datasets are not distributed with this repository. Users must obtain the required data separately.

## Evaluation Metrics

The study evaluates precipitation downscaling performance using:

- Bias
- Correlation Coefficient (CC)
- Root Mean Square Error (RMSE)
- Probability Density Function (PDF)
- Fractions Skill Score (FSS)
- Power Spectral Density (PSD)

Detailed experimental results are presented in the associated research manuscript.

## Repository Contents

- `src/Network.py` — Diffusion model architecture and EDM preconditioning.
- `src/datasetTaiwan.py` — Precipitation dataset loading and preprocessing.
- `src/diffusiontrain_Taiwan.py` — Diffusion model training and sampling.
- `src/inference.py` — Model inference and visualization.
- `Model/45.pt` — Trained model checkpoint used for precipitation downscaling.
- `requirements.txt` — Python package dependencies.
- `figure/` — Selected precipitation downscaling visualizations.

## Environment and Usage

The code was developed using Python and PyTorch.

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

A CUDA-compatible PyTorch installation may be required for GPU-based execution.

Users must configure dataset paths according to their local computing environments. QPESUMS data are not included in this repository.

## Original Research and Acknowledgements

This research builds upon the ClimateDiffuse framework developed by Robbie A. Watt and Laura A. Mansfield.

Original repository: https://github.com/robbiewatt1/ClimateDiffuse

The diffusion-based framework was adapted for precipitation downscaling over Taiwan, including the use of high-resolution QPESUMS precipitation data and applications involving GraphCast forecasts.

The original research and associated implementations should be acknowledged and cited where applicable.

## Research Team

- **Chia-Jeng Chen**
- **Ya-Heng Ma**

Department of Civil Engineering, National Chung Hsing University, Taiwan.

## Publication

**Manuscript title:** Application of a Diffusion-Based Deep Learning Model for Downscaling Precipitation Forecasts in Taiwan

**Authors:** Chia-Jeng Chen and Ya-Heng Ma

**Institution:** National Chung Hsing University, Taiwan

**Status:** Manuscript in preparation

Publication details, including the journal, DOI, and citation information, will be updated following publication.

## Code Availability

The research code, model architecture, training scripts, inference and visualization code, and trained model checkpoint (`Model/45.pt`) are provided in this repository.

The QPESUMS precipitation datasets are not included. Users are responsible for obtaining the necessary datasets and adjusting file paths to match their local environments.

GitHub repository: https://github.com/mahengggg/Taiwan-Precipitation-Downscaling
