# Taiwan Precipitation Downscaling

**Application of a Diffusion-Based Deep Learning Model for Downscaling Precipitation Forecasts in Taiwan**

## Overview

This repository is associated with a master's thesis conducted at the Department of Civil Engineering, National Chung Hsing University, Taiwan.

This study investigates the application of a diffusion-based generative model (DM) for high-resolution precipitation downscaling over Taiwan.

The proposed framework performs 20-fold spatial downscaling, transforming coarse-resolution precipitation fields (0.25°) into high-resolution precipitation estimates (0.0125°, approximately 1.25 km).

The study also investigates the application of the diffusion model to GraphCast precipitation forecasts.

## Research Objectives

- Develop a diffusion-based precipitation downscaling framework for Taiwan.
- Reconstruct high-resolution precipitation patterns from coarse-resolution inputs.
- Evaluate the downscaling performance against QPESUMS precipitation data.
- Compare the diffusion model with Quantile Mapping (QM).
- Investigate the application of diffusion-based downscaling to GraphCast precipitation forecasts.

## Dataset

- **Reference dataset:** QPESUMS precipitation estimates
- **Training period:** 2015–2023
- **Testing period:** 2024
- **Input resolution:** 0.25°
- **Target resolution:** 0.0125°
- **Precipitation accumulation:** 12 hours

## Evaluation

The study evaluates precipitation downscaling performance using:

- Bias
- Correlation Coefficient (CC)
- Root Mean Square Error (RMSE)
- Probability Density Function (PDF)
- Fractions Skill Score (FSS)
- Power Spectral Density (PSD)

## Original Research and Acknowledgements

This work builds upon the ClimateDiffuse framework developed by Robbie A. Watt and Laura A. Mansfield.

Original repository: https://github.com/robbiewatt1/ClimateDiffuse

The diffusion-based approach was adapted and investigated for precipitation downscaling over Taiwan.

## Thesis

**Title:** Application of a Diffusion-Based Deep Learning Model for Downscaling Precipitation Forecasts in Taiwan

**Author:** Chia-Jeng Chen & Ya-Heng Ma

**Institution:** National Chung Hsing University

**Department:** Civil Engineering

**Year:** 2026

## Code Availability

This repository contains the diffusion model architecture, training scripts, inference and visualization code, and the trained model checkpoint (`Model/45.pt`) used in this study.

QPESUMS precipitation datasets are not included. Users are required to obtain the necessary data separately and modify the file paths in the scripts according to their local environment.

Detailed experimental results and performance evaluations are presented in the accompanying thesis.