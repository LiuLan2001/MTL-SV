# MTL-SV

This repository contains the official implementation of the following EuroVis 2026 paper:

**Volume Data Reconstruction and Uncertainty Quantification in an Implicit Neural Representation via Multi-Task Learning**

## Overview

This repository includes implementations of multiple methods for volume data reconstruction and uncertainty quantification in implicit neural representations (INRs). Specifically, it contains:

- the **baseline method**,
- the **state-of-the-art comparison method RMDSRN**, and
- **our proposed method**.

In addition, we provide the three datasets used in the paper.

The input to the models is **4D spatiotemporal coordinates**, which means that the methods can handle both:

- **single-frame volumetric data**, and
- **time-varying volumetric sequences**.

## Repository Structure

- **`utils.py`** contains shared utility functions, such as random data sampling and coordinate grid generation.
- **`dataio.py`** provides the common data input/output classes.
- Each method has its own model definition and training procedure, which are implemented in the corresponding Python files.
- The **Jupyter notebooks** contain the code for running experiments and evaluating results.

## Environment

The code was developed and tested in a Jupyter-based environment with the following configuration:

- **Operating System:** Linux
- **Platform:** Linux-4.18.0-372.32.1.el8_6.x86_64-x86_64-with-glibc2.17
- **Python:** 3.8.16
- **IPython:** 8.12.2
- **NumPy:** 1.21.5
- **Matplotlib:** 3.5.3
- **SciPy:** 1.10.1
- **tqdm:** 4.64.1
- **PyTorch:** 2.0.0+cu118
- **CUDA:** 11.8
- **cuDNN:** 8.7.0
- **GPU:** NVIDIA A800 80GB PCIe
