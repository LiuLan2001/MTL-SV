
This repository contains the official implementation of the EuroVis 2026 paper:

**Volume Data Reconstruction and Uncertainty Quantification in an Implicit Neural Representation via Multi-Task Learning**

We provide the code for the baseline method, the state-of-the-art comparison method **RMDSRN**, and our proposed method, together with the three datasets used in the paper.

## Repository Structure

- **`utils.py`** contains shared utility functions, such as random data sampling and coordinate grid generation.
- **`dataio.py`** provides the common data input and output classes.
- Each method has its own model definition and training pipeline, which are implemented in the corresponding Python files.
- The **Jupyter notebooks** include the scripts for running experiments and evaluating results.

## Input Representation

The input to the models is **4D spatiotemporal coordinates**, which means the framework can handle both:

- **single-frame volumetric data**, and
- **time-varying volumetric sequences**.

This unified formulation allows the methods in this repository to be applied to a broad range of scientific volume data reconstruction tasks.
