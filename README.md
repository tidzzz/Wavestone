# Academic Project – IAM Analysis for Wavestone

This repository contains an academic project carried out as part of a course (PCE), but executed as a real-world mission for the company Wavestone. The goal is to explore and operationalize an approach to analyzing IAM (Identity & Access Management) permissions using provided datasets, with clustering and matrix factorization algorithms to identify access patterns and rationalization opportunities.

The work is organized around exploratory/experimental notebooks and scripts to build user–permission matrices. Deliverables aim to be reproducible and usable by a data/consulting team.

## Repository Structure

- `data.py`: utilities for data loading/processing.
- `requirements.txt`: Python dependencies for the project.
- `algo/`: experiment notebooks for algorithms.
  - `K-means.ipynb`: K-means clustering on rights/permissions matrices.
  - `NMF+RF.ipynb`: NMF factorization followed by Random Forest for classification/explainability.
  - `NMF+MultipleRF.ipynb`: multiple RF variants on NMF factors.
- `iam_dataset/`: IAM datasets (CSV, statistical summary JSON).
  - `users.csv`, `permissions.csv`, `rights.csv`, `applications.csv`, `statistical_summary.json`.
- `matrix/`: scripts to build analysis matrices.
  - `build_user_permission_matrix.py`: generates the user × permission matrix.

## Prerequisites

- Python 3.10+ (tested on macOS using a virtual environment).
- Jupyter Notebook support in VS Code.

## Installation

Using a dedicated virtual environment is recommended.

```zsh
# From the project’s root
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

- Build the user–permission matrix:

```zsh
source .venv/bin/activate
python matrix/build_user_permission_matrix.py
```

Outputs and/or generated artifacts will be stored according to the script’s configuration (adjust as needed in `build_user_permission_matrix.py`).

- Explore models in notebooks (`algo/`): open the notebooks in VS Code and run cell by cell. Notebooks expect the matrix and datasets to be present.

## Data

Files in `iam_dataset/` represent an anonymized IAM dataset (users, permissions, rights, applications) used in the study. The `statistical_summary.json` file provides a quick statistical overview for sanity checks.

## Methodology

- Data preparation (cleaning, joins, matrix construction).
- Clustering (K-means) to detect groups of users with similar access profiles.
- NMF (Non-negative Matrix Factorization) to reduce dimensionality and extract “access factors”.
- Random Forest modeling for explainability and classification on NMF components.

## Branch and Pipeline

Development is tracked on the `Pipeline-2-steps` branch. Notebooks and scripts have been validated via local runs and installation of the listed dependencies.

## Collaboration with Wavestone

This project is designed and executed as a real mission for Wavestone, with business objectives tied to IAM rights optimization and governance. Results, recommendations, and produced artifacts are aligned with an enterprise context (reproducibility, clear deliverables, methodological rigor).

## Authors and Academic Context

Academic project within the PCE course, completed by the student team, in partnership with Wavestone for an IAM use case.

## License and Confidentiality

Data and notebooks may contain sensitive information related to IAM. Any reuse must respect confidentiality and agreements with Wavestone. Unless otherwise stated, this repository is intended for pedagogical and demonstration use for the company.

