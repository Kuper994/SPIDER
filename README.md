# SPIDER

**SPIDER** (Specific Protein Interaction Detection and Estimation by Representation learning) is a deep learning framework for predicting context-specific protein–protein interaction (PPI) networks from multi-omics data.

The model integrates gene expression, protein abundance, protein localization, and experimental evidence to estimate whether an interaction is active in a given biological context, such as a specific cell line or tissue.

---

## Features

- Predicts context-specific PPIs from a general interaction network.
- Integrates multiple biological data sources:
  - Gene expression
  - Protein abundance
  - Protein co-abundance
  - Protein localization
  - Experimental interaction evidence
- Graph neural network architecture based on PyTorch Geometric.
- Supports transfer learning to new cell types and tissues.
- Includes an example notebook demonstrating model training and inference.

---

## Repository Structure

```
SPIDER/
├── data/                  # Input features (not included in the repository)
├── models         # spider model implementation
├──dataset              # dataset implementation
├── utils.py                 # Utility functions
├── example.ipynb   # Example notebook
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Kuper994/SPIDER.git
cd SPIDER
```

Create a Python environment and install the required packages:

```bash
pip install -r requirements.txt
```

The project requires Python 3.10+.

---

## Data

The repository does **not** include the datasets because several files exceed GitHub's size limits.

Expected inputs include:

- General protein–protein interaction network
- Gene expression data
- Protein abundance data
- Protein co-abundance matrices
- Protein localization annotations

Place the downloaded datasets inside the `data/` directory while preserving the expected folder structure.

---

## Example

An example workflow is provided in

```
example.ipynb
```

The notebook demonstrates:

1. Loading the biological features
2. Preparing graph inputs
3. Training SPIDER
4. Predicting context-specific interactions

---


## Contact

**Yael Kupershmidt** kupershmidt@mail.tau.ac.il

School of Computer Science and AI

Tel Aviv University

For questions or collaborations, please open an issue on GitHub.
