# Probabilistic Graphical Models Project

This repository contains our project for the *MVA* course titled **"Probabilistic Graphical Models"**.

## Overview
In this project, we studied and implemented key ideas from the paper:

**scVAE: Variational Autoencoders for Single-Cell Gene Expression Data**  
*Christopher Heje Grønbech<sup>1,2,3,*</sup>, Maximillian Fornitz Vording<sup>3</sup>, Pascal N. Timshel<sup>4</sup>, Casper Kaae Sønderby<sup>1</sup>, Tune H. Pers<sup>4</sup>, and Ole Winther<sup>1,2,3</sup>*  

The original paper explores the use of **Variational Autoencoders (VAEs)** to analyze single-cell gene expression data, providing a probabilistic generative framework to handle such high-dimensional datasets.

## Goals
Our objectives for this project were:
1. **Understand** the concepts, methods, and results presented in the paper.
2. **Implement** the proposed methods from scratch.
3. **Train and evaluate** models to reproduce results similar to those described in the paper.

## Results
We successfully trained several models that achieve the goals outlined in the paper. The trained models and results can be accessed [here](https://drive.google.com/drive/folders/1ZK7qd9wNTPS0qA9G4BhoP5M-5wuxcpsA).

## Getting Started
### Prerequisites
To run the code, ensure the following dependencies are installed:
- Python 3.x
- PyTorch / TensorFlow
- NumPy
- sklearn
- tqdm
- random
- request
- scipy
- h5py
- Matplotlib 
- tarfile
- csv
- 
### Running the Code
To train the model, execute the following command:
```bash
python3 main.py --training=True --epochs=25 --batch-size=128 --likelihood-distrib=NegativeBinomial --optimizer=AdamW --model-name=GMVariationalAutoEncoder_transformers --model_path=GMVariationalAutoEncoder_transformers
```

and to evaluate :
```bash
python3 main.py--evaluate=True --model-name=GMVariationalAutoEncoder_transformers  --model_path=GMVariationalAutoEncoder_transformers
```

## Contact
If you have any questions or require further clarification, feel free to reach out to us:
- **Team Members**: Darius Dabert, Hugo Fruchet, Samuel Sarfati

---
**Note**: This project was conducted as part of the *MVA - Probabilistic Graphical Models* course at ENS Paris-Saclay.
