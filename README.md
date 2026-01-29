# Math Expression Retrieval

This repository implements a **large-scale mathematical expression retrieval system**
based on **hierarchical similarity modeling** and **graph-aware ranking**.

The system is designed for research purposes and serves as the experimental
codebase for a PhD thesis on **similar mathematical expression retrieval**.

## ✨ Key Features

- Structural recall using Approach0-style formula hashing
- Semantic coarse ranking with MathBERT embeddings
- Learning-to-rank via LightGBM LambdaRank
- High-confidence semantic filtering
- Graph-aware re-ranking using a Formula Concept Graph (FCG)
- End-to-end retrieval latency under 50 ms (single GPU)

## 📁 Repository Structure

# Math Expression Retrieval – Reproducible Demo

This repository provides a **minimal reproducible demo** of the system
described in our paper:

> Hierarchical Similarity Modeling for Mathematical Expression Retrieval

The demo reproduces the main experimental findings using a small-scale dataset.

---

## Environment

- Python ≥ 3.9
- Optional: CUDA-enabled GPU (CPU also supported)

Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .


