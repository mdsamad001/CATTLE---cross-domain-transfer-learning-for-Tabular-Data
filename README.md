# CATTLE: Cross-Attention Transfer Learning for Tabular Data

This repository implements **CATTLE**, a framework for transfer learning on tabular data using cross-attention mechanisms.

---

## ⚙️ Installation

Clone the repository and create the conda environment:

```bash
cd cattle
conda env create -f env.yml
conda activate conda-cattle
```
---

## 🚀 Usage
### Pretraining on `credit-g` and Fine-Tuning on `diabetes`

```bash
python run_pipeline_self_supervised.py --source credit-g --target diabetes --folds 0 1 2 3 4 --epochs 100
```

---
