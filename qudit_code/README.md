# Quantum-Inspired Neural Network on EMNIST

This repository provides code and utilities for training and evaluating a **quantum-inspired neural network (QnitCNet)** on the **EMNIST dataset**, with feature preprocessing via **Discrete Cosine Transform (DCT)** and **Principal Component Analysis (PCA)**. The project includes data preparation, feature extraction, model implementation, and cross-validation.

---

## Repository Structure

```
.
├── emnist_functions.py   # Load EMNIST, apply DCT + ZigZag ordering, feature extraction
├── pca_emnist.py         # Load EMNIST, apply PCA for dimensionality reduction
├── QnitCNet.py           # Quantum-inspired neural network classifier (core model)
├── KFolderCV.py          # Cross-validation framework with PCA and QnitCNet
└── README.md             # Project documentation (this file)
```

---

## File Descriptions

### `emnist_functions.py`
- Loads EMNIST dataset splits (`balanced`, `byclass`, `bymerge`, `digits`, `letters`, `mnist`).  
- Preprocesses images with **2D Discrete Cosine Transform (DCT)**.  
- Extracts features using **zig-zag ordering** of coefficients.  
- Includes functions for **parallelized preprocessing** to speed up dataset handling.

### `pca_emnist.py`
- Loads EMNIST dataset (train + test).  
- Flattens images into vectors.  
- Provides functions to **fit PCA** on the dataset and transform it.  
- Includes **parallelized PCA preprocessing** of batches.

### `QnitCNet.py`
- Implements **QnitCNet**, a quantum-inspired neural network classifier.  
- Combines:
  - Polynomial feature expansion (`power_type=0`) or interaction terms (`power_type=1`)  
  - Cascade of linear SVMs  
  - Quantum-inspired decision rule using trigonometric functions of SVM outputs  
- Provides:
  - `train()` and `trainNP()` (parallel and non-parallel training)  
  - `predict()` for inference  
  - Feature preprocessing methods (`preprocess_dataP`, `preprocess_dataM`).

### `KFolderCV.py`
- Implements **k-fold cross-validation** for evaluating models.  
- Supports:
  - Parallel and sequential evaluation  
  - PCA dimensionality reduction  
  - Integration with **QnitCNet**  
- Returns mean accuracy, standard deviation, training time, and number of features.

---

## Installation

### Requirements
- Python ≥ 3.9  
- Dependencies:
  ```
  torch
  torchvision
  numpy
  scikit-learn
  scipy
  joblib
  matplotlib
  ```

### Setup
```bash
# Clone repository
git clone https://github.com/<username>/<repo>.git
cd <repo>

# (Optional) Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Feature Extraction with DCT
```python
from emnist_functions import load_emnist_split, parallel_process

train_loader, test_loader = load_emnist_split("balanced")
X_train, y_train = parallel_process(train_loader, coef_number=50)
```

### 2. Feature Extraction with PCA
```python
from pca_emnist import fit_pca_on_full_dataset

X, y = fit_pca_on_full_dataset("digits")
```

### 3. Training QnitCNet
```python
from QnitCNet import QnitCNet

model = QnitCNet(X, y, neuron_number=3, power_type=0)
model.train()
predictions = model.predict(X)
```

### 4. Cross-Validation
```python
from KFolderCV import KFolderCV

cv = KFolderCV(n_splits=10)
mean_acc, std_acc, mean_time, std_time, nv = cv.cross_validate(X, y, neurons=3, n_components=30, model_type=0)
print("Accuracy:", mean_acc)
```

---

## Citation

If you use this code, please cite the corresponding manuscript.

```bibtex
@misc{yourkey2025,
  title        = {Quantum-Inspired Neural Network on EMNIST},
  author       = {Your Name},
  year         = {2025},
  howpublished = {GitHub repository},
  url          = {https://github.com/<username>/<repo>}
}
```

---

## License

Specify your license here (MIT, Apache-2.0, GPL-3.0, etc.).
