# Smart-Agriculture-ANN

Project Overview
----------------

This repository contains a Jupyter notebook that builds, trains, evaluates, and saves an Artificial Neural Network (ANN) to predict crop yield categories from environmental and agricultural data.

Notebook: [notebooks/Smart_Agriculture_ANN.ipynb](notebooks/Smart_Agriculture_ANN.ipynb)

Dataset
-------

CSV files are stored under the `data/smart-agri-dataset/` directory:

- `pesticides.csv` — pesticide usage / counts
- `rainfall.csv` — rainfall measurements
- `temp.csv` — temperature records
- `yield.csv` — crop yield values/labels

Notebook Functionality
----------------------

- Data loading & EDA: inspect data, missing values, and correlations.
- Preprocessing: numeric/categorical imputation, encoding, and scaling with `StandardScaler`.
- Feature selection: univariate selection using `SelectKBest`.
- ANN modeling: model definition, compilation, training (Keras/TensorFlow), and callbacks.
- Evaluation metrics: accuracy, precision, recall, F1, confusion matrix, probability outputs and plots.
- Scenario testing: feed example scenarios through the preprocessing pipeline and model to obtain predictions.

Key Files Produced
------------------

| File | Description | Path |
|------|-------------|------|
| Trained ANN model | Keras HDF5 model | notebooks/models/ann_model.h5 |
| Label encoder | Encodes target classes | notebooks/models/label_encoder.joblib |
| Scaler | Feature scaling object | notebooks/models/scaler.joblib |
| Feature selector | `SelectKBest` object | notebooks/models/selector.joblib |

Dependencies
------------

Create a virtual environment and install dependencies (choose one):

Option A — Windows `venv`:

```bash
python -m venv .venv
.venv\Scripts\activate    # PowerShell/Command Prompt on Windows
pip install -r requirements.txt
```

Option B — conda environment:

```bash
conda create -n Smart-Agriculture-ANN python=3.12 -y
conda activate Smart-Agriculture-ANN
pip install -r requirements.txt
```

Example prompt after activating conda:

```
D:\A-Notes\project\Smart-Agriculture-ANN>conda activate Smart-Agriculture-ANN

(Smart-Agriculture-ANN) D:\A-Notes\project\Smart-Agriculture-ANN>
```

numpy
scikit-learn
matplotlib
seaborn
jupyter
tensorflow
joblib

# Optional (for faster training / GPU)
# tensorflow-gpu
```

Quick Start
-----------

1. Clone the repo and open the notebook:

```bash
git clone <repo-url>
cd Smart-Agriculture-ANN
jupyter lab    # or jupyter notebook
```

2. Run the notebook `notebooks/Smart_Agriculture_ANN.ipynb` and execute cells sequentially.

3. Load the saved model and artifacts to run predictions (example):

```python
from tensorflow.keras.models import load_model
import joblib
import numpy as np

# Paths
MODEL_PATH = 'notebooks/models/ann_model.h5'
SCALER_PATH = 'notebooks/models/scaler.joblib'
ENCODER_PATH = 'notebooks/models/label_encoder.joblib'
SELECTOR_PATH = 'notebooks/models/selector.joblib'

# Load
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
le = joblib.load(ENCODER_PATH)
selector = joblib.load(SELECTOR_PATH)

# Example: single sample with same predictor order used in the notebook
sample = np.array([[/* replace with numeric feature values */]])
sample_scaled = scaler.transform(sample)
sample_sel = selector.transform(sample_scaled)
probs = model.predict(sample_sel)
pred_label = le.inverse_transform([probs.argmax(axis=1)[0]])
print('Predicted:', pred_label)
```

How to Use
----------

- Run the notebook end-to-end to reproduce preprocessing, training, evaluation, and artifact saving.
- To use only saved artifacts, load the `scaler`, `selector`, and `label_encoder` from `notebooks/models/` and pass preprocessed inputs into `ann_model.h5` as shown above.

Notes & Tips
------------

- Visualizations (confusion matrices, plots) are best viewed in JupyterLab or Jupyter Notebook.
- Training time depends on dataset size and hardware; reduce `epochs` or `batch_size` for faster experiments.
- Typical batch sizes to try: 16, 32, 64. Smaller batches may improve generalization but increase training time.

License & Contributing
----------------------

This project is provided under the MIT License (or choose your preferred license). Contributions are welcome — please open an issue or a pull request with changes.

If you'd like, I can:

- add an explicit `requirements.txt` file and commit it.
- add a small script `scripts/predict.py` to load artifacts and run batch predictions.

---
Updated to provide a concise, user-friendly guide for `notebooks/Smart_Agriculture_ANN.ipynb` and associated artifacts.
