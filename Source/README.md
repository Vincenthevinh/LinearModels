
# Air Quality Sensor Calibration Project

This project implements linear and advanced models to calibrate electrochemical sensors for measuring **OZONE (O₃)** and **NO₂** levels, as part of **CSC14003 - Introduction to Artificial Intelligence**.

---

## 📁 Directory Structure

```
23127520/
├── Docs
│   ├── Report.pdf
│   ├── Presentation.pdf
├── Source
│   ├── data
│   │   ├── train.csv
│   │   ├── dummy_test.csv
│   ├── models
│   │   ├── linear_model.pkl
│   │   ├── advanced_model.pkl
│   ├── linear_model.py
│   ├── advanced_model.py
│   ├── predict.py
│   ├── train.py
│   ├── utils.py
│   ├── requirements.txt
│   ├── README.md
```

---

## 🧰 Requirements

Install the dependencies:

```bash
pip install -r Source/requirements.txt
```

---

## 🚀 Usage

### 🔧 Training Models

To train **both** linear and advanced models, run:

```bash
python Source/train.py
```

✅ Outputs:
- `linear_model.pkl`
- `advanced_model.pkl`

Both will be saved to `Source/models/`.

---

Alternatively, train each model individually:

- **Linear Model:**
  ```bash
  python Source/linear_model.py
  ```

- **Advanced Model (SVR, MLP, KNN):**
  ```bash
  python Source/advanced_model.py
  ```

---

### 📊 Prediction

Use `predict.py` to predict O₃ and NO₂ levels:

```bash
python Source/predict.py
```

> **Note:** It uses `Source/models/advanced_model.pkl` by default.  
To test with custom data, modify the `sample_input` inside `predict.py`.

---

## 📄 File Descriptions

- `utils.py`: Utility functions for data loading, feature extraction, scaling, and evaluation.
- `train.py`: Trains both linear and advanced models.
- `linear_model.py`: Implements Task 1 (voltage-based linear regression models).
- `advanced_model.py`: Implements Task 2 with SVR (RBF), MLP, and KNN using full feature set.
- `predict.py`: Loads saved models to predict O₃ and NO₂ from sample input.
- `linear_model.pkl`: Saved linear regression models.
- `advanced_model.pkl`: Saved best-performing advanced models.

---

## 📌 Notes

- Ensure both `train.csv` and `dummy_test.csv` are located in `Source/data/`.
- **Task 1** uses only voltage outputs: `o3op1`, `o3op2`, `no2op1`, `no2op2`.
- **Task 2** uses all features: `temp`, `humidity`, `hour`, along with voltage outputs.
- No scaler is saved — features are re-scaled at prediction time.

---

> **Course:** CSC14003 - Introduction to Artificial Intelligence  
> **Topic:** Air Quality Sensor Calibration  
