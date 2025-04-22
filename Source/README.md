# Air Quality Sensor Calibration Project

This project implements linear and advanced models to calibrate electrochemical sensors for measuring **OZONE (O₃)** and **NO₂** levels, as part of **CSC14003 - Introduction to Artificial Intelligence**.

## 📁 Directory Structure

```
StudentID1_StudentID2_StudentID3_StudentID4/
├── Docs
│   ├── Report.pdf
│   ├── Presentation.pdf
├── Source
│   ├── data
│   │   ├── train.csv
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

## 🧰 Requirements

Install the dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 Usage

### 🔧 Training Models

To train **both** linear and advanced models, run:

```bash
python train.py
```

✅ Outputs:
- `models/linear_model.pkl`
- `models/advanced_model.pkl`

Both will be saved to the `models/` directory, which will be created automatically if it doesn't exist.

Alternatively, train each model individually:

- **Linear Model:**
  ```bash
  python linear_model.py
  ```

- **Advanced Model:**
  ```bash
  python advanced_model.py
  ```

### 📊 Prediction

Use `predict.py` to predict O₃ and NO₂ levels:

```bash
python predict.py
```

> **Note:** It uses `models/advanced_model.pkl` by default.  
To test with custom data, modify the `sample_input` inside `predict.py`.

## 📄 File Descriptions

- `utils.py`: Utility functions for data loading, feature extraction, scaling, and evaluation.
- `train.py`: Trains both linear and advanced models.
- `linear_model.py`: Implements Task 1 (voltage-based linear regression models).
- `advanced_model.py`: Implements Task 2 with various non-linear models using full feature set.
- `predict.py`: Loads saved models to predict O₃ and NO₂ from sample input.
- `linear_model.pkl`: Saved linear regression models.
- `advanced_model.pkl`: Saved best-performing advanced models.

## 📌 Notes

- Ensure `train.csv` is located in the `data/` directory.
- **Task 1** uses only voltage outputs: `o3op1`, `o3op2`, `no2op1`, `no2op2`.
- **Task 2** uses all features: `temp`, `humidity`, `hour`, along with voltage outputs.

## ✏️ Model Methodology

### Linear Models (Task 1)
The linear models use only the sensor voltage outputs to predict pollutant levels according to the manufacturer's equations:
- O₃ = p_o3 · o3op1 + q_o3 · o3op2 + r_o3 · no2op1 + s_o3 · no2op2 + t_o3
- NO₂ = p_no2 · o3op1 + q_no2 · o3op2 + r_no2 · no2op1 + s_no2 · no2op2 + t_no2

We evaluate different linear model implementations, including Ridge and Lasso regularization.

### Advanced Models (Task 2)
The advanced models incorporate additional features such as temperature, humidity, and time of day to improve prediction accuracy. We implement various non-linear approaches, including SVR with RBF kernel, neural networks (MLP), and KNN.

> **Course:** CSC14003 - Introduction to Artificial Intelligence  
> **Topic:** Air Quality Sensor Calibration