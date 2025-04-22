import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

def load_data(file_path):
    """
    Load and validate CSV data.
    
    Parameters:
    file_path (str): Path to CSV file
    
    Returns:
    pd.DataFrame: Loaded data
    """
    try:
        df = pd.read_csv(file_path, parse_dates=['Timestamp'])
        required_columns = ['Timestamp', 'OZONE', 'NO2', 'temp', 'humidity', 'no2op1', 'no2op2', 'o3op1', 'o3op2']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV must contain columns: {required_columns}")
        return df
    except Exception as e:
        raise ValueError(f"Error loading {file_path}: {str(e)}")

def extract_features(df, include_hour=False):
    """
    Extract features from DataFrame, optionally adding 'hour' from timestamp.
    
    Parameters:
    df (pd.DataFrame): Input data
    include_hour (bool): Whether to extract 'hour' from timestamp
    
    Returns:
    pd.DataFrame: DataFrame with selected features
    """
    # Task 1 - Use only voltage outputs
    features = ['o3op1', 'o3op2', 'no2op1', 'no2op2']
    
    # Task 2 - Add temperature, humidity, and time features
    if include_hour:
        features.extend(['temp', 'humidity'])
        df = df.copy()
        df['hour'] = df['Timestamp'].dt.hour
        features.append('hour')
    
    return df[features]

def scale_features(X):
    """
    Scale features using StandardScaler.
    
    Parameters:
    X (pd.DataFrame or np.array): Features to scale
    
    Returns:
    np.array: Scaled features
    """
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def evaluate_model(model, X, y_true):
    """
    Evaluate model using Mean Absolute Error.
    
    Parameters:
    model: Trained model
    X (np.array): Features
    y_true (np.array): True labels
    
    Returns:
    float: MAE
    """
    y_pred = model.predict(X)
    return mean_absolute_error(y_true, y_pred)

def ensure_directory_exists(directory_path):
    """
    Create directory if it doesn't exist.
    
    Parameters:
    directory_path (str): Path to directory
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)