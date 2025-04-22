import pandas as pd
import pickle
from utils import extract_features, scale_features

def predict_air_quality(input_data, model_path='./Source/models/advanced_model.pkl', task='task2'):
    """
    Predict OZONE and NO2 levels from input features.
    
    Parameters:
    input_data (dict or pd.DataFrame): Input with columns:
        - Task 1: no2op1, no2op2, o3op1, o3op2
        - Task 2: Time, temp, humidity, no2op1, no2op2, o3op1, o3op2
    model_path (str): Path to the .pkl file containing the models
    task (str): 'task1' or 'task2' to specify feature set
    
    Returns:
    dict: Predicted OZONE and NO2 levels
    """
    # Convert input to DataFrame if dict
    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])
    else:
        input_df = input_data.copy()
    
    # Define required columns based on task
    if task == 'task1':
        required_cols = ['no2op1', 'no2op2', 'o3op1', 'o3op2']
        include_hour = False
    else:  # task2
        required_cols = ['Time', 'temp', 'humidity', 'no2op1', 'no2op2', 'o3op1', 'o3op2']
        include_hour = True
        input_df['Time'] = pd.to_datetime(input_df['Time'])
    
    # Check for required columns
    if not all(col in input_df.columns for col in required_cols):
        raise ValueError(f"Input data must contain: {required_cols}")
    
    # Extract features
    X = extract_features(input_df, include_hour=include_hour)
    
    # Scale features
    X_scaled = scale_features(X)
    
    # Load models
    with open(model_path, 'rb') as f:
        models_dict = pickle.load(f)
    
    o3_model = models_dict['o3_model']
    no2_model = models_dict['no2_model']
    
    # Predict
    o3_pred = o3_model.predict(X_scaled)
    no2_pred = no2_model.predict(X_scaled)
    
    return {
        'OZONE_pred': o3_pred.tolist(),
        'NO2_pred': no2_pred.tolist()
    }

if __name__ == "__main__":
    # Example usage for Task 2
    sample_input_task2 = {
        'Time': '2019-03-27 17:05:00',
        'temp': 41.2,
        'humidity': 28.9,
        'no2op1': 179.0,
        'no2op2': 194.0,
        'o3op1': 220.0,
        'o3op2': 192.0
    }
    
    # Example usage for Task 1
    sample_input_task1 = {
        'no2op1': 179.0,
        'no2op2': 194.0,
        'o3op1': 220.0,
        'o3op2': 192.0
    }
    
    # Predict for Task 2
    print("Task 2 Prediction:")
    predictions_task2 = predict_air_quality(sample_input_task2, model_path='./Source/models/advanced_model.pkl', task='task2')
    print(f"Predictions: {predictions_task2}")
    
    # Predict for Task 1
    print("\nTask 1 Prediction:")
    predictions_task1 = predict_air_quality(sample_input_task1, model_path='./Source/models/linear_model.pkl', task='task1')
    print(f"Predictions: {predictions_task1}")