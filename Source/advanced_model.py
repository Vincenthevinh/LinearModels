import os
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
import pickle
from utils import load_data, extract_features, scale_features, evaluate_model, ensure_directory_exists

def train_advanced_model():
    """
    Train SVR, MLP, KNN, RandomForest, and GradientBoosting models for OZONE and NO2 using all features.
    Save best models to advanced_model.pkl.
    """
    # Load dataset
    train_df = load_data('./Source/data/train.csv')

    # Extract features with hour (Task 2)
    X = extract_features(train_df, include_hour=True)
    y_o3 = train_df['OZONE']
    y_no2 = train_df['NO2']

    # Scale features
    X_scaled = scale_features(X)

    # Split data into train and validation sets
    X_train, X_val, y_train_o3, y_val_o3 = train_test_split(X_scaled, y_o3, test_size=0.2, random_state=42)
    _, _, y_train_no2, y_val_no2 = train_test_split(X_scaled, y_no2, test_size=0.2, random_state=42)

    # Define models and hyperparameter grids
    models = {
        'SVR': {
            'model': SVR(kernel='rbf'),
            'param_grid': {
                'C': [0.1, 1.0],
                'epsilon': [0.1, 1.0],
                'gamma': ['scale']
            },
            'loss': 'epsilon-insensitive'
        },
        'MLP': {
            'model': MLPRegressor(random_state=42, max_iter=1000),
            'param_grid': {
                'hidden_layer_sizes': [(50,), (100,)],
                'alpha': [0.001],
                'learning_rate_init': [0.001]
            },
            'loss': 'Mean Squared Error'
        },
        'KNN': {
            'model': KNeighborsRegressor(),
            'param_grid': {
                'n_neighbors': [3, 5],
                'weights': ['uniform'],
                'p': [2]
            },
            'loss': 'Mean Absolute Error'
        },
        'RandomForest': {
            'model': RandomForestRegressor(random_state=42),
            'param_grid': {
                'n_estimators': [50],
                'max_depth': [None],
                'min_samples_split': [2]
            },
            'loss': 'Mean Squared Error'
        },
        'GradientBoosting': {
            'model': GradientBoostingRegressor(random_state=42),
            'param_grid': {
                'n_estimators': [50],
                'learning_rate': [0.1],
                'max_depth': [3]
            },
            'loss': 'Least Squares'
        }
    }

    # Store best models and scores
    best_o3_model = None
    best_o3_score = float('inf')
    best_o3_params = None
    best_o3_model_name = None
    best_no2_model = None
    best_no2_score = float('inf')
    best_no2_params = None
    best_no2_model_name = None

    # Train and evaluate models for OZONE
    for name, config in models.items():
        grid_search = GridSearchCV(
            config['model'],
            config['param_grid'],
            cv=5,
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train_o3)
        model = grid_search.best_estimator_
        mae = evaluate_model(model, X_val, y_val_o3)
        print(f'OZONE - {name}: Best Params: {grid_search.best_params_}, Validation MAE: {mae}, Loss: {config["loss"]}')
        if mae < best_o3_score:
            best_o3_score = mae
            best_o3_model = model
            best_o3_params = grid_search.best_params_
            best_o3_model_name = name

    # Train and evaluate models for NO2
    for name, config in models.items():
        grid_search = GridSearchCV(
            config['model'],
            config['param_grid'],
            cv=5,
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train_no2)
        model = grid_search.best_estimator_
        mae = evaluate_model(model, X_val, y_val_no2)
        print(f'NO2 - {name}: Best Params: {grid_search.best_params_}, Validation MAE: {mae}, Loss: {config["loss"]}')
        if mae < best_no2_score:
            best_no2_score = mae
            best_no2_model = model
            best_no2_params = grid_search.best_params_
            best_no2_model_name = name

    # Train final models on full data
    best_o3_model.fit(X_scaled, y_o3)
    best_no2_model.fit(X_scaled, y_no2)

    # Ensure models directory exists
    ensure_directory_exists('.Source/models')

    # Save models
    models_dict = {
        'o3_model': best_o3_model,
        'no2_model': best_no2_model
    }
    with open('.Source/models/advanced_model.pkl', 'wb') as f:
        pickle.dump(models_dict, f)
    print('Advanced models saved as models/advanced_model.pkl')

    # For report: Print best model details
    print("\nBest Model Details for Report:")
    print(f"OZONE - Best Model: {best_o3_model_name}, MAE: {best_o3_score}, Params: {best_o3_params}")
    print(f"NO2 - Best Model: {best_no2_model_name}, MAE: {best_no2_score}, Params: {best_no2_params}")

if __name__ == "__main__":
    train_advanced_model()