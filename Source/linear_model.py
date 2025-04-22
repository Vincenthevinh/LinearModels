import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
import pickle
from utils import load_data, extract_features, scale_features, evaluate_model

def train_linear_model():
    """
    Train linear models (LinearRegression, Ridge, Lasso, SVR, SGDRegressor) for OZONE and NO2 using voltage outputs only.
    Save best models to linear_model.pkl.
    """
    # Load dataset
    train_df = load_data('./Source/data/train.csv')

    # Define features (voltage outputs only)
    features = ['o3op1', 'o3op2', 'no2op1', 'no2op2']
    X_train = train_df[features]
    y_train_o3 = train_df['OZONE']
    y_train_no2 = train_df['NO2']

    # Scale features
    X_train_scaled = scale_features(X_train)

    # Initialize models
    models = {
        'LinearRegression': {
            'model': LinearRegression(),
            'param_grid': {},
            'loss': 'Least Squares'
        },
        'Ridge': {
            'model': Ridge(),
            'param_grid': {'alpha': [0.1, 1.0, 10.0]},
            'loss': 'Least Squares + L2'
        },
        'Lasso': {
            'model': Lasso(max_iter=10000),
            'param_grid': {'alpha': [0.1, 1.0, 10.0]},
            'loss': 'Least Squares + L1'
        },
        'SVR_Linear': {
            'model': SVR(kernel='linear'),
            'param_grid': {'C': [0.1, 1.0], 'epsilon': [0.1, 1.0]},
            'loss': 'epsilon-insensitive'  # Changed from ε-insensitive
        },
        'SGDRegressor_MAE': {
            'model': SGDRegressor(loss='epsilon_insensitive', epsilon=0, random_state=42, max_iter=1000),
            'param_grid': {'alpha': [0.0001, 0.001], 'learning_rate': ['constant', 'optimal'], 'eta0': [0.01]},
            'loss': 'Approximated MAE'
        }
    }

    # Store results
    results = {'Model': [], 'O3 MAE': [], 'NO2 MAE': [], 'Avg MAE': [], 'Loss Function': [], 'Best Params': []}

    # Train and evaluate models
    from sklearn.model_selection import GridSearchCV
    for name, config in models.items():
        grid_search = GridSearchCV(
            config['model'],
            config['param_grid'],
            cv=5,
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )
        # OZONE
        grid_search.fit(X_train_scaled, y_train_o3)
        best_model_o3 = grid_search.best_estimator_
        mae_o3 = evaluate_model(best_model_o3, X_train_scaled, y_train_o3)
        # NO2
        grid_search.fit(X_train_scaled, y_train_no2)
        best_model_no2 = grid_search.best_estimator_
        mae_no2 = evaluate_model(best_model_no2, X_train_scaled, y_train_no2)
        # Store results
        results['Model'].append(name)
        results['O3 MAE'].append(mae_o3)
        results['NO2 MAE'].append(mae_no2)
        results['Avg MAE'].append((mae_o3 + mae_no2) / 2)
        results['Loss Function'].append(config['loss'])
        results['Best Params'].append(grid_search.best_params_)
        print(f'{name} - O3 MAE: {mae_o3}, NO2 MAE: {mae_no2}, Avg MAE: {(mae_o3 + mae_no2) / 2}, '
              f'Loss: {config["loss"]}, Params: {grid_search.best_params_}')

    # Find best model
    results_df = pd.DataFrame(results)
    best_model_row = results_df.loc[results_df['Avg MAE'].idxmin()]
    print('\nBest Linear Model for Report:')
    print(best_model_row)

    # Train best model
    best_model_name = best_model_row['Model']
    best_params = best_model_row['Best Params']
    if best_model_name == 'LinearRegression':
        best_model_o3 = LinearRegression()
        best_model_no2 = LinearRegression()
    elif best_model_name == 'Ridge':
        best_model_o3 = Ridge(**best_params)
        best_model_no2 = Ridge(**best_params)
    elif best_model_name == 'Lasso':
        best_model_o3 = Lasso(max_iter=10000, **best_params)
        best_model_no2 = Lasso(max_iter=10000, **best_params)
    elif best_model_name == 'SVR_Linear':
        best_model_o3 = SVR(kernel='linear', **best_params)
        best_model_no2 = SVR(kernel='linear', **best_params)
    elif best_model_name == 'SGDRegressor_MAE':
        best_model_o3 = SGDRegressor(loss='epsilon_insensitive', epsilon=0, random_state=42, max_iter=1000, **best_params)
        best_model_no2 = SGDRegressor(loss='epsilon_insensitive', epsilon=0, random_state=42, max_iter=1000, **best_params)

    # Train final models
    best_model_o3.fit(X_train_scaled, y_train_o3)
    best_model_no2.fit(X_train_scaled, y_train_no2)

    # Save models
    models_dict = {
        'o3_model': best_model_o3,
        'no2_model': best_model_no2
    }
    with open('./Source/models/linear_model.pkl', 'wb') as f:
        pickle.dump(models_dict, f)
    print('Linear models saved as Source/models/linear_model.pkl')