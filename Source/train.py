from linear_model import train_linear_model
from advanced_model import train_advanced_model

def train_all_models():
    """
    Train both linear and advanced models.
    """
    print("Training Linear Model...")
    train_linear_model()
    print("Training Advanced Model (SVR, MLP, KNN, RandomForest)...")
    train_advanced_model()

if __name__ == "__main__":
    train_all_models()