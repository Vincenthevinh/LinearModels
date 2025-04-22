from linear_model import train_linear_model
from advanced_model import train_advanced_model
from utils import ensure_directory_exists

def train_all_models():
    """
    Train both linear and advanced models.
    """
    # Ensure models directory exists
    ensure_directory_exists('./models')
    
    print("Training Linear Model...")
    train_linear_model()
    print("\nTraining Advanced Model...")
    train_advanced_model()

if __name__ == "__main__":
    train_all_models()