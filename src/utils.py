import joblib
import pandas as pd

def save_model(model, file_path):
    """
    Save the trained model to a file.
    """
    joblib.dump(model, file_path)
    print(f"Model saved to {file_path}")

def load_model(file_path):
    """
    Load a trained model from a file.
    """
    model = joblib.load(file_path)
    print(f"Model loaded from {file_path}")
    return model


def save_preprocessor(preprocessor, path):
    """
    Saving the used preprocessor details
    """
    print(f"preprocessor saved to {path}")
    joblib.dump(preprocessor, path)



def load_preprocessor(path):
    """
    Loading preprocessor previously saved
    """
    print(f"preprocessor loaded from {path}")
    return joblib.load(path)

def save_predictions(predictions, prediction_proba, output_path):
    """
    Save the predictions to a CSV file.
    """
    results = pd.DataFrame({
        'predictions': predictions,
        'prediction_proba': prediction_proba
    })
    print(f"Predictions saved to {output_path}")
    results.to_csv(output_path, index=False)