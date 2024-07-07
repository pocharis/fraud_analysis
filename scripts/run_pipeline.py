from predict_pipeline import predict
from src.config import RAW_DATA_PATH, LABELS_DATA_PATH, MODEL_PATH, REPORT_PATH, PROCESSED_DATA_PATH
from src.data_ingestion import load_data, preprocess_transaction_data, split_data
from src.data_preprocessing import create_preprocessor
from src.model_training import train_and_evaluate_models
from src.utils import (
    load_model,
    load_preprocessor,
    save_model,
    save_predictions,
    save_preprocessor,
)


if __name__ == "__main__":
    # Load and preprocess data
    df_transaction = load_data(f'{RAW_DATA_PATH}', f'{LABELS_DATA_PATH}')
    df_transaction = preprocess_transaction_data(df_transaction)
    print("data loaded and pre-preprocessed")

    # save processed data
    df_transaction.to_csv(f'{PROCESSED_DATA_PATH}feature_engineered_data.csv', index=False)

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_transaction)
    print("pre-processed data saved and train-eval-test sets obtained")


    # Preprocess data
    preprocessor = create_preprocessor()
    preprocessor.fit(X_train)
    save_preprocessor(preprocessor, f'{MODEL_PATH}preprocessor.pkl')

    print("model training started")
    # Train and evaluate models
    model_report = train_and_evaluate_models(X_train, y_train, X_val, y_val, preprocessor)

    # Determine the best model
    best_model_name = max(model_report, key=lambda x: model_report[x]["roc_auc"])
    best_model_info = model_report[best_model_name]
    best_model = best_model_info["model"]

    print(f"Best Model: {best_model_name}")
    print(f"Best Model ROC AUC Score: {best_model_info['roc_auc']}")
    print("Best Model Parameters:")
    print(best_model_info["best_params"])
    print(f"Accuracy: {best_model_info['accuracy']}")
    print("Confusion Matrix:")
    print(best_model_info['conf_matrix'])
    print("Classification Report:")
    print(best_model_info['classification_report'])

    # Save the best model
    save_model(best_model, f'models/{best_model_name}_best_model.pkl')

    # load model and prediction artifacts
    preprocessor = load_preprocessor('models/preprocessor.pkl')
    model = load_model(f'models/{best_model_name}_best_model.pkl')

    # Predict and save predictions
    predictions, prediction_proba, accuracy = predict(model, X_test, y_test, threshold=0.4)
    print(f'Test Accuracy: {accuracy}')
    save_predictions(predictions, prediction_proba, f'{REPORT_PATH}{best_model_name}_predictions.csv')

