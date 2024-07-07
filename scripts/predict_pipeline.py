from sklearn.metrics import accuracy_score

# set the threshold for prediction
def predict(model, X_test, y_test, threshold=0.5):
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Assuming binary classification
    y_pred = (y_pred_proba >= threshold).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    return y_pred, y_pred_proba, accuracy