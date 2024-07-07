from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

def evaluate_model(model, X_test, y_test, grid_search):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    best_params = grid_search.best_params_

    evaluation_metrics = {
        "model": model,
        "accuracy": accuracy,
        "conf_matrix": conf_matrix,
        "classification_report": classification_rep,
        "roc_auc": roc_auc,
        "best_params": best_params
    }

    return evaluation_metrics
