import time

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from src.model_evaluation import evaluate_model

def get_models():
    return {
        "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(random_state=42, class_weight='balanced'),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(random_state=42, class_weight='balanced')
    }


def get_params():
    return {
        "Decision Tree": {
            'classifier__criterion': ['gini', 'entropy'],
            'classifier__max_depth': [None, 10, 20, 30]
        },
        "Random Forest": {
            'classifier__n_estimators': [8, 16, 32, 64, 128, 256],
            'classifier__max_depth': [None, 10, 20, 30]
        },
        "Gradient Boosting": {
            'classifier__learning_rate': [0.1, 0.01, 0.05, 0.001],
            'classifier__subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
            'classifier__n_estimators': [8, 16, 32, 64, 128, 256]
        },
        "Logistic Regression": {
            'classifier__C': [0.01, 0.1, 1, 10, 100],
            'classifier__max_iter': [5000]
        }
    } 


def train_and_evaluate_models(X_train, y_train, X_val, y_val, preprocessor):
    models = get_models()
    params = get_params()
    model_report = {}

    for model_name in tqdm(models, desc="Evaluating Models"):
        start_time = time.time()
        pipeline = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', models[model_name])
        ])
        
        grid_search = GridSearchCV(estimator=pipeline, param_grid=params.get(model_name, {}), scoring='roc_auc', cv=5, verbose=10)
        
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_

        model_report[model_name] = evaluate_model(best_model, X_val, y_val,grid_search)

        end_time = time.time()
        print(f"{model_name} evaluation completed in {end_time - start_time:.2f} seconds")

    return model_report
