import os
import sys
import pickle
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException


def save_object(file_path, obj):
    """
    Save any Python object (e.g., model, transformer) using pickle.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Evaluate multiple classification models using F1-score (weighted).
    Primarily optimized for XGBoost, but works for any sklearn-compatible model.
    """
    try:
        report = {}

        for model_name, model in models.items():
            print(f"Training and tuning {model_name}...")

            # Get corresponding hyperparameters
            para = param.get(model_name, {})

            # Use GridSearchCV for hyperparameter tuning
            gs = GridSearchCV(
                estimator=model,
                param_grid=para,
                scoring='f1_weighted',  # directly use F1 scoring
                cv=3,
                n_jobs=-1,
                verbose=1
            )
            gs.fit(X_train, y_train)

            # Update model with best parameters
            best_model = gs.best_estimator_

            # Train with best params on full training data
            best_model.fit(X_train, y_train)

            # Predictions
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            # F1 scores
            train_f1 = f1_score(y_train, y_train_pred, average='weighted')
            test_f1 = f1_score(y_test, y_test_pred, average='weighted')

            # Store results
            report[model_name] = {
                "train_f1": round(train_f1, 4),
                "test_f1": round(test_f1, 4),
                "best_params": gs.best_params_
            }

            print(f"{model_name} → Train F1: {train_f1:.4f}, Test F1: {test_f1:.4f}")

        return report

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Load any saved pickle object from disk.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
