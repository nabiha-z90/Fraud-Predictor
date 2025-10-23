import os
import sys
from dataclasses import dataclass

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        """
        Trains and evaluates an XGBoost classifier on provided datasets.
        Saves the trained model to artifacts.
        """
        try:
            logging.info("Splitting training and test data")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            # === Define XGBoost Model ===
            model = XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=7,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric="logloss",
                n_jobs=-1
            )

            logging.info("Training XGBoost model...")
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

            # === Predictions ===
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # === Evaluation ===
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)

            train_report = classification_report(y_train, y_train_pred, output_dict=True)
            test_report = classification_report(y_test, y_test_pred, output_dict=True)

            logging.info(f"✅ Training Completed Successfully")
            logging.info(f"Train Accuracy: {train_acc:.4f} | Test Accuracy: {test_acc:.4f}")
            logging.info("\nTrain Classification Report:\n" + classification_report(y_train, y_train_pred))
            logging.info("\nTest Classification Report:\n" + classification_report(y_test, y_test_pred))

            # === Save Model ===
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )

            logging.info(f"Model saved at: {self.model_trainer_config.trained_model_file_path}")

            return {
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
                "train_report": train_report,
                "test_report": test_report,
            }

        except Exception as e:
            logging.error("Error occurred during model training", exc_info=True)
            raise CustomException(e, sys)
