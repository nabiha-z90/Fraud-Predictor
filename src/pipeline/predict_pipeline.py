import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        # Paths to saved model and preprocessor
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

    def predict(self, features: pd.DataFrame):
        """
        Predict whether transactions are fraud or not.
        """
        try:
            # Load pre-trained model and preprocessor
            model = load_object(file_path=self.model_path)
            preprocessor = load_object(file_path=self.preprocessor_path)

            # Preprocess the input features
            data_scaled = preprocessor.transform(features)

            # Predict fraud (1) or not (0)
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, step: int, type: str, amount: float,
                 nameOrig: str, oldbalanceOrg: float, newbalanceOrig: float,
                 nameDest: str, oldbalanceDest: float, newbalanceDest: float):
        self.step = step
        self.type = type
        self.amount = amount
        self.nameOrig = nameOrig
        self.oldbalanceOrg = oldbalanceOrg
        self.newbalanceOrig = newbalanceOrig
        self.nameDest = nameDest
        self.oldbalanceDest = oldbalanceDest
        self.newbalanceDest = newbalanceDest

    def get_data_as_data_frame(self):
        """
        Converts the input data into a pandas DataFrame for prediction.
        """
        try:
            data_dict = {
                "step": [self.step],
                "type": [self.type],
                "amount": [self.amount],
                "nameOrig": [self.nameOrig],
                "oldbalanceOrg": [self.oldbalanceOrg],
                "newbalanceOrig": [self.newbalanceOrig],
                "nameDest": [self.nameDest],
                "oldbalanceDest": [self.oldbalanceDest],
                "newbalanceDest": [self.newbalanceDest],
            }
            return pd.DataFrame(data_dict)

        except Exception as e:
            raise CustomException(e, sys)
