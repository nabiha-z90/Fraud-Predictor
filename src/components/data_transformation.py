import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
            categorical_columns = ['type']

            num_pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            logging.info("Preprocessor pipeline created successfully.")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info(f"Reading train and test data from:\nTrain: {train_path}\nTest: {test_path}")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info(f"✅ Train shape: {train_df.shape}, Test shape: {test_df.shape}")

            target_column_name = "isFraud"

            # --- Separate features and target ---
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            preprocessing_obj = self.get_data_transformer_object()

            # --- Apply preprocessing ---
            logging.info("🚀 Starting preprocessing on training data...")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info(f"✅ After preprocessing: Train features shape = {input_feature_train_arr.shape}, "
                         f"Target shape = {target_feature_train_df.shape}")

            # --- FIXED concat step ---
            target_feature_train_arr = np.array(target_feature_train_df).reshape(-1, 1)
            target_feature_test_arr = np.array(target_feature_test_df).reshape(-1, 1)

            train_arr = np.concatenate([input_feature_train_arr, target_feature_train_arr], axis=1)
            test_arr = np.concatenate([input_feature_test_arr, target_feature_test_arr], axis=1)

            logging.info("✅ Data transformation completed successfully.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    transformer = DataTransformation()
    train_path = os.path.join("artifacts", "train.csv")
    test_path = os.path.join("artifacts", "test.csv")
    transformer.initiate_data_transformation(train_path, test_path)
