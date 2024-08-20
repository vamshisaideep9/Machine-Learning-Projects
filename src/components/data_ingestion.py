import os
import pandas as pd
import sys
from ..logger import logging

from src.exception import CustmeException  # Corrected the typo
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from ..components.data_transformation import DataTransformation
from ..components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "raw.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data ingestion process started")
        try:
            logging.info("Reading data using pandas from the local system")
            data = pd.read_csv(os.path.join("C:/Users/vamsh/OneDrive/Desktop/ML project/note book/data", "income_cleandata.csv"))
            logging.info("Data reading completed successfully")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info(f"Raw data saved at {self.ingestion_config.raw_data_path}")

            logging.info("Splitting data into train and test sets")
            train_set, test_set = train_test_split(data, test_size=0.3, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info(f"Train data saved at {self.ingestion_config.train_data_path}")
            logging.info(f"Test data saved at {self.ingestion_config.test_data_path}")

            logging.info("Data ingestion completed successfully")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            logging.error("An error occurred during the data ingestion process")
            raise CustmeException(e, sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    train_arr, test_arr = data_transformation.initiate_data_transformation(train_data_path, test_data_path)[:2]

    model_trainer = ModelTrainer()
    print(model_trainer.inititate_model_trainer(train_arr, test_arr))
