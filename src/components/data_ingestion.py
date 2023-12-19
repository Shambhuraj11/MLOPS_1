import sys
import os 
from src.exception import CustomException
from src.logger import logging
import pandas as pd 
from sklearn.model_selection import train_test_split
from dataclasses import dataclass ###


@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join("artifact","train.csv")  # No need to define inside init as we are using dataclass decorator
    test_data_path: str=os.path.join("artifact","test.csv")
    raw_data_path: str=os.path.join("artifact","raw.csv")

    
class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered in the Data Ingestion Component ")
        try:
            df=pd.read_csv("notebook\StudentsPerformance.csv")
            logging.info("Exported or Read Dataset as Dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("Train Test Split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Ingestion of the data is Completed")

            return (self.ingestion_config.train_data_path,
                    self.ingestion_config.test_data_path)


        except Exception as e:
            raise CustomException(e,sys)

