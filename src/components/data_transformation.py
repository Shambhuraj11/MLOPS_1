import numpy as np
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object


@dataclass
class data_transformation_config:
    preprocessor_file_path=os.path.join("artifacts","preprocessor.pkl")


class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config=data_transformation_config()
    
    def get_data_transformation_object(self):
        '''
        This function do the data trnasformation
        '''
        try:
            numerical_features=['reading score', 'writing score']
            categorical_features=['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
            numerical_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())

                ])
            logging.info(f"Numerical column :{numerical_features}")
            
            categorical_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            logging.info(f"Categorical column :{categorical_features}")
            
            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",numerical_pipeline,numerical_features),
                    ("Cat_pipeline",categorical_pipeline,categorical_features)
                ]
            )
            return preprocessor

        except Exception as err:
            raise CustomException(err,sys)
        

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read Train and Test data completed")

            logging.info('Obtaining Preprocessor objects')
            preprocessor_obj=self.get_data_transformation_object()

            target_column='math score'
            numerical_features=['math score', 'reading score', 'writing score']

            input_feature_train_df=train_df.drop(columns=[target_column],axis=1)
            target_feature_train_df=train_df[target_column]

            # logging.info(f"input_feature_train_df.columns: {input_feature_train_df.columns}")
            # logging.info(f"target_feature_train_df.columns: {target_feature_train_df.columns}")


            input_feature_test_df=test_df.drop(columns=[target_column],axis=1)
            target_feature_test_df=test_df[target_column]

            # logging.info(f"input_feature_test_df.columns: {input_feature_test_df.columns}")
            # logging.info(f"target_feature_test_df.columns: {target_feature_test_df.columns}")

            logging.info("applying preprocessing object on Training DataFrame and Testing DataFrame")

            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.fit_transform(input_feature_test_df)
            
            # Concate target column with feature columns
            # print(input_feature_train_arr[1])

            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]
            # print(train_arr[:,-1])

            test_arr=np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]

            logging.info("Saved Preprocessing Object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_file_path,
                file_object=preprocessor_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_file_path
            )

        except Exception as err:
            raise CustomException(err,sys)


