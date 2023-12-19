import os 
import sys
import pandas as pd 
from src.exception import CustomException
from src.logger import logging

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from dataclasses import dataclass

from src.utils import save_object,evaluate_model


@dataclass 
class ModelTrainerConfig:
    train_model_file_path=os.path.join("artifacts","model.pkl")


class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_training(self,train_arr,test_arr):
        try:
            logging.info("Splitting Train and Test input Data")
            x_train,y_train,x_test,y_test=(
                train_arr[:,:,-1],
                train_arr[:,-1],
                test_arr[:,:,-1],
                test_arr[:,-1]
            )

            models={
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Gredient Boosting": GradientBoostingRegressor(),
                "KNN Regressor":KNeighborsRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "GredientBoost Regressor":GradientBoostingRegressor(),
                "XGBoost Regressor":XGBRegressor(),
                'Linear Regression':LinearRegression(),
                "CatBoost Regressor":CatBoostRegressor(verbose=True)
            
            }

            model_report:dict=evaluate_model(x_train=x_train,y_train=y_train,
                                             x_test=x_test,y_test=y_test,
                                             models=models)
            
            best_model_score=max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]


            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No Best model Found")
            
            logging.info(f"Best model found on both Train dataset and Test dataset is {best_model_name}")

            save_object(file_path=self.model_trainer_config.train_model_file_path,file_object=best_model)

            predicted=best_model.predict(x_test)
            r2_square=r2_score(y_test,predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e,sys)

    

