import os
import sys 
import numpy as np 
import pandas as pd 
from src.exception import CustomException
from src.logger import logging
import dill


def save_object(file_path:str,file_object):
    try:
        dir_path=os.path.dirname(file_path) 
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as f:
            dill.dump(file_object,f)
    except Exception as err:
        raise CustomException(err,sys)