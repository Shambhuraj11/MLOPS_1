import logging
import os
from datetime import datetime
from src.exception import CustomException
import sys 

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path=os.path.join(os.getcwd(),"logs",LOG_FILE.split(".")[0])
os.makedirs(logs_path,exist_ok=True)

LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s -%(levelname)s - %(message)s",
    level=logging.INFO
)

# if __name__=="__main__":
#     try:
#         a=1/0
#         logging.info("Logging has started")

#     except Exception as e:
#         logging.info("Divide by Zero error")
#         raise CustomException(e,sys)