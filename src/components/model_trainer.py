import os
import sys
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts',"model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            x_train,x_test,y_train,y_test=(
                train_array[:,:-1],
                test_array[:,:-1],
                train_array[:,-1],
                test_array[:,-1],
            )
            models={
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors": KNeighborsRegressor(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False),
                "Adaboost": AdaBoostRegressor()
            }
            model_report:dict=evaluate_models(x_train=x_train,x_test=x_test,y_train=y_train,y_test=y_test,models=models)
            
            #to get the best model score
            best_model_score=max(sorted(model_report.values()))
            #to get the best model name from dict
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]
            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and test dataset")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted=best_model.predict(x_test)
            r2_square=r2_score(y_test,predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e,sys)