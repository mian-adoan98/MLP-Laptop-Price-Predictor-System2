# Models
# Import pre-build models from sklearn
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor


# Import libraries for model building
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from typing import Type
""" Models that are used to train with data and help to build prediction on price sale are: 
    - Regression Models 
    - Linear Neural Network models
    - Decision Tree models 
"""
# Implement class RegressionModel
class RegressionModel:
    # Define attributes for the RegressionModel-class
    def __init__(self):
        self.active_model = False
        self.selected_model = None 

    # Define attribute 
    def select_model(self, model_name: str, alpha: float = 0.1):
        # Set model name into lower cases
        model_name = model_name.lower()
        # Select the model
        if model_name in "linear regression model":
            model = LinearRegression()
            self.active_model = True
            self.selected_model = "linear regression model"
            return model 
        elif model_name in "rigde regression model":
            model = Ridge(alpha=alpha)
            self.active_model = True
            self.selected_model = "ridge regression model"
            return model 
        elif model_name in "lasso regression model":
            model = Lasso(alpha=alpha)
            self.active_model = True
            self.selected_model = "lasso regression model"
            return model 
        elif model_name in "elasticnet regression model":
            model = ElasticNet(alpha=alpha)
            self.active_model = True
            self.selected_model = "elasticnet regression model"
            return model 

        # Display which model has been activated & selected
        print(f"Model: {model_name}")
        print(f"Status: {self.active_model}")

# Test Environment
if __name__ == "__main__":
    pass