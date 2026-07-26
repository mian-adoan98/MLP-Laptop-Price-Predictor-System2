# Model Training unilities: later phase after feature engineering
# Implement dependencies to build abstraction
from abc import ABC, abstractmethod

# Implement regression metrics to track model's performance 
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score

# Import libraries for model building
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from typing import Type


# Implement class ModelDeveloper 
class ModelDeveloper(ABC):
    # Abstract Method 1: train the model 
    @abstractmethod
    def train(self, xtrain: pd.DataFrame, ytrain: pd.DataFrame):
        pass 

    # Abstract Method 2: evaluate the model using any metrics
    @abstractmethod
    def evaluate(self, xtest:pd.DataFrame, ypred: np.ndarray, ytest: np.ndarray, metric: str) -> pd.DataFrame:
        """
        Params:
        - ypred: number of predictions after model being trained 
        - ytest: the number of testing samples of target feature
        - metrics: parameter that highlights accuracy of model's performance
        
        The fundamental metrics to evaluate a regression models contains: 
            - mean_squared_error 
            - mean_absolute_value_error
            - r2-score

        Return:
        a dataframe of number of metrics for evaluated model     
        """
        pass

# Implement class RegresModel
class RegresModelDeveloper(ModelDeveloper):
    # Initialise attributes for this class 
    def __init__(self, model):
        self.model = model          # model: an object from RegresModel that contains a selected model 
        self.model_perform_hist = []
        self.testing_sets = {}
        self.xtrain = None
        self.feature, self.feature_name  = None, None

        # Create a model performance overview dataframe
        self.model_perform_ds = pd.DataFrame()
        self.model_perform_ds[f"Regression Metrics ({self.feature_name})"] = ["MeanSquared", "MeanAbsolute", "R2Score"]

        # attributes to track model's performance 
        self.msq_errs = []
        self.msq_abs_errs = []
        self.r2scores = []

        # evaluate test trials: for clear overview of the performance
        self.eval_test_trial = 0

    # Method 1: train the model on x & y training sets 
    def train(self, xtrain: pd.DataFrame, ytrain: pd.DataFrame, feature: list): 
        # Train the linear regression model 
        self.model.fit(xtrain[feature], ytrain)

        # Set the xtrain in memory
        self.xtrain = xtrain[feature]
        self.feature = feature
        self.feature_name = feature[0]

    # Method 2: evaluate the model on y testing sets 
    def evaluate(self, xtest:pd.DataFrame, 
                 ypred: np.ndarray, ytest: np.ndarray) -> pd.DataFrame:
        # Make prediction on price 
        xtest = xtest[self.feature]             # x testing set with only 1 feature in selection 
        ypred = self.model.predict(xtest)

        # Store the x & y testing sets in a list 
        if (xtest not in self.testing_sets.values()) and (ytest not in self.testing_sets.values()): 
            self.testing_sets["Xtest"] = xtest
            self.testing_sets["Ytest"] = ytest

        # Evaluate the regression metrics on predicted values of y & compare with initial y testing set
        mse = mean_squared_error(ytest, ypred)
        mae = mean_absolute_error(ytest, ypred)
        r2score = r2_score(ytest, ypred)

        # Store the regression metrics in a list 
        self.msq_errs.append(mse)
        self.msq_abs_errs.append(mae)
        self.r2scores.append(r2score)

        # Create an overview of the model with each test trial 
        self.eval_test_trial += 1
        trial_df = pd.DataFrame()
        trial_df[f"Trial {self.eval_test_trial}"] = [mse, mae, r2score]
        self.model_perform_ds = pd.concat([self.model_perform_ds, trial_df], axis=1)

        return self.model_perform_ds

    # Method 3: visualize the model's performance 
    def visualize(self, coordn: tuple = (10,6)):
        # Extract testings sets to build visual relationship 
        xtest = self.testing_sets["Xtest"]
        ytest = self.testing_sets["Ytest"]

        ## CRITICAL ISSUE
        ypred = self.model.predict(xtest)  

        """The feature names should match those that were passed during fit.
        Feature names unseen at fit time:"""

        # Create visualization 
        plt.figure(figsize=coordn)
        plt.scatter(xtest[self.feature], ytest, color="purple")
        plt.plot(xtest[self.feature], ypred, color="black")

        # Add plot details 
        plt.xlabel(f"{self.feature_name}")
        plt.ylabel("Sale Price ($)")
        plt.title(f"Relationship: {self.feature_name} vs Price ")

        # Add layout
        plt.grid(visible=True)

    # Method 4: manipulate model performance df 
    def manipulate_df(self, operation: str):
        # Select operation
        if operation == "remove":
            feature_cols = self.model_perform_ds.columns
            self.model_perform_ds = self.model_perform_ds.drop(columns=[column for column in 
                                                                feature_cols if "Trial" in column], axis=1)
            

# Test Environment
if __name__ == "__main__":
    pass 