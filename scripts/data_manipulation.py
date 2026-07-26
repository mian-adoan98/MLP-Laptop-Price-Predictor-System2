# Data Manipulation & Preparation
from abc import abstractmethod, ABC
from typing import Type

# Import libraries for sklearn
from sklearn.model_selection import train_test_split


# Import libraries for data analysis & visualisation
import numpy as np 
import pandas as pd 
import os
import sys

# Define constants for this project 
PROJECT_PATH =  r"C:\Development\03_ML_Engineering\Projects\ML_Projects\Laptop_Price_Prediction"
DataPackages = Type

# Implement class DataPreparator 
class DataPreparator(ABC): 
    # Attributes assign to DataPreparator-class
    def __init__(self, target_feature: list):
        pass 

    # Abstract Method 1: prepare the data
    @abstractmethod
    def prepare(self, dataset: pd.DataFrame) -> DataPackages:
        pass 

# Implement class DataSplitter 
class DataSplitter(DataPreparator):
    # Define attributes
    def __init__(self, target_feature: list):
        self.target_feature = target_feature
        self.test_size_hist = []
        self.idx = 0

    # Method 1: prepare data
    def prepare(self, dataset: pd.DataFrame, test_size: int) -> tuple[list, list]:
        # Prepare the data 
        XData = dataset.drop(columns=self.target_feature, axis=1)
        YData = dataset[self.target_feature]

        # Split the dataset into training and testing sets
        xtrain, xtest, ytrain, ytest = train_test_split(XData, YData, test_size=test_size, random_state=1234)

        # Group by training and testing sets
        training_set = [xtrain, ytrain]
        testing_set = [xtest, ytest]

        # Store the used test size history
        self.idx += 1
        self.test_size_hist.append((self.idx, test_size))

        return (training_set, testing_set)

    # Method 2: provide summary
    def provide_summary(self, dataset: pd.DataFrame):
        # Retrieve the training testing sets
        training_set, testing_set = self.prepare(dataset, test_size=self.test_size_hist[-1][-1])
        xtrain, ytrain = training_set
        xtest, ytest = testing_set

        # Compute statistical references for X and Y training and testing (in percentage)
        set_perc = lambda x:( x.shape[0]/dataset.shape[0])*100
        training_stats = [set_perc(train_set) for train_set in [xtrain, ytrain]]
        testing_stats = [set_perc(train_set) for train_set in [xtest, ytest]]

        print(f"Training sets", "--"*1000)
        print(f"X Training: {xtrain.shape[0]} ({training_stats[0]:.2f}%)")
        print(f"Y Training: {ytrain.shape[0]} ({training_stats[-1]:.2f}%)")
        print(f"Testing sets", "--"*1000)
        print(f"X Testing: {xtest.shape[0]} ({testing_stats[0]:.2f}%)")
        print(f"Y Testing: {ytest.shape[0]} ({testing_stats[-1]:.2f}%)")


# Test Environment
if __name__ == "__main__":
    pass