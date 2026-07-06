# Features Utilities: Feature Selection, Transformation, Extraction etc. 

# Import abstract functionalities to build abstract classes
from abc import ABC, abstractmethod

# Import libraries for feature engineering and manipulation
import os 
import numpy as np 
import pandas as pd 

# Import dependencies from sklearn 
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Initialise project constrains
PROJECT_PATH =  r"C:\Development\03_ML_Engineering\Projects\ML_Projects\Laptop_Price_Prediction"
DATA_FOLDER_PATH = os.path.join(PROJECT_PATH, "data")

# Import abstract class FeatureEngineering
class FeatureEngineering(ABC):
    # Defin the attributes for FeatureEngineering-class 
    def __init__(self, dataset: pd.DataFrame, encoder:str):
        self.dataset = dataset
        self.encoder = encoder 

    # Abstract method 1: transform the selected feature
    @abstractmethod
    def transform(self, feature: str) -> pd.DataFrame:
        pass 
 

# Feature Selection: implement a class FeatureSelector
class FeatureSelector:
    # Define attributes for FeatureSelector class 
    def __init__(self, dataset: pd.DataFrame):
         self.dataset = dataset
    
    # Method 1: include features
    def include(self, features:list) -> pd.DataFrame:
        pass 

    # Method 2: exclude features from dataset
    def exclude(self, features: list) -> pd.DataFrame:
        # include features 
        new_dataset = self.dataset.drop(columns=features, axis=1)
        return new_dataset

# Feature Encoding: implement a class FeatureEncoder 
class FeatureEncoder(FeatureEngineering):
    # Define attributes for FeatureSelector class 
    def __init__(self, dataset: pd.DataFrame, encoder:str):
        self.dataset = dataset
        self.encoder = encoder 
        self.feature_ds_folder_path = os.path.join(DATA_FOLDER_PATH, "feature_ds")

        self.specified_encoding = ["OneHotEncoding", "LabelEncoding", "TargetEncoding", "FrequencyEncoding"]

    # Method 1: transform the feature 
    def transform(self, feature:str) -> pd.DataFrame:
        # Transform the nominal feature using a specified encoder
        if self.encoder == "OneHotEncoding": 
            # Extract a feature that is nominal & convert in 2D-array
            nominal_feature_data = self.dataset[[feature]]
            nominal_feature_name = nominal_feature_data.columns

            # Encode the feature data using OneHotEncoder
            encoder = OneHotEncoder(sparse_output=False)
            encoded_data = encoder.fit_transform(nominal_feature_data.values)

        elif self.encoder == "LabelEncoding":
            # Extract a feature that is nominal & convert into 1D-array
            nominal_feature_data = self.dataset[feature]
            nominal_feature_name = nominal_feature_data.columns

            # Encode the feature data using LabelEncoder
            encoder = LabelEncoder(sparse_output=True)
            encoded_data = encoder.fit_transform(nominal_feature_data.values)

        else: 
            print(f"Other encoding method: {self.encoder}")
            
        # Check if the encoding is specified 
        if self.encoder in self.specified_encoding:
            # Create a encoded dataframe using the encoded data
            encoded_nominal_df = pd.DataFrame(
                encoded_data,
                columns=encoder.get_feature_names_out()
            )


            # Remove inconsistencies in the columns of the data
            adapt_feature_names = [name.replace("x0_", "") for name in encoded_nominal_df.columns]
            encoded_nominal_df.columns = adapt_feature_names
            
            # Combined the encoded dataframe with the nominal feature dataframe
            feature_combined_ds = pd.concat([nominal_feature_data, encoded_nominal_df], axis=1) 
            return feature_combined_ds
        else: 
            specified_encoding_meth = "BinaryEncoding"
            feature_data = self.dataset[feature].apply(lambda x: 1 if x in ["Y", "Yes"] else 0)
            print(f"Binary Encoding is succesful for Feature {feature}")
            return feature_data
    
    def save_ds(self, feature:str , file_idx: int):
        # Check which specific encoding is selected
        if self.encoding in self.specified_encoding:
            # Retrieve the combined encoded dataset 
            combined_encoded_df = self.transform(feature)

            # Check if feature dataset folder path exists: if not, make a new one
            if not os.path.exists(self.feature_ds_folder_path):
                os.makedirs(self.feature_ds_folder_path, exist_ok=True)

            # Save combined dataset in feature dataset folder
            file_path = os.path.join(self.feature_ds_folder_path, f"{feature}_encoded_ds_{file_idx}.csv")
            combined_encoded_df.to_csv(file_path)
        else:
            # Binary feature 
            binary_feature_data = self.transform(feature)

            # Save combined dataset in feature dataset folder
            file_path = os.path.join(self.feature_ds_folder_path, f"{feature}_encoded_ds_{file_idx}.csv")
            binary_feature_data.to_csv(file_path)

# Implement class FeatureTransformation
class FeatureTransforer(FeatureEngineering):
    pass 

# Test Environment
if __name__ == "__main__":
    pass