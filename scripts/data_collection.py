# Data Collection
from abc import abstractmethod, ABC

# Import libraries
# Import libraries for data analysis & visualisation
import numpy as np 
import pandas as pd 
import os
import sys

# Define constants for this project 
PROJECT_PATH =  r"C:\Development\03_ML_Engineering\Projects\ML_Projects\Laptop_Price_Prediction"

# Implement class DataCollector
class DataCollector(ABC):
    # Attributes assigned to DataCollector-class
    def __init__(self):
        pass 

    # Abstract methods: collect data
    @abstractmethod
    def collect(self, filename:str) -> pd.DataFrame:
        pass 

    # Abstract method 2: store data in the correct folder 
    @abstractmethod
    def save(self, store_package: tuple[str, pd.DataFrame], folder:str, version_nr:int):
        pass 

# Implement class DataLoader 
class DataLoader(DataCollector):
    # Attributes assigned to DataLoader-class
    def __init__(self, folder: str):
        self.datafolder_path = os.path.join(PROJECT_PATH, "data")
        self.selected_folder = os.path.join(self.datafolder_path, folder)

    # Abstract methods: collect data 
    def collect(self, filename: str) -> pd.DataFrame:
        # Define file path
        file_path = os.path.join(self.selected_folder, filename)

        # Check if file not exists
        if (not os.path.exists(path=file_path)) and filename.endswith(".csv"):
           raise FileExistsError(f"No File in the folder path {file_path}. Please provide the required file (with .csv)")
        
        # Load the dataset from the selected folder location
        dataset = pd.read_csv(file_path)
        dataset = dataset.drop(columns=["Unnamed: 0"], axis=1)
        return dataset

    # Abstract method 2: store data in the correct folder 
    def save(self, store_package: tuple[str, pd.DataFrame], folder:str, version_nr:int):
        # Store package 
        new_filename = store_package[0]
        dataset = store_package[-1]

        # Check if the folder path is already existential
        if folder not in self.selected_folder: 
            # Initialise new file path & build a new renaming for new file
            folder_path = os.path.join(self.selected_folder, folder)
            filename = f"{new_filename}_v{version_nr}.csv"
            new_file_path = os.path.join(folder_path, filename)

            # Check if folder exists
            if not os.path.exists(folder_path):
                os.makedirs(folder_path, exist_ok=True)

            print(f"Dataset {filename} has been stored succesfully in path loc. {folder_path}")
        else: 
            # Initialise new file path by using the existed path of the selected folder
            filename = f"{new_filename}_v{version_nr}.csv"
            new_file_path = os.path.join(self.selected_folder, filename)

            # Save dataset into folder 
            dataset.to_csv(new_file_path)
            print(f"Dataset {filename} has been stored succesfully in path loc. {self.selected_folder}")
    


# Test Environment
if __name__ == "__main__":
    # Test 1: Print the selected folder path to verify the path construction
    dataloader = DataLoader("processed")
    selected_folder_path = dataloader.collect(filename="laptop_price_dataset_v2.csv")
    print(f"Path name: {selected_folder_path}")