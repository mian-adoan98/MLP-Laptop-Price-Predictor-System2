# Feature Transformation: Utilities
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import FunctionTransformer
from abc import ABC, abstractmethod


# Plot the feature with a scatter plot
def create_plot_distribution(dataset: pd.DataFrame, 
                             in_feature_nm: str, 
                             target_feature_nm: str = "price"): 
    
    in_feature = dataset[in_feature_nm].values
    target_feature = dataset[target_feature_nm].values

    # Show scatter plot 
    plt.scatter(in_feature, target_feature, color="black")

    # Add plot details 
    plt.xlabel(f"{in_feature_nm}")
    plt.ylabel(f"{target_feature_nm}")
    plt.title(f"Diagram: {in_feature_nm} vs {target_feature_nm}")

    plt.show()

# Implement abstract class (see later)
class FeatureTransformer(ABC):
    # Define attributes of the transformer 
    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset
    
    # Abstract method 1
    def transform(self, feature:str, mode:str) -> pd.DataFrame:
        pass 

# Implement class FeatureTransformer
class FunctionTransformer:
    # Define attributes of the transformer 
    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset
    
    # Method 1: apply FunctionTransformer
    def apply_log_transform(self, feature:str) -> pd.DataFrame:
        # Extract feature data from the dataframe
        feature_data = self.dataset[feature]

        # Apply the log transformation to that feature data
        transformer = FunctionTransformer(func=np.log1p)
        feature_transf_data = transformer.fit_transform()

        return feature_transf_data
    
    # Method 2: apply square transformation
    def apply_square_transform(self, feature:str) -> pd.DataFrame:
        # Extract feature data from the dataframe
        feature_data = self.dataset[feature]

        # Apply square transformation
        feature_transf_data = np.square(feature_data)
        
        return feature_transf_data
    
    # Method 3: apply square root transformation
    def apply_square_root(self, feature:str) -> pd.DataFrame:
        # Extract feature data from the dataframe
        feature_data = self.dataset[feature]

        # Apply square transformation
        feature_transf_data = np.sqrt(feature_data)
        
        return feature_transf_data
    
    # Method 4: apply reciprocal transformer
    def apply_reciprocal_transform(self, feature:str) -> pd.DataFrame:
        # Extract feature data from the dataframe
        feature_data = self.dataset[feature]

        # Apply square transformation
        feature_transf_data = np.reciprocal(feature_data)

    # Method 5: apply custom transformer (out of choice)
    def apply_custom_transform(self, feature:str, op_method:str) -> pd.DataFrame:
        # Extract feature data from the dataframe
        feature_data = self.dataset[feature]
        operation_dict = {
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan
        }

        # Select transformer operation method to perform feature transformation
        try:
            feature_transf_data = operation_dict[op_method](feature_data)
        except KeyError:
            raise ValueError(f"Unknown operation method: {op_method}")
            
        return feature_transf_data

# Test environment
if __name__ == "__main__":
    pass