# Data Analysis Utilities

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

# Feature Analysis
# Function 1: build a one feature dataset
def build_feature_ds(dataset: pd.DataFrame, feature: str, orientation: str = None) -> pd.DataFrame:
    # Check the distribution of feature: Brand
    feature_data = dataset[feature]
    
    # Choose feature orientation: transposed (T), not transposed(none)
    if orientation == "T": 
        feature_ds = feature_data.value_counts().to_frame()
        feature_ds = feature_ds.transpose()
    else: 
        feature_ds = feature_data.value_counts().to_frame()

    return feature_ds

# Function 2: build a diagnostic feature dataset
def build_diagnositc_feature_ds(dataset: pd.DataFrame) -> pd.DataFrame:
    # Initialize a new dataset called diagnositc dataset
    diagnostic_ds = pd.DataFrame()
    feature_lst = dataset.columns

    # Build an algorithm for constructing a diagnostic dataset only categorical features 
    diagnostic_ds["FeatureNames"] = [feature for feature in feature_lst]
    diagnostic_ds["DataTypes"] = [dataset[feature].dtype for feature in feature_lst]
    diagnostic_ds["NullValues"] = [dataset[feature].isnull().sum() for feature in feature_lst]
    diagnostic_ds["NumDistinctValues"] = [dataset[feature].nunique() for feature in feature_lst]
    diagnostic_ds["UniqueValues"] = [dataset[feature].unique() for feature in feature_lst]

    # Give brief summaries of number of numerical features & categorical features 
    return diagnostic_ds

# Function 3: Correlation Analysis
def correlation_analysis(dataset: pd.DataFrame, features: list = None):
    # Select the type of correlation analysis
    if features is None: 
        # Create a correlated dataset
        corr_ds = dataset.corr()

        # Create a correlation map for the entire dataset 
        plt.figure(figsize=(14,8))
        sns.heatmap(corr_ds, annot=True)
        plt.title(f"Correlation Map") 
        return 

    # If features are provided, validate them 
    missing = [f for f in features if f not in dataset.columns]
    if missing: 
        raise ValueError(f"Features not found in dataset: {missing}")
    
    # Create a specific feature-based correlated dataset
    corr_ds = dataset.corr()
    spec_corr_ds = corr_ds[features]

    # Create a correlated map for given input features
    plt.figure(figsize=(14,8))
    sns.heatmap(spec_corr_ds, annot=True)
    plt.title(f"Correlation Map({features})") 
    return 

# Test Environment
if __name__ == "__main__":
    pass