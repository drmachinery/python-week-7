# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load the Iris dataset
try:
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    
    print("Dataset loaded successfully!")
    print(f"Dataset shape: {df.shape}")
    
except Exception as e:
    print(f"Error loading dataset: {e}")

# Display first few rows
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Explore data structure
print("\nDataset information:")
print(df.info())

print("\nData types:")
print(df.dtypes)

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Clean the dataset (no missing values in Iris dataset, but demonstrating the process)
if df.isnull().sum().sum() > 0:
    # Fill numerical columns with mean and categorical with mode
    for column in df.columns:
        if df[column].dtype in ['int64', 'float64']:
            df[column].fillna(df[column].mean(), inplace=True)
        else:
            df[column].fillna(df[column].mode()[0], inplace=True)
    print("Missing values handled.")
else:
    print("No missing values found. Dataset is clean.")

