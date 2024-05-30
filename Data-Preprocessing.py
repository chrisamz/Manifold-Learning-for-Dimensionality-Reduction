# data_preprocessing.py

"""
Data Preprocessing Module for Manifold Learning for Dimensionality Reduction

This module contains functions for collecting, cleaning, normalizing, and preparing
high-dimensional data for further analysis and modeling.

Techniques Used:
- Data cleaning
- Normalization
- Feature extraction
- Handling missing data

Libraries/Tools:
- pandas
- numpy
- scikit-learn
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

class DataPreprocessing:
    def __init__(self):
        """
        Initialize the DataPreprocessing class.
        """
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')

    def load_data(self, filepath):
        """
        Load data from a CSV file.
        
        :param filepath: str, path to the CSV file
        :return: DataFrame, loaded data
        """
        data = pd.read_csv(filepath)
        return data

    def clean_data(self, data):
        """
        Clean the data by removing duplicates and handling missing values.
        
        :param data: DataFrame, input data
        :return: DataFrame, cleaned data
        """
        data = data.drop_duplicates()
        data = pd.DataFrame(self.imputer.fit_transform(data), columns=data.columns)
        return data

    def normalize_data(self, data):
        """
        Normalize the data using standard scaling.
        
        :param data: DataFrame, input data
        :return: DataFrame, normalized data
        """
        data = pd.DataFrame(self.scaler.fit_transform(data), columns=data.columns)
        return data

    def preprocess(self, filepath):
        """
        Execute the full preprocessing pipeline.
        
        :param filepath: str, path to the input data file
        :return: DataFrame, preprocessed data
        """
        data = self.load_data(filepath)
        data = self.clean_data(data)
        data = self.normalize_data(data)
        return data

if __name__ == "__main__":
    raw_data_filepath = 'data/raw/high_dimensional_data.csv'
    processed_data_filepath = 'data/processed/preprocessed_high_dimensional_data.csv'

    preprocessing = DataPreprocessing()

    # Preprocess the data
    preprocessed_data = preprocessing.preprocess(raw_data_filepath)
    preprocessed_data.to_csv(processed_data_filepath, index=False)
    print("Data preprocessing completed and saved to 'data/processed/preprocessed_high_dimensional_data.csv'.")
