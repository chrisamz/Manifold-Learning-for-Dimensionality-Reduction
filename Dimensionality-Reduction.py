# dimensionality_reduction.py

"""
Dimensionality Reduction Module for High-Dimensional Data

This module contains functions for implementing various dimensionality reduction techniques
to transform high-dimensional data into a lower-dimensional space.

Techniques Used:
- Principal Component Analysis (PCA)
- Independent Component Analysis (ICA)
- Singular Value Decomposition (SVD)
- Linear Discriminant Analysis (LDA)

Libraries/Tools:
- scikit-learn
- matplotlib
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
import joblib

class DimensionalityReduction:
    def __init__(self):
        """
        Initialize the DimensionalityReduction class.
        """
        self.models = {
            'pca': PCA(n_components=2),
            'ica': FastICA(n_components=2),
            'svd': TruncatedSVD(n_components=2),
            'lda': LDA(n_components=2)
        }
        self.reduced_data = {}

    def load_data(self, filepath):
        """
        Load preprocessed data from a CSV file.
        
        :param filepath: str, path to the CSV file
        :return: DataFrame, loaded data
        """
        data = pd.read_csv(filepath)
        return data

    def fit_transform(self, data, method, target=None):
        """
        Fit and transform the data using the specified dimensionality reduction method.
        
        :param data: DataFrame, input data
        :param method: str, dimensionality reduction method to use
        :param target: Series, target labels for supervised methods (optional)
        :return: array, transformed data
        """
        model = self.models.get(method)
        if model is None:
            raise ValueError(f"Method {method} is not defined.")
        if method == 'lda' and target is not None:
            transformed_data = model.fit_transform(data, target)
        else:
            transformed_data = model.fit_transform(data)
        self.reduced_data[method] = transformed_data
        return transformed_data

    def save_model(self, method, filepath):
        """
        Save the fitted model to a file.
        
        :param method: str, dimensionality reduction method used
        :param filepath: str, path to save the model
        """
        model = self.models.get(method)
        if model is None:
            raise ValueError(f"Method {method} is not defined.")
        joblib.dump(model, filepath)
        print(f"{method} model saved to {filepath}")

    def load_model(self, method, filepath):
        """
        Load a fitted model from a file.
        
        :param method: str, dimensionality reduction method used
        :param filepath: str, path to load the model from
        """
        model = joblib.load(filepath)
        self.models[method] = model
        print(f"{method} model loaded from {filepath}")

    def plot_results(self, method, labels=None):
        """
        Plot the results of the dimensionality reduction transformation.
        
        :param method: str, dimensionality reduction method used
        :param labels: array, labels for coloring the plot (optional)
        """
        transformed_data = self.reduced_data.get(method)
        if transformed_data is None:
            raise ValueError(f"No reduced data found for method {method}.")

        plt.figure(figsize=(8, 6))
        if labels is not None:
            plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c=labels, cmap='viridis', s=50)
        else:
            plt.scatter(transformed_data[:, 0], transformed_data[:, 1], s=50)
        plt.title(f'{method.upper()} Result')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.colorbar()
        plt.show()

if __name__ == "__main__":
    preprocessed_data_filepath = 'data/processed/preprocessed_high_dimensional_data.csv'
    model_filepath = 'models/dimensionality_reduction_model.pkl'
    method = 'pca'  # Choose from 'pca', 'ica', 'svd', 'lda'
    target_filepath = 'data/processed/target_labels.csv'  # Needed for LDA

    dim_reduction = DimensionalityReduction()

    # Load preprocessed data
    data = dim_reduction.load_data(preprocessed_data_filepath)
    target = pd.read_csv(target_filepath).values.ravel() if method == 'lda' else None

    # Fit and transform the data using the specified dimensionality reduction method
    transformed_data = dim_reduction.fit_transform(data, method, target)

    # Save the fitted model
    dim_reduction.save_model(method, model_filepath)

    # Plot the results
    dim_reduction.plot_results(method, labels=target)
    print("Dimensionality reduction completed and results plotted.")
