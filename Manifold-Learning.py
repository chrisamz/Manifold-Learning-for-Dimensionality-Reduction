# manifold_learning.py

"""
Manifold Learning Module for Dimensionality Reduction

This module contains functions for implementing manifold learning techniques to reduce the dimensionality
of high-dimensional data and visualize the results.

Techniques Used:
- Isomap
- Locally Linear Embedding (LLE)
- t-Distributed Stochastic Neighbor Embedding (t-SNE)
- Uniform Manifold Approximation and Projection (UMAP)

Libraries/Tools:
- scikit-learn
- umap-learn
- matplotlib
"""

import pandas as pd
import numpy as np
from sklearn.manifold import Isomap, LocallyLinearEmbedding, TSNE
import umap
import matplotlib.pyplot as plt
import joblib

class ManifoldLearning:
    def __init__(self):
        """
        Initialize the ManifoldLearning class.
        """
        self.models = {
            'isomap': Isomap(n_components=2),
            'lle': LocallyLinearEmbedding(n_components=2),
            'tsne': TSNE(n_components=2),
            'umap': umap.UMAP(n_components=2)
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

    def fit_transform(self, data, method):
        """
        Fit and transform the data using the specified manifold learning method.
        
        :param data: DataFrame, input data
        :param method: str, manifold learning method to use
        :return: array, transformed data
        """
        model = self.models.get(method)
        if model is None:
            raise ValueError(f"Method {method} is not defined.")
        transformed_data = model.fit_transform(data)
        self.reduced_data[method] = transformed_data
        return transformed_data

    def save_model(self, method, filepath):
        """
        Save the fitted model to a file.
        
        :param method: str, manifold learning method used
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
        
        :param method: str, manifold learning method used
        :param filepath: str, path to load the model from
        """
        model = joblib.load(filepath)
        self.models[method] = model
        print(f"{method} model loaded from {filepath}")

    def plot_results(self, method, labels=None):
        """
        Plot the results of the manifold learning transformation.
        
        :param method: str, manifold learning method used
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
    model_filepath = 'models/manifold_model.pkl'
    method = 'umap'  # Choose from 'isomap', 'lle', 'tsne', 'umap'

    manifold = ManifoldLearning()

    # Load preprocessed data
    data = manifold.load_data(preprocessed_data_filepath)

    # Fit and transform the data using the specified manifold learning method
    transformed_data = manifold.fit_transform(data, method)

    # Save the fitted model
    manifold.save_model(method, model_filepath)

    # Plot the results
    manifold.plot_results(method)
    print("Manifold learning completed and results plotted.")
