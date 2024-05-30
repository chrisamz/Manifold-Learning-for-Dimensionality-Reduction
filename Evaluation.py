# evaluation.py

"""
Evaluation Module for Dimensionality Reduction and Manifold Learning

This module contains functions for evaluating the performance of manifold learning
and dimensionality reduction models using appropriate metrics.

Metrics Used:
- Explained variance
- Reconstruction error
- Visualization quality

Libraries/Tools:
- scikit-learn
- pandas
- numpy
- matplotlib
"""

import pandas as pd
import numpy as np
from sklearn.metrics import explained_variance_score, mean_squared_error
import matplotlib.pyplot as plt
import joblib

class ModelEvaluation:
    def __init__(self):
        """
        Initialize the ModelEvaluation class.
        """
        pass

    def load_data(self, filepath):
        """
        Load data from a CSV file.
        
        :param filepath: str, path to the CSV file
        :return: DataFrame, loaded data
        """
        data = pd.read_csv(filepath)
        return data

    def load_model(self, filepath):
        """
        Load a fitted model from a file.
        
        :param filepath: str, path to the model file
        :return: model, loaded model
        """
        model = joblib.load(filepath)
        return model

    def evaluate_explained_variance(self, original_data, reduced_data):
        """
        Evaluate the explained variance of the dimensionality reduction model.
        
        :param original_data: DataFrame, original high-dimensional data
        :param reduced_data: DataFrame, reduced-dimensionality data
        :return: float, explained variance score
        """
        explained_variance = explained_variance_score(original_data, reduced_data)
        return explained_variance

    def evaluate_reconstruction_error(self, original_data, reduced_data, model):
        """
        Evaluate the reconstruction error of the dimensionality reduction model.
        
        :param original_data: DataFrame, original high-dimensional data
        :param reduced_data: DataFrame, reduced-dimensionality data
        :param model: model, dimensionality reduction model used for transformation
        :return: float, reconstruction error
        """
        reconstructed_data = model.inverse_transform(reduced_data)
        reconstruction_error = mean_squared_error(original_data, reconstructed_data)
        return reconstruction_error

    def plot_explained_variance(self, explained_variance):
        """
        Plot the explained variance score.
        
        :param explained_variance: float, explained variance score
        """
        plt.figure(figsize=(8, 6))
        plt.bar(['Explained Variance'], [explained_variance], color='blue')
        plt.title('Explained Variance')
        plt.ylabel('Score')
        plt.show()

    def plot_reconstruction_error(self, reconstruction_error):
        """
        Plot the reconstruction error.
        
        :param reconstruction_error: float, reconstruction error
        """
        plt.figure(figsize=(8, 6))
        plt.bar(['Reconstruction Error'], [reconstruction_error], color='red')
        plt.title('Reconstruction Error')
        plt.ylabel('Error')
        plt.show()

if __name__ == "__main__":
    original_data_filepath = 'data/processed/preprocessed_high_dimensional_data.csv'
    reduced_data_filepath = 'data/processed/reduced_dimensionality_data.csv'
    model_filepath = 'models/dimensionality_reduction_model.pkl'

    evaluator = ModelEvaluation()

    # Load original and reduced data
    original_data = evaluator.load_data(original_data_filepath)
    reduced_data = evaluator.load_data(reduced_data_filepath)

    # Load the fitted model
    model = evaluator.load_model(model_filepath)

    # Evaluate explained variance
    explained_variance = evaluator.evaluate_explained_variance(original_data, reduced_data)
    print(f"Explained Variance: {explained_variance}")

    # Evaluate reconstruction error
    reconstruction_error = evaluator.evaluate_reconstruction_error(original_data, reduced_data, model)
    print(f"Reconstruction Error: {reconstruction_error}")

    # Plot evaluation metrics
    evaluator.plot_explained_variance(explained_variance)
    evaluator.plot_reconstruction_error(reconstruction_error)
    print("Model evaluation completed.")
