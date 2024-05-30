# data_visualization.py

"""
Data Visualization Module for Dimensionality Reduction

This module contains functions for visualizing high-dimensional data after it has been reduced
to a lower-dimensional space using various techniques.

Techniques Used:
- Scatter plots
- 3D plots
- Heatmaps

Libraries/Tools:
- matplotlib
- seaborn
- plotly
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib

class DataVisualization:
    def __init__(self):
        """
        Initialize the DataVisualization class.
        """
        pass

    def load_data(self, filepath):
        """
        Load reduced-dimensionality data from a CSV file.
        
        :param filepath: str, path to the CSV file
        :return: DataFrame, loaded data
        """
        data = pd.read_csv(filepath)
        return data

    def plot_2d_scatter(self, data, labels=None):
        """
        Plot a 2D scatter plot of the reduced-dimensionality data.
        
        :param data: DataFrame, reduced-dimensionality data
        :param labels: array, labels for coloring the plot (optional)
        """
        plt.figure(figsize=(8, 6))
        if labels is not None:
            plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=labels, cmap='viridis', s=50)
        else:
            plt.scatter(data.iloc[:, 0], data.iloc[:, 1], s=50)
        plt.title('2D Scatter Plot')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.colorbar()
        plt.show()

    def plot_3d_scatter(self, data, labels=None):
        """
        Plot a 3D scatter plot of the reduced-dimensionality data.
        
        :param data: DataFrame, reduced-dimensionality data
        :param labels: array, labels for coloring the plot (optional)
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        if labels is not None:
            scatter = ax.scatter(data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2], c=labels, cmap='viridis', s=50)
        else:
            scatter = ax.scatter(data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2], s=50)
        plt.title('3D Scatter Plot')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
        fig.colorbar(scatter)
        plt.show()

    def plot_heatmap(self, data):
        """
        Plot a heatmap of the reduced-dimensionality data.
        
        :param data: DataFrame, reduced-dimensionality data
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(data, cmap='viridis', annot=True, fmt=".2f")
        plt.title('Heatmap')
        plt.show()

    def plot_interactive_2d(self, data, labels=None):
        """
        Plot an interactive 2D scatter plot of the reduced-dimensionality data using Plotly.
        
        :param data: DataFrame, reduced-dimensionality data
        :param labels: array, labels for coloring the plot (optional)
        """
        if labels is not None:
            fig = px.scatter(data, x=data.columns[0], y=data.columns[1], color=labels, title='Interactive 2D Scatter Plot')
        else:
            fig = px.scatter(data, x=data.columns[0], y=data.columns[1], title='Interactive 2D Scatter Plot')
        fig.show()

    def plot_interactive_3d(self, data, labels=None):
        """
        Plot an interactive 3D scatter plot of the reduced-dimensionality data using Plotly.
        
        :param data: DataFrame, reduced-dimensionality data
        :param labels: array, labels for coloring the plot (optional)
        """
        if labels is not None:
            fig = px.scatter_3d(data, x=data.columns[0], y=data.columns[1], z=data.columns[2], color=labels, title='Interactive 3D Scatter Plot')
        else:
            fig = px.scatter_3d(data, x=data.columns[0], y=data.columns[1], z=data.columns[2], title='Interactive 3D Scatter Plot')
        fig.show()

if __name__ == "__main__":
    reduced_data_filepath = 'data/processed/reduced_dimensionality_data.csv'
    labels_filepath = 'data/processed/target_labels.csv'  # Optional: needed for coloring the plots

    visualization = DataVisualization()

    # Load reduced-dimensionality data
    data = visualization.load_data(reduced_data_filepath)
    labels = pd.read_csv(labels_filepath).values.ravel() if labels_filepath else None

    # Plot 2D scatter plot
    visualization.plot_2d_scatter(data, labels)

    # Plot 3D scatter plot
    visualization.plot_3d_scatter(data, labels)

    # Plot heatmap
    visualization.plot_heatmap(data)

    # Plot interactive 2D scatter plot
    visualization.plot_interactive_2d(data, labels)

    # Plot interactive 3D scatter plot
    visualization.plot_interactive_3d(data, labels)

    print("Data visualization completed.")
