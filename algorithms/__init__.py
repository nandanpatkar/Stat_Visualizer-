"""
Machine Learning Algorithms Package

This package contains implementations of various machine learning algorithms
with educational explanations and visualizations.
"""

from .linear_regression import LinearRegression
from .logistic_regression import LogisticRegression
from .decision_tree import DecisionTree
from .random_forest import RandomForest
from .k_means import KMeans
from .k_nearest_neighbors import KNearestNeighbors
from .support_vector_machine import SupportVectorMachine
from .naive_bayes import NaiveBayes
from .gradient_boosting import GradientBoosting
from .neural_network import NeuralNetwork

__all__ = [
    'LinearRegression',
    'LogisticRegression',
    'DecisionTree',
    'RandomForest',
    'KMeans',
    'KNearestNeighbors',
    'SupportVectorMachine',
    'NaiveBayes',
    'GradientBoosting',
    'NeuralNetwork'
]