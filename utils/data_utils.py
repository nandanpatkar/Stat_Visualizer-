"""
Data Utilities Module

Contains utility functions for data processing, validation, and transformation.
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional


class DataUtils:
    """
    Utility class for data processing and validation.
    
    This class provides static methods for common data operations
    following PEP8 standards.
    """
    
    @staticmethod
    def validate_data(data: Union[np.ndarray, list, pd.Series]) -> np.ndarray:
        """
        Validate and convert input data to numpy array.
        
        Args:
            data: Input data in various formats
            
        Returns:
            np.ndarray: Validated and converted data
            
        Raises:
            ValueError: If data is invalid or empty
        """
        if data is None:
            raise ValueError("Data cannot be None")
            
        # Convert to numpy array
        if isinstance(data, (list, pd.Series)):
            data = np.array(data)
        elif not isinstance(data, np.ndarray):
            raise ValueError("Data must be array-like (list, numpy array, or pandas Series)")
        
        # Remove NaN values
        data = data[~np.isnan(data)]
        
        if len(data) == 0:
            raise ValueError("Data is empty after removing NaN values")
            
        return data
    
    @staticmethod
    def split_data(X: np.ndarray, y: np.ndarray, 
                   test_size: float = 0.2, 
                   random_state: Optional[int] = 42) -> Tuple[np.ndarray, ...]:
        """
        Split data into training and testing sets.
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        from sklearn.model_selection import train_test_split
        
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    @staticmethod
    def normalize_data(data: np.ndarray, method: str = 'standard') -> np.ndarray:
        """
        Normalize data using specified method.
        
        Args:
            data: Input data to normalize
            method: Normalization method ('standard', 'minmax', 'robust')
            
        Returns:
            np.ndarray: Normalized data
        """
        if method == 'standard':
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
        elif method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        elif method == 'robust':
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
        else:
            raise ValueError("Method must be 'standard', 'minmax', or 'robust'")
        
        if data.ndim == 1:
            data = data.reshape(-1, 1)
            return scaler.fit_transform(data).flatten()
        else:
            return scaler.fit_transform(data)
    
    @staticmethod
    def handle_missing_values(data: np.ndarray, strategy: str = 'mean') -> np.ndarray:
        """
        Handle missing values in data.
        
        Args:
            data: Input data with potential missing values
            strategy: Strategy for handling missing values ('mean', 'median', 'mode')
            
        Returns:
            np.ndarray: Data with missing values handled
        """
        if strategy == 'mean':
            fill_value = np.nanmean(data)
        elif strategy == 'median':
            fill_value = np.nanmedian(data)
        elif strategy == 'mode':
            from scipy import stats
            fill_value = stats.mode(data, nan_policy='omit')[0][0]
        else:
            raise ValueError("Strategy must be 'mean', 'median', or 'mode'")
        
        data_filled = data.copy()
        data_filled[np.isnan(data_filled)] = fill_value
        
        return data_filled
    
    @staticmethod
    def detect_outliers(data: np.ndarray, method: str = 'iqr') -> np.ndarray:
        """
        Detect outliers in data.
        
        Args:
            data: Input data
            method: Method for outlier detection ('iqr', 'zscore', 'isolation')
            
        Returns:
            np.ndarray: Boolean array indicating outliers
        """
        if method == 'iqr':
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return (data < lower_bound) | (data > upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs((data - np.mean(data)) / np.std(data))
            return z_scores > 3
            
        elif method == 'isolation':
            from sklearn.ensemble import IsolationForest
            clf = IsolationForest(contamination=0.1, random_state=42)
            outliers = clf.fit_predict(data.reshape(-1, 1))
            return outliers == -1
            
        else:
            raise ValueError("Method must be 'iqr', 'zscore', or 'isolation'")
    
    @staticmethod
    def generate_synthetic_data(data_type: str, n_samples: int = 100, 
                              **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate synthetic data for testing and demonstration.
        
        Args:
            data_type: Type of data to generate
            n_samples: Number of samples to generate
            **kwargs: Additional parameters for data generation
            
        Returns:
            Generated data (features only or features and targets)
        """
        np.random.seed(kwargs.get('random_state', 42))
        
        if data_type == 'normal':
            mean = kwargs.get('mean', 0)
            std = kwargs.get('std', 1)
            return np.random.normal(mean, std, n_samples)
            
        elif data_type == 'uniform':
            low = kwargs.get('low', 0)
            high = kwargs.get('high', 1)
            return np.random.uniform(low, high, n_samples)
            
        elif data_type == 'classification':
            from sklearn.datasets import make_classification
            n_features = kwargs.get('n_features', 2)
            n_classes = kwargs.get('n_classes', 2)
            return make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_classes=n_classes,
                random_state=kwargs.get('random_state', 42)
            )
            
        elif data_type == 'regression':
            from sklearn.datasets import make_regression
            n_features = kwargs.get('n_features', 1)
            noise = kwargs.get('noise', 0.1)
            return make_regression(
                n_samples=n_samples,
                n_features=n_features,
                noise=noise,
                random_state=kwargs.get('random_state', 42)
            )
            
        elif data_type == 'clustering':
            from sklearn.datasets import make_blobs
            n_centers = kwargs.get('n_centers', 3)
            cluster_std = kwargs.get('cluster_std', 1.0)
            return make_blobs(
                n_samples=n_samples,
                centers=n_centers,
                cluster_std=cluster_std,
                random_state=kwargs.get('random_state', 42)
            )
            
        else:
            raise ValueError("data_type must be one of: 'normal', 'uniform', 'classification', 'regression', 'clustering'")
    
    @staticmethod
    def calculate_correlation_matrix(data: np.ndarray) -> np.ndarray:
        """
        Calculate correlation matrix for multivariate data.
        
        Args:
            data: Input data matrix (samples x features)
            
        Returns:
            np.ndarray: Correlation matrix
        """
        return np.corrcoef(data, rowvar=False)
    
    @staticmethod
    def encode_categorical_data(data: Union[list, np.ndarray], 
                              method: str = 'onehot') -> np.ndarray:
        """
        Encode categorical data.
        
        Args:
            data: Categorical data to encode
            method: Encoding method ('onehot', 'label')
            
        Returns:
            np.ndarray: Encoded data
        """
        if method == 'onehot':
            from sklearn.preprocessing import OneHotEncoder
            encoder = OneHotEncoder(sparse=False)
            data_reshaped = np.array(data).reshape(-1, 1)
            return encoder.fit_transform(data_reshaped)
            
        elif method == 'label':
            from sklearn.preprocessing import LabelEncoder
            encoder = LabelEncoder()
            return encoder.fit_transform(data)
            
        else:
            raise ValueError("Method must be 'onehot' or 'label'")