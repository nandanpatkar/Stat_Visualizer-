"""
K-Nearest Neighbors (KNN) Algorithm Implementation

K-Nearest Neighbors is a simple, non-parametric, lazy learning algorithm used
for classification and regression. It makes predictions based on the k closest
training examples in the feature space.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report,
                           mean_squared_error, r2_score, mean_absolute_error)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler
import streamlit as st
import seaborn as sns


class KNearestNeighbors:
    """
    K-Nearest Neighbors implementation with educational explanations.
    
    KNN is a lazy learning algorithm that stores all training data and makes
    predictions by finding the k nearest neighbors to a query point and:
    - Classification: Taking majority vote of neighbors' classes
    - Regression: Taking average of neighbors' target values
    
    Distance Metrics:
    - Euclidean: √(Σ(xi - yi)²) - Most common
    - Manhattan: Σ|xi - yi| - Good for high dimensions
    - Minkowski: (Σ|xi - yi|^p)^(1/p) - Generalization of above
    """
    
    def __init__(self, n_neighbors=5, task_type='classification', metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.task_type = task_type
        self.metric = metric
        self.model = None
        self.is_fitted = False
        
    @staticmethod
    def get_theory():
        """Return comprehensive theoretical explanation of K-Nearest Neighbors."""
        return {
            'name': 'K-Nearest Neighbors (KNN)',
            'type': 'Supervised Learning - Classification/Regression',
            
            # 1. What the algorithm is
            'definition': """
            🔹 **What is K-Nearest Neighbors?**
            KNN is like asking your k closest friends for advice before making a decision.
            It's the ultimate "crowd wisdom" algorithm - it looks at the k most similar 
            examples and goes with what the majority says!
            """,
            
            # 2. Why the algorithm is used
            'motivation': """
            🔹 **Why Use K-Nearest Neighbors?**
            • 🎬 **Recommendation Systems**: "Users like you also liked..."
            • 🔍 **Image Recognition**: Find similar images to classify new ones
            • 🏘️ **Real Estate**: Price houses based on similar nearby properties
            • 👥 **Customer Segmentation**: Group customers by similarity
            • 🎯 **Anomaly Detection**: Find outliers that don't fit any pattern
            • 📊 **Pattern Recognition**: Identify patterns in complex data
            """,
            
            # 3. Intuition with real-life analogy
            'intuition': """
            🔹 **Real-Life Analogy: The New Kid in School**
            
            Imagine you're a new student trying to figure out which lunch table to sit at:
            
            🏫 **The Problem**: Where should I sit? What group do I belong to?
            👥 **Your Strategy**: Look at students most similar to me
            
            **KNN is like being the new kid who:**
            
            **Step 1**: 📏 Measures similarity to everyone
            - Similar interests? ✓
            - Similar style? ✓ 
            - Similar academic level? ✓
            - Lives in same neighborhood? ✓
            
            **Step 2**: 🎯 Finds the k=5 most similar students
            - Student A: 95% similar (loves math, plays guitar, wears hoodies)
            - Student B: 92% similar (loves science, plays piano, wears hoodies)
            - Student C: 90% similar (loves art, plays drums, wears hoodies)  
            - Student D: 88% similar (loves math, plays violin, wears jeans)
            - Student E: 85% similar (loves science, plays bass, wears jeans)
            
            **Step 3**: 🗳️ Takes majority vote of where they sit
            - A, B, C sit at "Creative STEM" table
            - D, E sit at "Academic" table
            - Majority vote: "Creative STEM" (3 vs 2)
            
            **Step 4**: 🪑 Sits at "Creative STEM" table!
            
            **Key Insights:**
            • **No learning phase**: Just remembers everyone's info
            • **Local decisions**: Only similar people matter
            • **Flexible boundaries**: Works for any group structure
            • **Similarity matters**: Need good way to measure similarity
            
            **For regression** (predicting GPA):
            Instead of voting, average the GPAs of k similar students!
            
            🎯 **In data terms**: 
            - Students = Training samples
            - Characteristics = Features
            - Lunch tables = Classes
            - Similarity = Distance metric
            - You = New sample to classify
            """,
            
            # 4. Mathematical foundation
            'math_foundation': """
            🔹 **Mathematical Foundation (Step-by-Step)**
            
            **Distance Metrics (Similarity Measurement):**
            
            **1. Euclidean Distance (Most Common):**
            ```
            d(x, y) = √(Σᵢ₌₁ⁿ (xᵢ - yᵢ)²)
            ```
            
            **2. Manhattan Distance:**
            ```
            d(x, y) = Σᵢ₌₁ⁿ |xᵢ - yᵢ|
            ```
            
            **3. Minkowski Distance (Generalization):**
            ```
            d(x, y) = (Σᵢ₌₁ⁿ |xᵢ - yᵢ|ᵖ)^(1/p)
            ```
            Where p=1 → Manhattan, p=2 → Euclidean
            
            **4. Cosine Distance:**
            ```
            d(x, y) = 1 - (x·y)/(|x||y|) = 1 - cos(θ)
            ```
            
            **Classification Decision:**
            ```
            ŷ = mode{y₁, y₂, ..., yₖ}
            ```
            Where y₁, y₂, ..., yₖ are labels of k nearest neighbors
            
            **Regression Prediction:**
            ```
            ŷ = (1/k) × Σᵢ₌₁ᵏ yᵢ
            ```
            
            **Weighted KNN (Distance-based weights):**
            ```
            Classification: ŷ = argmaxc Σᵢ₌₁ᵏ wᵢ × I(yᵢ = c)
            Regression: ŷ = Σᵢ₌₁ᵏ wᵢ × yᵢ / Σᵢ₌₁ᵏ wᵢ
            ```
            Where wᵢ = 1/d(x, xᵢ) or wᵢ = exp(-d(x, xᵢ)²/σ²)
            
            **Probability Estimates (Classification):**
            ```
            P(y = c | x) = (1/k) × Σᵢ₌₁ᵏ I(yᵢ = c)
            ```
            
            **Curse of Dimensionality:**
            As dimensions increase, all points become equidistant:
            ```
            limₚ→∞ (max d(x, xᵢ) - min d(x, xᵢ)) / min d(x, xᵢ) → 0
            ```
            """,
            
            # 5. Step-by-step working
            'algorithm_steps': """
            🔹 **How KNN Works (Step-by-Step)**
            
            **Step 1: Store Training Data** 📦
            • Store all training samples (X_train, y_train)
            • No model building or parameter learning
            • Just keep everything in memory ("lazy learning")
            
            **Step 2: Preprocess Features** 🔧
            • Scale features to similar ranges (crucial!)
            • Handle categorical variables appropriately
            • Remove irrelevant features if possible
            
            **Step 3: Choose Hyperparameters** ⚙️
            • Select k (number of neighbors)
            • Choose distance metric (usually Euclidean)
            • Decide on uniform vs distance-weighted voting
            
            **Step 4: New Sample Arrives** 🆕
            • Receive new sample x_new for prediction
            • Apply same preprocessing as training data
            
            **Step 5: Calculate Distances** 📏
            • Compute distance from x_new to ALL training samples
            • d(x_new, x₁), d(x_new, x₂), ..., d(x_new, x_n)
            • This is the computationally expensive step!
            
            **Step 6: Find k Nearest Neighbors** 🎯
            • Sort all distances in ascending order
            • Select k samples with smallest distances
            • Store their indices and corresponding labels/values
            
            **Step 7: Make Prediction** 🔮
            
            **For Classification:**
            • Count votes for each class among k neighbors
            • Return class with most votes (majority voting)
            • Handle ties by: random choice, reduce k, or distance weighting
            
            **For Regression:**
            • Average target values of k neighbors
            • Can use distance-weighted average for better results
            
            **Step 8: Handle Edge Cases** 🚨
            • If k > number of training samples, use all samples
            • For distance weighting, handle zero distances (identical points)
            • Consider outlier detection and removal
            
            **Optimization Tricks:**
            • Use KD-trees or Ball trees for faster neighbor search
            • Approximate methods for very large datasets
            • Dimensionality reduction for high-dimensional data
            """,
            
            # 6. Pseudocode
            'pseudocode': """
            🔹 **Pseudocode (Easy to Understand)**
            
            ```
            ALGORITHM: K-Nearest Neighbors
            
            INPUT: 
                - X_train: training features (n_samples × n_features)
                - y_train: training labels (n_samples × 1)
                - k: number of neighbors
                - distance_metric: function to compute distance
                - weights: 'uniform' or 'distance'
            
            OUTPUT:
                - trained_model: stored training data
            
            TRAINING:
            BEGIN
                1. # No explicit training - just store data
                   stored_X = X_train
                   stored_y = y_train
                   
                2. # Optionally preprocess
                   stored_X = standardize(X_train)
                   
                3. RETURN (stored_X, stored_y, k, distance_metric, weights)
            END
            
            PREDICTION:
            BEGIN
                1. INPUT: new_sample x
                
                2. # Preprocess new sample same way as training
                   x = standardize(x)
                   
                3. # Calculate distances to all training samples
                   distances = []
                   FOR i = 1 to n_samples:
                       dist = distance_metric(x, stored_X[i])
                       distances.append((dist, i))
                   
                4. # Sort by distance and get k nearest
                   distances.sort()  # Sort by distance
                   k_nearest = distances[1:k+1]  # Get k closest
                   
                5. # Extract labels/values of k nearest neighbors
                   neighbor_labels = []
                   neighbor_distances = []
                   FOR (dist, idx) in k_nearest:
                       neighbor_labels.append(stored_y[idx])
                       neighbor_distances.append(dist)
                   
                6. # Make prediction based on task type
                   IF classification:
                       IF weights == 'uniform':
                           # Simple majority vote
                           prediction = mode(neighbor_labels)
                       ELSE:
                           # Distance-weighted vote
                           vote_weights = [1/max(d, 1e-10) for d in neighbor_distances]
                           prediction = weighted_mode(neighbor_labels, vote_weights)
                   
                   ELIF regression:
                       IF weights == 'uniform':
                           # Simple average
                           prediction = mean(neighbor_labels)
                       ELSE:
                           # Distance-weighted average
                           weights = [1/max(d, 1e-10) for d in neighbor_distances]
                           prediction = weighted_average(neighbor_labels, weights)
                   
                7. RETURN prediction
            END
            
            HELPER FUNCTIONS:
            BEGIN
                FUNCTION distance_metric(x1, x2):
                    # Euclidean distance
                    RETURN sqrt(sum((x1[i] - x2[i])^2 for i in 1:n_features))
                
                FUNCTION mode(labels):
                    # Return most frequent label
                    RETURN most_common_element(labels)
                
                FUNCTION weighted_mode(labels, weights):
                    # Return label with highest weighted vote
                    class_weights = {}
                    FOR i = 1 to len(labels):
                        class_weights[labels[i]] += weights[i]
                    RETURN key_with_max_value(class_weights)
            END
            ```
            """,
            
            # 7. Python implementation
            'python_implementation': """
            🔹 **Python Implementation**
            
            **From Scratch (Simplified KNN):**
            ```python
            import numpy as np
            from collections import Counter
            from scipy.spatial.distance import euclidean, manhattan, minkowski
            
            class KNNFromScratch:
                def __init__(self, k=5, distance_metric='euclidean', weights='uniform'):
                    self.k = k
                    self.distance_metric = distance_metric
                    self.weights = weights
                    self.X_train = None
                    self.y_train = None
                    
                def fit(self, X, y):
                    \"\"\"Store training data (no actual training).\"\"\"
                    self.X_train = np.array(X)
                    self.y_train = np.array(y)
                    
                def _calculate_distance(self, x1, x2):
                    \"\"\"Calculate distance between two points.\"\"\"
                    if self.distance_metric == 'euclidean':
                        return np.sqrt(np.sum((x1 - x2) ** 2))
                    elif self.distance_metric == 'manhattan':
                        return np.sum(np.abs(x1 - x2))
                    elif self.distance_metric == 'minkowski':
                        # p=3 for example
                        return np.sum(np.abs(x1 - x2) ** 3) ** (1/3)
                    else:
                        raise ValueError(f"Unknown distance metric: {self.distance_metric}")
                    
                def _get_neighbors(self, x):
                    \"\"\"Find k nearest neighbors for a given point.\"\"\"
                    distances = []
                    
                    # Calculate distance to all training points
                    for i, x_train in enumerate(self.X_train):
                        dist = self._calculate_distance(x, x_train)
                        distances.append((dist, i))
                    
                    # Sort by distance and get k nearest
                    distances.sort(key=lambda x: x[0])
                    neighbors = distances[:self.k]
                    
                    return neighbors
                
                def predict(self, X):
                    \"\"\"Make predictions for new data.\"\"\"
                    predictions = []
                    
                    for x in X:
                        neighbors = self._get_neighbors(x)
                        
                        # Extract neighbor labels and distances
                        neighbor_labels = [self.y_train[idx] for _, idx in neighbors]
                        neighbor_distances = [dist for dist, _ in neighbors]
                        
                        # Make prediction based on task type
                        if self._is_classification():
                            if self.weights == 'uniform':
                                # Simple majority vote
                                prediction = Counter(neighbor_labels).most_common(1)[0][0]
                            else:
                                # Distance-weighted vote
                                weighted_votes = {}
                                for label, dist in zip(neighbor_labels, neighbor_distances):
                                    weight = 1 / (dist + 1e-10)  # Avoid division by zero
                                    weighted_votes[label] = weighted_votes.get(label, 0) + weight
                                prediction = max(weighted_votes, key=weighted_votes.get)
                        else:
                            # Regression
                            if self.weights == 'uniform':
                                prediction = np.mean(neighbor_labels)
                            else:
                                # Distance-weighted average
                                weights = [1 / (dist + 1e-10) for dist in neighbor_distances]
                                prediction = np.average(neighbor_labels, weights=weights)
                        
                        predictions.append(prediction)
                    
                    return np.array(predictions)
                
                def _is_classification(self):
                    \"\"\"Check if this is a classification task.\"\"\"
                    # Simple heuristic: if labels are integers or strings, it's classification
                    return isinstance(self.y_train[0], (int, str, bool)) or len(np.unique(self.y_train)) < len(self.y_train) * 0.05
                
                def predict_proba(self, X):
                    \"\"\"Predict class probabilities (classification only).\"\"\"
                    if not self._is_classification():
                        raise ValueError("predict_proba only available for classification")
                        
                    probabilities = []
                    classes = np.unique(self.y_train)
                    
                    for x in X:
                        neighbors = self._get_neighbors(x)
                        neighbor_labels = [self.y_train[idx] for _, idx in neighbors]
                        neighbor_distances = [dist for dist, _ in neighbors]
                        
                        if self.weights == 'uniform':
                            # Count votes for each class
                            class_counts = Counter(neighbor_labels)
                            class_probs = {cls: class_counts.get(cls, 0) / self.k for cls in classes}
                        else:
                            # Distance-weighted probabilities
                            class_weights = {cls: 0 for cls in classes}
                            total_weight = 0
                            for label, dist in zip(neighbor_labels, neighbor_distances):
                                weight = 1 / (dist + 1e-10)
                                class_weights[label] += weight
                                total_weight += weight
                            class_probs = {cls: weight / total_weight for cls, weight in class_weights.items()}
                        
                        # Convert to array in consistent order
                        prob_array = [class_probs[cls] for cls in classes]
                        probabilities.append(prob_array)
                    
                    return np.array(probabilities)
            
            # Example usage
            from sklearn.datasets import make_classification
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            
            # Generate data
            X, y = make_classification(n_samples=1000, n_features=4, n_classes=3, random_state=42)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features (important for distance-based algorithms!)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train and predict
            knn = KNNFromScratch(k=5, distance_metric='euclidean', weights='distance')
            knn.fit(X_train_scaled, y_train)
            predictions = knn.predict(X_test_scaled)
            probabilities = knn.predict_proba(X_test_scaled)
            ```
            
            **Using Scikit-learn:**
            ```python
            from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import GridSearchCV
            
            # Feature scaling (CRUCIAL for KNN!)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Classification
            knn_clf = KNeighborsClassifier(
                n_neighbors=5,              # Number of neighbors
                weights='distance',         # Weight by inverse distance
                algorithm='auto',          # Let sklearn choose best algorithm
                metric='euclidean',        # Distance metric
                p=2                        # Power parameter for Minkowski
            )
            knn_clf.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = knn_clf.predict(X_test_scaled)
            y_proba = knn_clf.predict_proba(X_test_scaled)
            
            # Get neighbors for specific points
            distances, indices = knn_clf.kneighbors(X_test_scaled[:5], return_distance=True)
            
            # Regression
            knn_reg = KNeighborsRegressor(
                n_neighbors=5,
                weights='uniform',
                algorithm='kd_tree',       # Use KD-tree for speed
                leaf_size=30
            )
            knn_reg.fit(X_train_scaled, y_train)
            y_pred_reg = knn_reg.predict(X_test_scaled)
            
            # Hyperparameter tuning
            param_grid = {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            }
            
            grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
            grid_search.fit(X_train_scaled, y_train)
            best_knn = grid_search.best_estimator_
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
            ```
            
            **Distance Metrics Examples:**
            ```python
            # Different distance metrics for different data types
            
            # Euclidean (continuous features)
            knn_euclidean = KNeighborsClassifier(metric='euclidean')
            
            # Manhattan (when features have different units)
            knn_manhattan = KNeighborsClassifier(metric='manhattan')
            
            # Cosine (text data, high-dimensional sparse features)
            knn_cosine = KNeighborsClassifier(metric='cosine')
            
            # Hamming (categorical features)
            knn_hamming = KNeighborsClassifier(metric='hamming')
            
            # Custom distance function
            def custom_distance(x, y):
                return np.sum((x - y) ** 2)  # Squared Euclidean
                
            knn_custom = KNeighborsClassifier(metric=custom_distance)
            ```
            """,
            
            # 8. Example with sample input/output
            'example': """
            🔹 **Complete Example: Movie Recommendation System**
            
            **Input Data (User Movie Ratings):**
            ```
            User | Action | Comedy | Drama | Horror | Romance | Sci-Fi | User_Type
            A    | 5      | 2      | 3     | 1      | 2       | 4      | Action_Fan
            B    | 1      | 5      | 4     | 2      | 5       | 2      | Comedy_Fan
            C    | 2      | 4      | 5     | 1      | 4       | 3      | Drama_Fan
            D    | 5      | 1      | 2     | 5      | 1       | 4      | Action_Fan
            E    | 1      | 5      | 3     | 2      | 5       | 2      | Comedy_Fan
            F    | 3      | 3      | 5     | 1      | 4       | 3      | Drama_Fan
            ```
            
            **Step-by-Step KNN Prediction:**
            ```
            New User X: [4, 3, 3, 2, 2, 4] → What type of fan?
            
            1. Calculate distances to all users (using k=3):
               
               Distance to A: √((4-5)² + (3-2)² + (3-3)² + (2-1)² + (2-2)² + (4-4)²)
                            = √(1 + 1 + 0 + 1 + 0 + 0) = √3 = 1.73
               
               Distance to B: √((4-1)² + (3-5)² + (3-4)² + (2-2)² + (2-5)² + (4-2)²)
                            = √(9 + 4 + 1 + 0 + 9 + 4) = √27 = 5.20
               
               Distance to C: √((4-2)² + (3-4)² + (3-5)² + (2-1)² + (2-4)² + (4-3)²)
                            = √(4 + 1 + 4 + 1 + 4 + 1) = √15 = 3.87
               
               Distance to D: √((4-5)² + (3-1)² + (3-2)² + (2-5)² + (2-1)² + (4-4)²)
                            = √(1 + 4 + 1 + 9 + 1 + 0) = √16 = 4.00
               
               Distance to E: √((4-1)² + (3-5)² + (3-3)² + (2-2)² + (2-5)² + (4-2)²)
                            = √(9 + 4 + 0 + 0 + 9 + 4) = √26 = 5.10
               
               Distance to F: √((4-3)² + (3-3)² + (3-5)² + (2-1)² + (2-4)² + (4-3)²)
                            = √(1 + 0 + 4 + 1 + 4 + 1) = √11 = 3.32
            
            2. Sort by distance:
               A: 1.73 (Action_Fan)
               F: 3.32 (Drama_Fan) 
               C: 3.87 (Drama_Fan)
               D: 4.00 (Action_Fan)
               E: 5.10 (Comedy_Fan)
               B: 5.20 (Comedy_Fan)
            
            3. Select k=3 nearest neighbors:
               A: Action_Fan
               F: Drama_Fan
               C: Drama_Fan
            
            4. Make prediction (majority vote):
               Action_Fan: 1 vote
               Drama_Fan: 2 votes
               Comedy_Fan: 0 votes
               
               Prediction: DRAMA_FAN! 🎭
            ```
            
            **With Distance Weighting:**
            ```
            1. Calculate weights (1/distance):
               A: weight = 1/1.73 = 0.58
               F: weight = 1/3.32 = 0.30
               C: weight = 1/3.87 = 0.26
            
            2. Weighted votes:
               Action_Fan: 0.58 points
               Drama_Fan: 0.30 + 0.26 = 0.56 points
               
            3. Closer prediction: STILL Drama_Fan, but much closer!
            ```
            
            **Recommendation Logic:**
            ```
            Since User X is predicted as Drama_Fan, recommend:
            - Highly rated dramas
            - Movies liked by similar drama fans (A, F, C)
            - Avoid pure action or comedy movies
            ```
            """,
            
            # 9. Visualization explanation
            'visualization': """
            🔹 **Understanding Through Visualizations**
            
            **2D Scatter Plot with Decision Boundaries:**
            📊 Shows how KNN creates decision regions
            • Each color represents different class
            • Voronoi-like boundaries between classes
            • Complex, non-linear decision boundaries
            • Boundaries change based on k value
            
            **K-Value Effect Visualization:**
            🔄 Side-by-side plots showing different k values
            • k=1: Very complex boundaries (overfitting)
            • k=5: Smoother boundaries
            • k=large: Very simple boundaries (underfitting)
            • Shows bias-variance trade-off
            
            **Distance Metric Comparison:**
            📏 Same data with different distance functions
            • Euclidean: Circular neighborhoods
            • Manhattan: Diamond-shaped neighborhoods  
            • Cosine: Angle-based similarities
            • Shows impact of distance choice
            
            **Neighborhood Visualization:**
            🎯 Highlighting k nearest neighbors for specific point
            • Query point in center
            • k nearest points highlighted
            • Distance circles showing neighborhood
            • Arrows showing voting/averaging process
            
            **Curse of Dimensionality Demo:**
            📈 Distance distribution in different dimensions
            • Low dimensions: Clear nearest/farthest distinction
            • High dimensions: All points become equidistant
            • Shows why KNN struggles in high dimensions
            
            **Performance vs K Plot:**
            📊 Validation accuracy/error vs different k values
            • U-shaped curve showing optimal k
            • Cross-validation scores for different k
            • Helps choose optimal hyperparameter
            """,
            
            # 10. Time and space complexity
            'complexity': """
            🔹 **Time & Space Complexity**
            
            **Time Complexity:**
            • **Training**: O(1) - just stores data ("lazy learning")
            • **Prediction**: O(n × d × q) where:
              - n = training samples
              - d = dimensions (features)
              - q = query samples
            • **k-NN Search**: O(n × d) per query (brute force)
            • **Sorting**: O(n × log(n)) per query
            
            **Space Complexity:**
            • **Model Storage**: O(n × d) - stores entire training set
            • **Prediction Memory**: O(k) for storing neighbors
            • **Distance Calculation**: O(n) temporary storage
            
            **Optimization Algorithms:**
            • **KD-Tree**: O(log n) search in low dimensions (d < 10)
            • **Ball Tree**: O(log n) search, works better in higher dimensions
            • **LSH (Locality Sensitive Hashing)**: Approximate, O(1) average case
            • **Brute Force**: O(n) search, but simple and works always
            
            **Scalability Issues:**
            • ❌ **Large Datasets**: Linear search doesn't scale
            • ❌ **High Dimensions**: Curse of dimensionality (d > 10-20)
            • ❌ **Real-time Prediction**: Can be too slow for online systems
            • ✅ **Parallel Search**: Distance calculations can be parallelized
            • ✅ **Approximate Methods**: Trade accuracy for speed
            
            **Performance Characteristics:**
            • Training: Instant (no computation)
            • Memory: Scales linearly with training data size
            • Prediction: Gets slower as training data grows
            • Works best with small to medium datasets (< 10K samples)
            """,
            
            # 11. Advantages and disadvantages
            'pros_cons': """
            🔹 **Advantages** ✅
            • **Conceptually Simple**: Easy to understand and explain
            • **No Training Period**: Instant "training" (just store data)
            • **Flexible Decision Boundaries**: Can capture complex patterns
            • **Multi-class Native**: Naturally handles multiple classes
            • **Probabilistic Outputs**: Can provide confidence estimates
            • **No Data Assumptions**: Works with any data distribution
            • **Local Learning**: Adapts to local patterns in data
            • **Online Learning**: Easy to add new training samples
            • **Versatile**: Works for both classification and regression
            • **Robust to Noisy Training Data**: Outliers have limited impact
            
            🔹 **Disadvantages** ❌
            • **Computationally Expensive**: Slow prediction for large datasets
            • **Memory Intensive**: Stores entire training dataset
            • **Curse of Dimensionality**: Performance degrades in high dimensions
            • **Feature Scaling Sensitive**: Requires careful preprocessing
            • **Sensitive to Irrelevant Features**: Noise features hurt performance
            • **No Model Interpretability**: Can't extract simple rules
            • **Boundary Artifacts**: Can create jagged decision boundaries
            • **Class Imbalance Issues**: Majority class can dominate
            • **Storage Requirements**: Model size grows with training data
            • **Distance Metric Choice**: Performance depends on good distance function
            """,
            
            # 12. When to use and when NOT to use
            'usage_guide': """
            🔹 **When TO Use KNN** ✅
            
            **Perfect for:**
            • 🎬 **Recommendation Systems**: "Users like you also liked..."
            • 🔍 **Image/Pattern Recognition**: Find similar images or patterns
            • 🏘️ **Geographic Applications**: Real estate, location-based services
            • 📊 **Small to Medium Datasets**: < 10K samples with good structure
            • 🎯 **Anomaly Detection**: Find outliers that don't fit patterns
            • 📝 **Prototype Selection**: Quick baseline for new problems
            
            **Good when:**
            • Local patterns are more important than global trends
            • Data has natural clustering structure
            • Need quick prototyping without model training
            • Decision boundaries are complex and non-linear
            • Want probabilistic outputs
            • Have sufficient computational resources for prediction
            
            🔹 **When NOT to Use KNN** ❌
            
            **Avoid when:**
            • 📈 **Large Datasets**: > 100K samples (too slow)
            • 🌐 **High Dimensions**: > 20-50 features (curse of dimensionality)
            • ⚡ **Real-time Predictions**: Need millisecond response times
            • 🎯 **Uniform Data**: All samples look similar (no clear neighbors)
            • 📏 **Different Feature Scales**: Can't easily standardize features
            • 🔄 **Streaming Data**: Constantly changing data distribution
            • 💾 **Memory Constraints**: Limited storage for model
            
            **Use instead:**
            • **Large Data**: Random Forest, Gradient Boosting, Neural Networks
            • **High Dimensions**: SVM, Naive Bayes, dimensionality reduction + KNN
            • **Speed Critical**: Naive Bayes, Linear models, pre-computed lookup tables
            • **Interpretability**: Decision Trees, Linear models
            • **Feature Interactions**: Tree-based methods, polynomial features
            """,
            
            # 13. Common interview questions
            'interview_questions': """
            🔹 **Common Interview Questions & Answers**
            
            **Q1: How do you choose the optimal value of k?**
            A: Several approaches:
            • **Cross-validation**: Try odd values (3, 5, 7, 9, 11) and pick best CV score
            • **Rule of thumb**: k = √n where n is number of training samples
            • **Domain knowledge**: Consider how many similar cases you'd want to consult
            • **Validation curve**: Plot accuracy vs k and look for sweet spot
            • **Avoid even k**: Can lead to ties in voting
            
            **Q2: Why is feature scaling crucial for KNN?**
            A: KNN uses distance calculations, so features with larger scales dominate:
            • Example: Age (0-100) vs Income (0-100,000) - income will dominate distance
            • Solution: Use StandardScaler or MinMaxScaler to normalize all features
            • Without scaling, algorithm becomes biased toward high-magnitude features
            
            **Q3: What is the curse of dimensionality and how does it affect KNN?**
            A: In high dimensions, all points become roughly equidistant:
            • Distance to nearest and farthest points converge
            • "Nearest" neighbors aren't actually very near
            • Solutions: Dimensionality reduction (PCA), feature selection, or different algorithms
            
            **Q4: How do you handle categorical features in KNN?**
            A: Options:
            • **One-hot encoding**: Convert to binary features
            • **Label encoding**: For ordinal categories only
            • **Hamming distance**: For categorical data specifically
            • **Mixed distance metrics**: Combine different distances for different feature types
            
            **Q5: What's the difference between uniform and distance-weighted KNN?**
            A:
            • **Uniform**: All k neighbors have equal vote (simple majority)
            • **Distance-weighted**: Closer neighbors have more influence
            • Distance weighting often performs better, especially with larger k
            • Helps when relevant neighbors are at different distances
            
            **Q6: How can you speed up KNN for large datasets?**
            A:
            • **KD-trees/Ball trees**: For exact results in moderate dimensions
            • **Approximate methods**: LSH, random projections
            • **Dimensionality reduction**: PCA before KNN
            • **Sampling**: Use representative subset of training data
            • **Parallel processing**: Distribute distance calculations
            """,
            
            # 14. Common mistakes
            'common_mistakes': """
            🔹 **Common Beginner Mistakes & How to Avoid**
            
            **Mistake 1: Not scaling features** 🚫
            ❌ Using raw features with different scales (age vs income)
            ✅ **Fix**: Always use StandardScaler() or MinMaxScaler() first
            
            **Mistake 2: Using even values of k** 🚫
            ❌ Choosing k=4, k=6, etc. which can lead to ties
            ✅ **Fix**: Use odd values (3, 5, 7, 9) to avoid ties in voting
            
            **Mistake 3: Using KNN with high-dimensional data** 🚫
            ❌ Applying KNN to data with 100+ features without preprocessing
            ✅ **Fix**: Use dimensionality reduction or feature selection first
            
            **Mistake 4: Not cross-validating k selection** 🚫
            ❌ Picking k arbitrarily or using k=5 always
            ✅ **Fix**: Use GridSearchCV to find optimal k for your specific data
            
            **Mistake 5: Ignoring class imbalance** 🚫
            ❌ Using KNN when one class dominates (90% vs 10%)
            ✅ **Fix**: Use stratified sampling, class weights, or SMOTE
            
            **Mistake 6: Wrong distance metric choice** 🚫
            ❌ Using Euclidean distance for text data or categorical features
            ✅ **Fix**: Match distance to data type (cosine for text, Hamming for categorical)
            
            **Mistake 7: Not handling outliers** 🚫
            ❌ Keeping extreme outliers that skew neighborhood selection
            ✅ **Fix**: Remove obvious outliers or use robust distance metrics
            
            **Mistake 8: Using KNN for large datasets without optimization** 🚫
            ❌ Running brute force KNN on 1M+ samples
            ✅ **Fix**: Use approximate methods, sampling, or choose different algorithm
            """,
            
            # 15. Comparison with similar algorithms
            'comparisons': """
            🔹 **KNN vs Similar Algorithms**
            
            **KNN vs Naive Bayes:**
            • **KNN**: Local decisions, no assumptions about data distribution
            • **Naive Bayes**: Global model, assumes feature independence
            • **Use Naive Bayes**: When features are approximately independent and need speed
            
            **KNN vs SVM:**
            • **KNN**: Instance-based, no training phase, flexible boundaries
            • **SVM**: Model-based, finds optimal hyperplane, global decision
            • **Use SVM**: For high-dimensional data or when maximum margin is important
            
            **KNN vs Random Forest:**
            • **KNN**: Stores all data, local neighborhoods, simple
            • **Random Forest**: Tree ensemble, feature interactions, faster prediction
            • **Use Random Forest**: For tabular data with feature interactions
            
            **KNN vs Logistic Regression:**
            • **KNN**: Non-parametric, complex boundaries, no model assumptions
            • **Logistic Regression**: Parametric, linear boundaries, interpretable
            • **Use Logistic Regression**: When relationships are roughly linear
            
            **KNN vs Clustering (K-Means):**
            • **KNN**: Supervised, uses labels, predicts new points
            • **K-Means**: Unsupervised, finds clusters, no prediction on new points
            • Different purposes: KNN for prediction, K-Means for discovery
            
            **KNN vs Neural Networks:**
            • **KNN**: Simple, interpretable, no training, works with small data
            • **Neural Networks**: Complex patterns, needs lots of data, long training
            • **Use Neural Networks**: With large datasets and complex patterns
            """,
            
            # 16. Real-world applications
            'real_world_applications': """
            🔹 **Real-World Applications & Industry Use Cases**
            
            **🎬 Recommendation Systems:**
            • Netflix: "Users who watched X also watched Y"
            • Amazon: "Customers who bought this item also bought"
            • Spotify: Music recommendations based on listening history
            • YouTube: Video suggestions based on viewing patterns
            • Dating apps: Match users with similar preferences
            
            **🖼️ Computer Vision & Image Recognition:**
            • Face recognition systems in security cameras
            • Handwriting recognition in postal services
            • Medical image analysis (X-ray, MRI similarity matching)
            • Content-based image retrieval in photo galleries
            • Quality control in manufacturing (defect detection)
            
            **🏘️ Geographic & Location Services:**
            • Real estate price prediction based on neighborhood
            • Location-based restaurant recommendations
            • Urban planning and demographic analysis
            • Ride-sharing driver-passenger matching
            • Weather prediction using nearby station data
            
            **📱 Technology & Software:**
            • Search engines: Find similar documents/web pages
            • Spam detection using similar email patterns
            • Bug tracking: Find similar bug reports
            • Code recommendation systems
            • Network intrusion detection
            
            **🔬 Scientific Research:**
            • Gene classification in bioinformatics
            • Drug discovery: Find compounds with similar properties
            • Astronomical object classification
            • Chemical compound property prediction
            • Climate pattern analysis
            
            **🛒 E-commerce & Retail:**
            • Customer segmentation for targeted marketing
            • Fraud detection in transactions
            • Inventory management based on similar products
            • Price comparison and optimization
            • Supply chain optimization
            
            **🏥 Healthcare & Medicine:**
            • Disease diagnosis based on symptom similarity
            • Drug dosage recommendations
            • Patient risk assessment
            • Clinical trial patient matching
            • Medical research data analysis
            
            **💡 Key Success Factors:**
            • Proper feature engineering and selection
            • Appropriate distance metric for data type
            • Feature scaling and preprocessing
            • Optimal k selection through validation
            • Computational efficiency considerations
            • Domain expertise in similarity definition
            """
        }
    
    def generate_sample_data(self, task_type, n_samples=300, n_features=4):
        """Generate sample data for demonstration."""
        if task_type == 'classification':
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=max(2, n_features // 2),
                n_redundant=0,
                n_clusters_per_class=1,
                random_state=42
            )
        else:  # regression
            X, y = make_regression(
                n_samples=n_samples,
                n_features=n_features,
                noise=0.1,
                random_state=42
            )
        return X, y
    
    def fit(self, X, y):
        """Fit the KNN model."""
        if self.task_type == 'classification':
            self.model = KNeighborsClassifier(
                n_neighbors=self.n_neighbors,
                metric=self.metric
            )
        else:
            self.model = KNeighborsRegressor(
                n_neighbors=self.n_neighbors,
                metric=self.metric
            )
        
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Make predictions using the fitted model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities (classification only)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        if self.task_type != 'classification':
            raise ValueError("Probability prediction only available for classification")
        return self.model.predict_proba(X)
    
    def get_metrics(self, X, y):
        """Calculate and return model performance metrics."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calculating metrics")
            
        y_pred = self.predict(X)
        
        if self.task_type == 'classification':
            metrics = {
                'Accuracy': accuracy_score(y, y_pred),
                'Precision': precision_score(y, y_pred, average='weighted', zero_division=0),
                'Recall': recall_score(y, y_pred, average='weighted', zero_division=0),
                'F1-Score': f1_score(y, y_pred, average='weighted', zero_division=0)
            }
        else:  # regression
            metrics = {
                'Mean Squared Error (MSE)': mean_squared_error(y, y_pred),
                'Root Mean Squared Error (RMSE)': np.sqrt(mean_squared_error(y, y_pred)),
                'Mean Absolute Error (MAE)': mean_absolute_error(y, y_pred),
                'R² Score': r2_score(y, y_pred)
            }
        
        return metrics
    
    def find_optimal_k(self, X, y, max_k=20):
        """Find optimal k using cross-validation."""
        k_range = range(1, min(max_k + 1, len(X) // 5))  # Ensure we don't exceed reasonable limits
        cv_scores = []
        
        for k in k_range:
            if self.task_type == 'classification':
                knn = KNeighborsClassifier(n_neighbors=k, metric=self.metric)
                scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
            else:
                knn = KNeighborsRegressor(n_neighbors=k, metric=self.metric)
                scores = cross_val_score(knn, X, y, cv=5, scoring='r2')
            
            cv_scores.append(scores.mean())
        
        return k_range, cv_scores
    
    def plot_optimal_k(self, X, y, max_k=20):
        """Plot cross-validation scores vs k to find optimal k."""
        k_range, cv_scores = self.find_optimal_k(X, y, max_k)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(k_range, cv_scores, 'bo-', linewidth=2, markersize=8)
        ax.set_xlabel('Number of Neighbors (k)')
        
        if self.task_type == 'classification':
            ax.set_ylabel('Cross-Validation Accuracy')
            ax.set_title('Optimal k Selection: Cross-Validation Accuracy vs k')
        else:
            ax.set_ylabel('Cross-Validation R² Score')
            ax.set_title('Optimal k Selection: Cross-Validation R² vs k')
        
        ax.grid(True, alpha=0.3)
        
        # Highlight optimal k
        optimal_k_idx = np.argmax(cv_scores)
        optimal_k = k_range[optimal_k_idx]
        ax.axvline(x=optimal_k, color='red', linestyle='--', 
                  label=f'Optimal k={optimal_k}')
        ax.legend()
        
        return fig, optimal_k
    
    def plot_decision_boundary(self, X, y, title="KNN Decision Boundary"):
        """Plot decision boundary for 2D classification data."""
        if X.shape[1] != 2:
            raise ValueError("Decision boundary plot only available for 2D data")
            
        if self.task_type != 'classification':
            raise ValueError("Decision boundary plot only available for classification")
            
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create a mesh for plotting decision boundary
        h = 0.02  # Step size in the mesh
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                           np.arange(y_min, y_max, h))
        
        # Predict on the mesh
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = self.model.predict(mesh_points)
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        contour = ax.contourf(xx, yy, Z, alpha=0.6, cmap=plt.cm.RdYlBu)
        
        # Plot data points
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
        
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title(f'{title} (k={self.n_neighbors})')
        
        # Add colorbar
        plt.colorbar(contour, ax=ax, label='Predicted Class')
        
        return fig
    
    def plot_distance_visualization(self, X, y, query_point_idx=0):
        """Visualize distances and neighbors for a specific query point."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting")
            
        if X.shape[1] != 2:
            st.warning("Distance visualization only available for 2D data")
            return None
        
        query_point = X[query_point_idx:query_point_idx+1]
        
        # Calculate distances to all points
        distances = self.model.kneighbors(query_point, n_neighbors=len(X), 
                                        return_distance=True)
        neighbor_distances = distances[0][0]
        neighbor_indices = distances[1][0]
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: All points with distances to query point
        axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.6, 
                       s=50, edgecolors='black')
        
        # Highlight query point
        axes[0].scatter(X[query_point_idx, 0], X[query_point_idx, 1], 
                       c='red', s=200, marker='*', edgecolor='black', 
                       label='Query Point')
        
        # Highlight k nearest neighbors
        k_neighbors_idx = neighbor_indices[:self.n_neighbors]
        axes[0].scatter(X[k_neighbors_idx, 0], X[k_neighbors_idx, 1], 
                       s=100, facecolors='none', edgecolors='red', 
                       linewidth=3, label=f'{self.n_neighbors}-Nearest Neighbors')
        
        # Draw lines to k nearest neighbors
        for idx in k_neighbors_idx:
            axes[0].plot([X[query_point_idx, 0], X[idx, 0]], 
                        [X[query_point_idx, 1], X[idx, 1]], 
                        'r--', alpha=0.6, linewidth=1)
        
        axes[0].set_xlabel('Feature 1')
        axes[0].set_ylabel('Feature 2')
        axes[0].set_title(f'K-Nearest Neighbors Visualization (k={self.n_neighbors})')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Distance distribution
        axes[1].bar(range(len(neighbor_distances)), neighbor_distances, 
                   alpha=0.7, color='skyblue', edgecolor='black')
        
        # Highlight k nearest neighbors
        axes[1].bar(range(self.n_neighbors), neighbor_distances[:self.n_neighbors], 
                   alpha=0.9, color='red', edgecolor='black', 
                   label=f'{self.n_neighbors}-Nearest Neighbors')
        
        axes[1].set_xlabel('Point Index (sorted by distance)')
        axes[1].set_ylabel('Distance from Query Point')
        axes[1].set_title('Distance Distribution from Query Point')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_classification_results(self, X, y):
        """Create classification visualization."""
        if self.task_type != 'classification':
            raise ValueError("Classification plots only available for classification tasks")
            
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting")
            
        y_pred = self.predict(X)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Confusion Matrix
        cm = confusion_matrix(y, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        axes[0].set_title('Confusion Matrix')
        
        # Class distribution
        unique_classes = np.unique(y)
        class_counts_actual = [np.sum(y == cls) for cls in unique_classes]
        class_counts_pred = [np.sum(y_pred == cls) for cls in unique_classes]
        
        x = np.arange(len(unique_classes))
        width = 0.35
        
        axes[1].bar(x - width/2, class_counts_actual, width, 
                   label='Actual', alpha=0.7, color='skyblue', edgecolor='black')
        axes[1].bar(x + width/2, class_counts_pred, width,
                   label='Predicted', alpha=0.7, color='lightcoral', edgecolor='black')
        
        axes[1].set_xlabel('Class')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Class Distribution: Actual vs Predicted')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([f'Class {cls}' for cls in unique_classes])
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_regression_results(self, X, y):
        """Create regression visualization."""
        if self.task_type != 'regression':
            raise ValueError("Regression plots only available for regression tasks")
            
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting")
            
        y_pred = self.predict(X)
        residuals = y - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Actual vs Predicted
        axes[0, 0].scatter(y, y_pred, alpha=0.6, color='blue')
        axes[0, 0].plot([y.min(), y.max()], [y.min(), y.max()], 
                       'r--', linewidth=2, label='Perfect Prediction')
        axes[0, 0].set_xlabel('Actual')
        axes[0, 0].set_ylabel('Predicted')
        axes[0, 0].set_title('Actual vs Predicted')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residual plot
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6, color='green')
        axes[0, 1].axhline(y=0, color='red', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residual Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Histogram of residuals
        axes[1, 0].hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='orange')
        axes[1, 0].axvline(np.mean(residuals), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(residuals):.4f}')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Residuals')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Model complexity visualization
        if X.shape[1] == 1:
            # For 1D input, show the prediction function
            x_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
            y_plot = self.predict(x_plot)
            
            axes[1, 1].scatter(X, y, alpha=0.6, color='blue', label='Training Data')
            axes[1, 1].plot(x_plot, y_plot, color='red', linewidth=2, 
                           label=f'KNN (k={self.n_neighbors})')
            axes[1, 1].set_xlabel('Feature')
            axes[1, 1].set_ylabel('Target')
            axes[1, 1].set_title('KNN Regression Function')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            # For multi-dimensional input, show feature importance (mock)
            axes[1, 1].text(0.5, 0.5, 'Feature analysis\nnot available\nfor multi-dimensional data', 
                           ha='center', va='center', transform=axes[1, 1].transAxes,
                           fontsize=12, bbox=dict(boxstyle="round", facecolor="lightgray"))
            axes[1, 1].set_title('Model Complexity')
        
        plt.tight_layout()
        return fig
    
    def streamlit_interface(self):
        """Create comprehensive Streamlit interface for K-Nearest Neighbors."""
        st.subheader("👥 K-Nearest Neighbors (KNN)")
        
        theory = self.get_theory()
        
        # Main tabs for comprehensive coverage
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🎯 Overview", "📚 Deep Dive", "💻 Implementation", 
            "🧪 Interactive Demo", "❓ Q&A", "🏢 Applications"
        ])
        
        with tab1:
            # Overview Tab - Essential Information
            st.markdown("### 🎯 What is K-Nearest Neighbors?")
            st.markdown(theory['definition'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 🌟 Why Use It?")
                st.markdown(theory['motivation'])
                
            with col2:
                st.markdown("### 🎉 Simple Analogy")
                st.markdown(theory['intuition'])
            
            # Quick advantages/disadvantages
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### ✅ Pros")
                st.markdown(theory['pros_cons'].split('🔹 **Disadvantages**')[0])
                
            with col2:
                st.markdown("### ❌ Cons")
                if '🔹 **Disadvantages**' in theory['pros_cons']:
                    st.markdown("🔹 **Disadvantages**" + theory['pros_cons'].split('🔹 **Disadvantages**')[1])
        
        with tab2:
            # Deep Dive Tab - Mathematical and Technical Details
            st.markdown("### 📊 Mathematical Foundation")
            st.markdown(theory['math_foundation'])
            
            st.markdown("### 🔄 Algorithm Steps")
            st.markdown(theory['algorithm_steps'])
            
            st.markdown("### 💾 Pseudocode")
            st.markdown(theory['pseudocode'])
            
            st.markdown("### ⚡ Time & Space Complexity")
            st.markdown(theory['complexity'])
            
        with tab3:
            # Implementation Tab
            st.markdown("### 💻 Python Implementation")
            st.markdown(theory['python_implementation'])
            
            st.markdown("### 📋 Complete Example")
            st.markdown(theory['example'])
            
            st.markdown("### 📈 Visualization Guide")
            st.markdown(theory['visualization'])
        
        with tab4:
            # Interactive Demo Tab
            st.markdown("### 🧪 Try K-Nearest Neighbors Yourself!")
            self._create_interactive_demo()
        
        with tab5:
            # Q&A Tab
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 🎯 When to Use")
                st.markdown(theory['usage_guide'])
                
                st.markdown("### 🚫 Common Mistakes")
                st.markdown(theory['common_mistakes'])
                
            with col2:
                st.markdown("### ❓ Interview Questions")
                st.markdown(theory['interview_questions'])
                
                st.markdown("### ⚖️ Algorithm Comparisons")
                st.markdown(theory['comparisons'])
        
        with tab6:
            # Applications Tab
            st.markdown("### 🌍 Real-World Applications")
            st.markdown(theory['real_world_applications'])
    
    def _create_interactive_demo(self):
        """Create the interactive demo section."""
        
        # Parameters section
        st.markdown("### 🔧 Parameters")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            task_type = st.selectbox("Task Type:", ['classification', 'regression'])
            
        with col2:
            n_samples = st.slider("Number of samples:", 100, 1000, 300)
            
        with col3:
            n_features = st.slider("Number of features:", 2, 8, 4)
            
        with col4:
            metric = st.selectbox("Distance metric:", ['euclidean', 'manhattan', 'cosine'])
        
        # Update task type and metric
        self.task_type = task_type
        self.metric = metric
        
        # Generate and split data
        X, y = self.generate_sample_data(task_type, n_samples, n_features)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Feature scaling
        scale_features = st.checkbox("Standardize features", value=True)
        if scale_features:
            scaler = StandardScaler()
            X_train_processed = scaler.fit_transform(X_train)
            X_test_processed = scaler.transform(X_test)
        else:
            X_train_processed = X_train
            X_test_processed = X_test
        
        # Find optimal k
        st.markdown("### 📊 Finding Optimal k")
        
        max_k = min(20, len(X_train_processed) // 5)
        fig_k, optimal_k = self.plot_optimal_k(X_train_processed, y_train, max_k)
        st.pyplot(fig_k)
        plt.close()
        
        st.info(f"Suggested optimal k: {optimal_k}")
        
        # KNN parameters
        st.markdown("### ⚙️ KNN Configuration")
        n_neighbors = st.slider("Number of neighbors (k):", 1, max_k, min(optimal_k, max_k))
        
        # Update model parameters and fit
        self.n_neighbors = n_neighbors
        self.fit(X_train_processed, y_train)
        
        # Results section
        st.markdown("### 📊 Results")
        
        # Metrics
        train_metrics = self.get_metrics(X_train_processed, y_train)
        test_metrics = self.get_metrics(X_test_processed, y_test)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Training Metrics:**")
            for metric_name, value in train_metrics.items():
                st.metric(metric_name, f"{value:.4f}")
                
        with col2:
            st.markdown("**Test Metrics:**")
            for metric_name, value in test_metrics.items():
                st.metric(metric_name, f"{value:.4f}")
        
        # Visualizations
        st.markdown("### 📈 Visualizations")
        
        # Decision boundary (only for 2D classification)
        if n_features == 2 and task_type == 'classification':
            st.markdown("#### Decision Boundary")
            fig_boundary = self.plot_decision_boundary(X_test_processed, y_test)
            st.pyplot(fig_boundary)
            plt.close()
            
            # Distance visualization
            st.markdown("#### Distance Visualization")
            query_idx = st.slider("Select query point:", 0, len(X_test_processed)-1, 0)
            fig_distance = self.plot_distance_visualization(X_test_processed, y_test, query_idx)
            if fig_distance:
                st.pyplot(fig_distance)
                plt.close()
        
        # Task-specific results
        if task_type == 'classification':
            fig_results = self.plot_classification_results(X_test_processed, y_test)
            st.pyplot(fig_results)
            plt.close()
            
            # Classification report
            y_pred_test = self.predict(X_test_processed)
            report = classification_report(y_test, y_pred_test, output_dict=True)
            
            st.markdown("**Detailed Classification Report:**")
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.round(4))
            
        else:  # regression
            fig_results = self.plot_regression_results(X_test_processed, y_test)
            st.pyplot(fig_results)
            plt.close()
        
        # Interpretation
        st.markdown("### 🔍 Interpretation")
        
        if task_type == 'classification':
            accuracy = test_metrics['Accuracy']
            if accuracy > 0.9:
                st.success(f"**Excellent performance!** Accuracy: {accuracy:.1%}")
            elif accuracy > 0.8:
                st.info(f"**Good performance.** Accuracy: {accuracy:.1%}")
            elif accuracy > 0.7:
                st.warning(f"**Moderate performance.** Accuracy: {accuracy:.1%}")
            else:
                st.error(f"**Poor performance.** Accuracy: {accuracy:.1%}")
        else:
            r2_score_val = test_metrics['R² Score']
            if r2_score_val > 0.8:
                st.success(f"**Excellent fit!** R² Score: {r2_score_val:.3f}")
            elif r2_score_val > 0.6:
                st.info(f"**Good fit.** R² Score: {r2_score_val:.3f}")
            elif r2_score_val > 0.3:
                st.warning(f"**Moderate fit.** R² Score: {r2_score_val:.3f}")
            else:
                st.error(f"**Poor fit.** R² Score: {r2_score_val:.3f}")
        
        # Model characteristics
        st.markdown("**Model Characteristics:**")
        col1, col2, col3 = st.columns(3)
        col1.metric("k (neighbors)", self.n_neighbors)
        col2.metric("Distance Metric", self.metric.capitalize())
        col3.metric("Feature Scaling", "Yes" if scale_features else "No")
        
        # Recommendations
        st.markdown("**Recommendations:**")
        if self.n_neighbors == 1:
            st.write("• k=1 may lead to overfitting. Consider increasing k.")
        if not scale_features and self.metric == 'euclidean':
            st.write("• Feature scaling is recommended for Euclidean distance.")
        if n_features > 10:
            st.write("• High dimensionality may affect KNN performance. Consider dimensionality reduction.")


def main():
    """Main function for testing K-Nearest Neighbors."""
    knn = KNearestNeighbors()
    knn.streamlit_interface()


if __name__ == "__main__":
    main()