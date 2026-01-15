"""
Support Vector Machine (SVM) Algorithm Implementation

Support Vector Machines are powerful supervised learning models used for
classification and regression that work by finding the optimal hyperplane
to separate different classes.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC, SVR
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report,
                           mean_squared_error, r2_score, mean_absolute_error)
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler
import streamlit as st


class SupportVectorMachine:
    """
    Support Vector Machine implementation with educational explanations.
    
    SVM finds the optimal hyperplane that maximizes the margin between
    different classes. It can handle both linear and non-linear relationships
    using kernel functions.
    """
    
    def __init__(self, task_type='classification', kernel='rbf', C=1.0):
        self.task_type = task_type
        self.kernel = kernel
        self.C = C
        self.model = None
        self.is_fitted = False
        
    @staticmethod
    def get_theory():
        """Return comprehensive theoretical explanation of Support Vector Machine."""
        return {
            'name': 'Support Vector Machine (SVM)',
            'type': 'Supervised Learning - Classification/Regression',
            
            # 1. What the algorithm is
            'definition': """
            🔹 **What is Support Vector Machine?**
            SVM is like finding the perfect boundary line that separates different groups with the 
            maximum safety margin. Think of it as drawing the widest possible "no-man's land" 
            between opposing teams on a battlefield.
            """,
            
            # 2. Why the algorithm is used
            'motivation': """
            🔹 **Why Use Support Vector Machine?**
            • 📄 **Text Classification**: Spam detection, document categorization
            • 🖼️ **Image Recognition**: Face detection, object classification
            • 🧬 **Bioinformatics**: Gene classification, protein analysis
            • 📊 **High-Dimensional Data**: Works well when features > samples
            • 🎯 **Small Datasets**: Effective with limited training data
            • 🔬 **Scientific Research**: Reliable for research applications
            """,
            
            # 3. Intuition with real-life analogy
            'intuition': """
            🔹 **Real-Life Analogy: The Optimal Highway Construction**
            
            Imagine you're a civil engineer designing a highway between two cities:
            
            🏙️ **The Problem**: Build highway separating City A from City B
            🎯 **Goal**: Maximum safety margin (widest median strip possible)
            
            **SVM is like a master engineer who:**
            
            **Step 1**: 📏 Measures distance to nearest buildings on each side
            **Step 2**: 🎯 Finds the line that maximizes distance to BOTH sides
            **Step 3**: 🏗️ Builds highway along this optimal line
            **Step 4**: 🛡️ The nearest buildings become "support vectors"
            
            **Why this works:**
            • **Maximum Margin**: Safest possible separation
            • **Support Vectors**: Only critical buildings matter for placement
            • **Robust Design**: Small building changes don't affect highway
            • **Optimal Solution**: Mathematically proven best placement
            
            **For non-linear terrain** (curved boundaries):
            • Use **Kernel Trick**: Transform terrain into higher dimension
            • Find straight highway in new dimension
            • Project back to get curved path in original terrain
            
            🎯 **In data terms**: 
            - Cities = Classes
            - Buildings = Data Points
            - Highway = Decision Boundary
            - Nearest Buildings = Support Vectors
            - Median Strip Width = Margin
            """,
            
            # 4. Mathematical foundation
            'math_foundation': """
            🔹 **Mathematical Foundation (Step-by-Step)**
            
            **Optimization Problem (Linear SVM):**
            ```
            Minimize: (1/2) ||w||² + C Σᵢ ξᵢ
            Subject to: yᵢ(w·xᵢ + b) ≥ 1 - ξᵢ, ξᵢ ≥ 0
            ```
            
            **Where:**
            • `w` = Weight vector (perpendicular to decision boundary)
            • `||w||` = Magnitude of weight vector
            • `C` = Regularization parameter (trade-off between margin and errors)
            • `ξᵢ` = Slack variables (allow some misclassification)
            • `b` = Bias term (intercept)
            • `yᵢ` = Class label (+1 or -1)
            
            **Decision Function:**
            ```
            f(x) = sign(w·x + b) = sign(Σᵢ αᵢ yᵢ K(xᵢ, x) + b)
            ```
            
            **Margin Calculation:**
            ```
            Margin = 2 / ||w||
            # Goal: Maximize margin → Minimize ||w||
            ```
            
            **Kernel Functions:**
            ```
            Linear: K(xᵢ, xⱼ) = xᵢ·xⱼ
            Polynomial: K(xᵢ, xⱼ) = (γ xᵢ·xⱼ + r)ᵈ
            RBF: K(xᵢ, xⱼ) = exp(-γ ||xᵢ - xⱼ||²)
            Sigmoid: K(xᵢ, xⱼ) = tanh(γ xᵢ·xⱼ + r)
            ```
            
            **Support Vectors:**
            ```
            Support vectors are points where αᵢ > 0
            These are the only points that matter for the decision boundary
            ```
            
            **Dual Formulation (Lagrangian):**
            ```
            Maximize: Σᵢ αᵢ - (1/2) Σᵢ Σⱼ αᵢ αⱼ yᵢ yⱼ K(xᵢ, xⱼ)
            Subject to: Σᵢ αᵢ yᵢ = 0, 0 ≤ αᵢ ≤ C
            ```
            """,
            
            # 5. Step-by-step working
            'algorithm_steps': """
            🔹 **How SVM Works (Step-by-Step)**
            
            **Step 1: Data Preparation** 📋
            • Collect labeled training data (X, y)
            • Scale features to similar ranges (very important for SVM!)
            • Choose kernel function based on data characteristics
            
            **Step 2: Formulate Optimization** 🎯
            • Set up constrained optimization problem
            • Goal: Find hyperplane with maximum margin
            • Balance between margin size and classification errors
            
            **Step 3: Solve Dual Problem** 🔢
            • Convert to dual optimization (easier to solve)
            • Use quadratic programming or SMO (Sequential Minimal Optimization)
            • Find Lagrange multipliers (αᵢ) for each training point
            
            **Step 4: Identify Support Vectors** 🎯
            • Support vectors are points where αᵢ > 0
            • These points lie on or inside the margin
            • Only these points influence the decision boundary
            
            **Step 5: Construct Decision Function** 📐
            • Use support vectors to build decision function
            • Non-support vectors are discarded (αᵢ = 0)
            • Calculate bias term using support vectors
            
            **Step 6: Handle Non-Linear Data** 🌀
            • Apply kernel trick for non-linear separation
            • Transform data to higher dimensional space implicitly
            • Find linear separator in new space
            
            **Step 7: Make Predictions** 🔮
            • For new point: f(x) = Σᵢ αᵢ yᵢ K(xᵢ, x) + b
            • Classification: sign(f(x)) → class label
            • Regression: f(x) → continuous value
            
            **Step 8: Evaluate Performance** ✅
            • Test on unseen data
            • Monitor for overfitting (especially with complex kernels)
            """,
            
            # 6. Pseudocode
            'pseudocode': """
            🔹 **Pseudocode (Easy to Understand)**
            
            ```
            ALGORITHM: Support Vector Machine
            
            INPUT: 
                - X: feature matrix (n_samples × n_features)
                - y: target values (n_samples × 1)
                - kernel: kernel function type
                - C: regularization parameter
            
            OUTPUT:
                - support_vectors: critical data points
                - alpha: Lagrange multipliers
                - bias: intercept term
            
            BEGIN
                1. PREPROCESS data:
                   X_scaled = standardize(X)  # Very important!
                
                2. SETUP optimization problem:
                   MINIMIZE: (1/2) ||w||² + C × Σ(slack_variables)
                   SUBJECT TO: yᵢ(w·xᵢ + b) ≥ 1 - ξᵢ
                
                3. SOLVE dual problem:
                   MAXIMIZE: Σ αᵢ - (1/2) Σᵢ Σⱼ αᵢ αⱼ yᵢ yⱼ K(xᵢ, xⱼ)
                   SUBJECT TO: Σ αᵢ yᵢ = 0, 0 ≤ αᵢ ≤ C
                   
                   # Use SMO (Sequential Minimal Optimization)
                   REPEAT until convergence:
                       SELECT two variables αᵢ, αⱼ to optimize
                       SOLVE 2-variable sub-problem analytically
                       UPDATE αᵢ, αⱼ
                
                4. IDENTIFY support vectors:
                   support_vectors = {xᵢ where αᵢ > 0}
                   
                5. CALCULATE bias:
                   FOR each support vector on margin (0 < αᵢ < C):
                       b = yᵢ - Σⱼ αⱼ yⱼ K(xⱼ, xᵢ)
                   bias = average(all_bias_calculations)
                
                6. RETURN (support_vectors, alpha, bias)
            END
            
            PREDICTION:
            BEGIN
                1. FOR new sample x_new:
                   score = Σᵢ αᵢ yᵢ K(support_vector_i, x_new) + bias
                   
                2. IF classification:
                   prediction = sign(score)
                   
                3. IF regression:
                   prediction = score
                   
                4. RETURN prediction
            END
            ```
            """,
            
            # 7. Python implementation
            'python_implementation': """
            🔹 **Python Implementation**
            
            **From Scratch (Simplified Linear SVM):**
            ```python
            import numpy as np
            from scipy.optimize import minimize
            
            class SimpleSVM:
                def __init__(self, C=1.0, max_iter=1000):
                    self.C = C
                    self.max_iter = max_iter
                    self.w = None
                    self.b = None
                    self.support_vectors = None
                    self.alpha = None
                
                def linear_kernel(self, x1, x2):
                    \"\"\"Linear kernel function.\"\"\"
                    return np.dot(x1, x2)
                
                def objective(self, alpha, y, X):
                    \"\"\"Objective function to maximize (dual problem).\"\"\"
                    n = len(alpha)
                    # Compute kernel matrix
                    K = np.zeros((n, n))
                    for i in range(n):
                        for j in range(n):
                            K[i, j] = self.linear_kernel(X[i], X[j])
                    
                    # Dual objective function
                    obj = np.sum(alpha) - 0.5 * np.sum(alpha[:, None] * alpha * y[:, None] * y * K)
                    return -obj  # Minimize negative = maximize
                
                def constraint1(self, alpha, y):
                    \"\"\"Constraint: sum(alpha * y) = 0\"\"\"
                    return np.sum(alpha * y)
                
                def fit(self, X, y):
                    \"\"\"Train the SVM.\"\"\"
                    n_samples, n_features = X.shape
                    
                    # Initial guess for alpha
                    alpha0 = np.random.random(n_samples)
                    
                    # Constraints
                    constraints = [{
                        'type': 'eq',
                        'fun': lambda alpha: self.constraint1(alpha, y)
                    }]
                    
                    # Bounds for alpha (0 <= alpha <= C)
                    bounds = [(0, self.C) for _ in range(n_samples)]
                    
                    # Solve optimization problem
                    result = minimize(
                        fun=lambda alpha: self.objective(alpha, y, X),
                        x0=alpha0,
                        method='SLSQP',
                        bounds=bounds,
                        constraints=constraints
                    )
                    
                    self.alpha = result.x
                    
                    # Find support vectors (alpha > threshold)
                    sv_threshold = 1e-5
                    sv_indices = self.alpha > sv_threshold
                    self.support_vectors = X[sv_indices]
                    self.sv_labels = y[sv_indices]
                    self.sv_alpha = self.alpha[sv_indices]
                    
                    # Calculate weights
                    self.w = np.sum((self.sv_alpha * self.sv_labels)[:, None] * self.support_vectors, axis=0)
                    
                    # Calculate bias using support vectors
                    self.b = np.mean(self.sv_labels - np.dot(self.support_vectors, self.w))
                
                def predict(self, X):
                    \"\"\"Make predictions.\"\"\"
                    scores = np.dot(X, self.w) + self.b
                    return np.sign(scores)
                
                def decision_function(self, X):
                    \"\"\"Return decision scores.\"\"\"
                    return np.dot(X, self.w) + self.b
            
            # Example usage
            from sklearn.datasets import make_classification
            from sklearn.preprocessing import StandardScaler
            
            # Generate data
            X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                                     n_informative=2, random_state=42)
            y[y == 0] = -1  # Convert to -1, +1 labels
            
            # Scale features (important for SVM!)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train SVM
            svm = SimpleSVM(C=1.0)
            svm.fit(X_scaled, y)
            
            # Make predictions
            predictions = svm.predict(X_scaled)
            ```
            
            **Using Scikit-learn:**
            ```python
            from sklearn.svm import SVC, SVR
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, classification_report
            
            # Classification
            # Data preprocessing (CRITICAL for SVM)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train SVM classifier
            svm_clf = SVC(
                kernel='rbf',          # RBF kernel for non-linear data
                C=1.0,                # Regularization parameter
                gamma='scale',        # Kernel coefficient
                probability=True,     # Enable probability estimates
                random_state=42
            )
            svm_clf.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = svm_clf.predict(X_test_scaled)
            y_proba = svm_clf.predict_proba(X_test_scaled)
            
            # Get support vectors
            support_vectors = svm_clf.support_vectors_
            n_support = svm_clf.n_support_
            
            print(f"Number of support vectors: {len(support_vectors)}")
            print(f"Support vectors per class: {n_support}")
            
            # Regression
            svm_reg = SVR(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                epsilon=0.1          # Tolerance for regression
            )
            svm_reg.fit(X_train_scaled, y_train)
            y_pred_reg = svm_reg.predict(X_test_scaled)
            ```
            
            **Kernel Examples:**
            ```python
            # Different kernels for different data types
            
            # Linear kernel (for linearly separable data)
            svm_linear = SVC(kernel='linear', C=1.0)
            
            # Polynomial kernel (for polynomial boundaries)
            svm_poly = SVC(kernel='poly', degree=3, C=1.0)
            
            # RBF kernel (most popular, for complex non-linear data)
            svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
            
            # Custom kernel
            def my_kernel(X, Y):
                return np.dot(X, Y.T)  # Same as linear
                
            svm_custom = SVC(kernel=my_kernel)
            ```
            """,
            
            # 8. Example with sample input/output
            'example': """
            🔹 **Complete Example: Email Spam Classification**
            
            **Input Data (Email Features):**
            ```
            Email | URGENT_words | Money_words | Length | Links | Spam?
            1     | 0           | 1           | 50     | 1     | No (-1)
            2     | 5           | 3           | 200    | 8     | Yes (+1)
            3     | 1           | 0           | 100    | 2     | No (-1)
            4     | 3           | 5           | 150    | 10    | Yes (+1)
            5     | 0           | 0           | 80     | 0     | No (-1)
            6     | 4           | 2           | 120    | 6     | Yes (+1)
            ```
            
            **Step-by-Step SVM Training:**
            ```
            1. Scale Features (Critical!):
               URGENT_words: [0, 5, 1, 3, 0, 4] → [-1.2, 1.8, -0.8, 0.4, -1.2, 1.0]
               Money_words: [1, 3, 0, 5, 0, 2] → [-0.5, 0.5, -1.0, 2.0, -1.0, 0]
               Length: [50, 200, 100, 150, 80, 120] → [-1.5, 1.8, -0.3, 0.9, -1.0, 0.1]
               Links: [1, 8, 2, 10, 0, 6] → [-1.0, 1.2, -0.8, 1.8, -1.2, 0.6]
            
            2. Find Optimal Hyperplane:
               Goal: Maximize margin between spam (+1) and non-spam (-1)
               
            3. Solve Optimization:
               Found: w = [0.8, 1.2, 0.3, 0.9], b = -0.1
               
            4. Identify Support Vectors:
               Email 2: [5, 3, 200, 8] → Support vector (closest spam)
               Email 3: [1, 0, 100, 2] → Support vector (closest non-spam)
               Email 4: [3, 5, 150, 10] → Support vector (on margin)
               
            5. Decision Boundary:
               0.8×URGENT + 1.2×Money + 0.3×Length + 0.9×Links - 0.1 = 0
            ```
            
            **New Email Prediction:**
            ```
            New Email: URGENT=2, Money=1, Length=75, Links=3
            
            1. Scale features: [2, 1, 75, 3] → [-0.4, -0.5, -1.1, -0.6]
            
            2. Calculate decision score:
               f(x) = 0.8×(-0.4) + 1.2×(-0.5) + 0.3×(-1.1) + 0.9×(-0.6) - 0.1
                    = -0.32 - 0.6 - 0.33 - 0.54 - 0.1 = -1.89
            
            3. Make prediction:
               Since f(x) = -1.89 < 0 → Prediction: NOT SPAM ✅
               
            4. Confidence:
               Distance from boundary = |f(x)| = 1.89
               High confidence (far from decision boundary)
            ```
            
            **Support Vector Interpretation:**
            ```
            Only 3 out of 6 emails matter for the decision boundary!
            These support vectors define the optimal separation.
            If we remove other emails, decision boundary stays the same.
            ```
            """,
            
            # 9. Visualization explanation
            'visualization': """
            🔹 **Understanding Through Visualizations**
            
            **2D Decision Boundary:**
            📊 Shows hyperplane separating classes
            • Solid line = decision boundary
            • Dashed lines = margin boundaries
            • Circled points = support vectors
            • Shaded regions = class predictions
            
            **Margin Visualization:**
            📏 Shows width of separation corridor
            • Wider margin = better generalization
            • Support vectors define margin width
            • Goal: maximize this margin
            
            **Support Vector Highlighting:**
            🎯 Shows which points matter for decision
            • Support vectors are critical points
            • Other points can be removed without changing boundary
            • Usually only small fraction of data
            
            **Kernel Effect Comparison:**
            🔄 Side-by-side comparison of different kernels
            • Linear: Straight line boundaries
            • Polynomial: Curved boundaries
            • RBF: Complex, smooth boundaries
            • Shows how kernel choice affects decision regions
            
            **3D Kernel Transformation:**
            📈 Shows how kernel maps data to higher dimensions
            • Original 2D data becomes linearly separable in 3D
            • Hyperplane in 3D becomes curve in 2D
            • Illustrates the "kernel trick"
            
            **Regularization Effect (C parameter):**
            ⚖️ Shows trade-off between margin and errors
            • Low C: Wide margin, some misclassifications
            • High C: Narrow margin, fewer misclassifications
            • Helps understand bias-variance trade-off
            """,
            
            # 10. Time and space complexity
            'complexity': """
            🔹 **Time & Space Complexity**
            
            **Time Complexity:**
            • **Training**: O(n² × p) to O(n³ × p) where n=samples, p=features
            • **Prediction**: O(s × p) where s=number of support vectors
            • **SMO Algorithm**: O(n²) on average, O(n³) worst case
            • **Kernel Computation**: O(p) per kernel evaluation
            
            **Space Complexity:**
            • **Model Storage**: O(s × p) where s=support vectors
            • **Training Memory**: O(n²) for kernel matrix (can be reduced)
            • **Support Vector Storage**: Typically 10-50% of training data
            
            **Scalability:**
            • ✅ **High Dimensions**: Excellent with many features
            • ❌ **Large Datasets**: Poor scaling beyond 10K samples
            • ✅ **Sparse Data**: Works well with sparse feature vectors
            • ⚠️ **Memory Usage**: Kernel matrix can be huge
            • ✅ **Support Vectors**: Model size independent of training size
            
            **Optimization Notes:**
            • SMO breaks large problem into series of small 2-variable problems
            • Working set selection can improve convergence
            • Kernel caching reduces computation time
            • Feature scaling essential for performance
            """,
            
            # 11. Advantages and disadvantages
            'pros_cons': """
            🔹 **Advantages** ✅
            • **High-Dimensional Excellence**: Works great with many features
            • **Memory Efficient**: Only stores support vectors (not all data)
            • **Kernel Flexibility**: Can handle complex non-linear patterns
            • **Robust**: Resistant to overfitting in high dimensions
            • **Global Optimum**: Convex optimization guarantees global solution
            • **Theoretical Foundation**: Strong mathematical backing
            • **Small Dataset Friendly**: Effective with limited training data
            • **Versatile**: Works for classification and regression
            • **No Local Minima**: Always finds globally optimal solution
            
            🔹 **Disadvantages** ❌
            • **Slow on Large Data**: Poor scalability beyond ~10K samples
            • **Feature Scaling Required**: Very sensitive to feature scales
            • **No Probability Estimates**: Doesn't naturally output probabilities
            • **Black Box**: Difficult to interpret decision reasoning
            • **Hyperparameter Sensitive**: Requires careful tuning of C, gamma
            • **Memory Intensive**: Kernel matrix can be very large
            • **No Feature Importance**: Doesn't rank feature importance
            • **Outlier Sensitive**: Can be affected by noisy data
            • **Limited to Binary**: Needs modifications for multi-class problems
            """,
            
            # 12. When to use and when NOT to use
            'usage_guide': """
            🔹 **When TO Use SVM** ✅
            
            **Perfect for:**
            • 🔬 **High-Dimensional Data**: Text data, genomics (features >> samples)
            • 📄 **Text Classification**: Document classification, spam detection
            • 🖼️ **Image Recognition**: Face detection, handwriting recognition
            • 🧬 **Bioinformatics**: Gene classification, protein analysis
            • 📊 **Small Datasets**: When you have limited training data
            • 🎯 **Binary Classification**: Clear two-class problems
            
            **Good when:**
            • Need robust performance with small datasets
            • Features are more numerous than samples
            • Decision boundary is complex but smooth
            • Data is approximately linearly separable (with kernels)
            • You want guaranteed global optimum
            
            🔹 **When NOT to Use SVM** ❌
            
            **Avoid when:**
            • 📈 **Large Datasets**: More than 10,000 samples (use alternatives)
            • ⚡ **Real-time Predictions**: Need very fast inference
            • 🔍 **Interpretability Required**: Need to explain decisions
            • 📊 **Probability Estimates**: Need calibrated probability outputs
            • 🎯 **Multi-class Problems**: Many classes (>10)
            • 📉 **Noisy Data**: High noise levels in features or labels
            • 💰 **Limited Compute**: Constrained computational resources
            
            **Use instead:**
            • Random Forest (for interpretability)
            • Logistic Regression (for probabilities)
            • Gradient Boosting (for structured data)
            • Neural Networks (for very large datasets)
            • Naive Bayes (for very fast training/prediction)
            """,
            
            # 13. Common interview questions
            'interview_questions': """
            🔹 **Common Interview Questions & Answers**
            
            **Q1: What is the kernel trick and why is it important?**
            A: The kernel trick allows SVM to handle non-linear data without explicitly transforming to higher dimensions:
            • Instead of φ(x)·φ(y), compute K(x,y) directly
            • Saves computation and memory
            • Makes infinite-dimensional mappings possible
            • Examples: RBF kernel maps to infinite dimensions
            
            **Q2: How do you choose the right kernel?**
            A:
            • Linear: When data is linearly separable or has many features
            • Polynomial: When you suspect polynomial relationships
            • RBF: Default choice, works well for most non-linear data
            • Sigmoid: Rarely used, similar to neural networks
            • Try linear first (fastest), then RBF if needed
            
            **Q3: What's the difference between hard margin and soft margin SVM?**
            A:
            • Hard margin: No misclassification allowed, only works on linearly separable data
            • Soft margin: Allows some misclassification via slack variables
            • C parameter controls trade-off: high C → hard margin, low C → soft margin
            • Soft margin is more practical for real-world noisy data
            
            **Q4: Why is feature scaling crucial for SVM?**
            A: SVM finds optimal hyperplane based on distances:
            • Features with larger scales dominate the distance calculation
            • Example: Age (0-100) vs Income (0-100000) - income will dominate
            • Always use StandardScaler or MinMaxScaler before training
            • One of the most important preprocessing steps for SVM
            
            **Q5: How does SVM handle multi-class classification?**
            A: SVM is naturally binary, so multi-class needs strategies:
            • One-vs-One: Train classifier for each pair of classes
            • One-vs-Rest: Train one classifier per class vs all others
            • Scikit-learn automatically handles this with decision_function_shape parameter
            """,
            
            # 14. Common mistakes
            'common_mistakes': """
            🔹 **Common Beginner Mistakes & How to Avoid**
            
            **Mistake 1: Not scaling features** 🚫
            ❌ Training SVM on raw features with different scales
            ✅ **Fix**: Always use StandardScaler() or MinMaxScaler() first
            
            **Mistake 2: Using wrong kernel** 🚫
            ❌ Using linear kernel on clearly non-linear data
            ✅ **Fix**: Start with RBF kernel, try linear only for high-dim/sparse data
            
            **Mistake 3: Not tuning hyperparameters** 🚫
            ❌ Using default C=1.0 without testing other values
            ✅ **Fix**: Use GridSearchCV to tune C and gamma parameters
            
            **Mistake 4: Using SVM on large datasets** 🚫
            ❌ Training SVM on 100K+ samples and wondering why it's slow
            ✅ **Fix**: Use Random Forest or Gradient Boosting for large datasets
            
            **Mistake 5: Expecting probability outputs** 🚫
            ❌ Assuming SVM naturally outputs well-calibrated probabilities
            ✅ **Fix**: Use probability=True and consider calibration with CalibratedClassifierCV
            
            **Mistake 6: Ignoring outliers** 🚫
            ❌ Not removing obvious outliers that can skew support vectors
            ✅ **Fix**: Clean data and consider robust scaling methods
            
            **Mistake 7: Wrong evaluation metric** 🚫
            ❌ Using accuracy on imbalanced datasets
            ✅ **Fix**: Use F1-score, precision, recall for imbalanced classes
            """,
            
            # 15. Comparison with similar algorithms
            'comparisons': """
            🔹 **SVM vs Similar Algorithms**
            
            **SVM vs Logistic Regression:**
            • **SVM**: Maximum margin principle, uses support vectors
            • **Logistic Regression**: Maximum likelihood, uses all data points
            • **Use Logistic**: When you need probability estimates
            
            **SVM vs Random Forest:**
            • **SVM**: Better for high-dimensional data, mathematical guarantees
            • **Random Forest**: Better for large datasets, interpretable
            • **Use Random Forest**: For tabular data and feature importance
            
            **SVM vs Neural Networks:**
            • **SVM**: Convex optimization, no local minima, less data needed
            • **Neural Networks**: More flexible, better for very complex patterns
            • **Use Neural Networks**: With large datasets and complex patterns
            
            **SVM vs K-Nearest Neighbors:**
            • **SVM**: Global model, faster prediction, memory efficient
            • **KNN**: Local model, no training needed, adapts to local patterns
            • **Use KNN**: When local patterns matter more than global structure
            
            **Linear SVM vs Kernel SVM:**
            • **Linear SVM**: Faster training/prediction, works well in high dimensions
            • **Kernel SVM**: Handles non-linear patterns, more flexible
            • **Use Linear**: When data is high-dimensional or approximately linear
            """,
            
            # 16. Real-world applications
            'real_world_applications': """
            🔹 **Real-World Applications & Industry Use Cases**
            
            **📄 Text & Document Analysis:**
            • Spam email detection and filtering
            • Document classification and categorization
            • Sentiment analysis for social media
            • Language detection in multilingual documents
            • News article topic classification
            
            **🖼️ Computer Vision & Image Processing:**
            • Face detection and recognition systems
            • Handwritten digit recognition (postal services)
            • Medical image analysis (X-ray, MRI classification)
            • Object detection in security cameras
            • Quality control in manufacturing
            
            **🧬 Bioinformatics & Healthcare:**
            • Gene classification and protein analysis
            • Drug discovery and molecular design
            • Disease diagnosis from symptoms
            • Cancer detection from tissue samples
            • Personalized medicine recommendations
            
            **💰 Finance & Risk Management:**
            • Credit scoring and loan approval
            • Fraud detection in transactions
            • Stock market prediction
            • Risk assessment for insurance
            • Algorithmic trading strategies
            
            **🔬 Scientific Research:**
            • Particle physics data analysis
            • Climate change modeling
            • Chemical compound classification
            • Astronomical object detection
            • Materials science property prediction
            
            **📱 Technology & Software:**
            • Search engine ranking algorithms
            • Recommendation system components
            • Network intrusion detection
            • Software bug prediction
            • User behavior classification
            
            **🏭 Manufacturing & Quality Control:**
            • Defect detection in production lines
            • Predictive maintenance systems
            • Process optimization in chemical plants
            • Supply chain risk assessment
            • Equipment failure prediction
            
            **💡 Key Success Factors:**
            • Proper feature scaling and preprocessing
            • Careful hyperparameter tuning
            • Appropriate kernel selection for data type
            • Sufficient but not excessive training data
            • Domain expertise for feature engineering
            """
        }
    
    def generate_sample_data(self, task_type, n_samples=300, n_features=4):
        """Generate sample data for demonstration."""
        if task_type == 'classification':
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_redundant=0,
                n_informative=n_features,
                random_state=42
            )
        else:
            X, y = make_regression(
                n_samples=n_samples,
                n_features=n_features,
                noise=0.1,
                random_state=42
            )
        return X, y
    
    def fit(self, X, y):
        """Fit the SVM model."""
        if self.task_type == 'classification':
            self.model = SVC(kernel=self.kernel, C=self.C, random_state=42)
        else:
            self.model = SVR(kernel=self.kernel, C=self.C)
        
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Make predictions using the fitted model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)
    
    def get_metrics(self, X, y):
        """Calculate performance metrics for the model."""
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
                'MSE': mean_squared_error(y, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y, y_pred)),
                'MAE': mean_absolute_error(y, y_pred),
                'R² Score': r2_score(y, y_pred)
            }
        
        return metrics
    
    def streamlit_interface(self):
        """Create comprehensive Streamlit interface for Support Vector Machine."""
        st.subheader("🎯 Support Vector Machine (SVM)")
        
        theory = self.get_theory()
        
        # Main tabs for comprehensive coverage
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🎯 Overview", "📚 Deep Dive", "💻 Implementation", 
            "🧪 Interactive Demo", "❓ Q&A", "🏢 Applications"
        ])
        
        with tab1:
            # Overview Tab - Essential Information
            st.markdown("### 🎯 What is Support Vector Machine?")
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
            st.markdown("### 🧪 Try Support Vector Machine Yourself!")
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
            kernel = st.selectbox("Kernel:", ['linear', 'poly', 'rbf', 'sigmoid'])
        with col3:
            C = st.slider("Regularization (C):", 0.1, 10.0, 1.0, 0.1)
        with col4:
            n_samples = st.slider("Samples:", 100, 500, 300)
        
        # Update parameters
        self.task_type = task_type
        self.kernel = kernel
        self.C = C
        
        # Generate data and train model
        X, y = self.generate_sample_data(task_type, n_samples)
        
        # Feature scaling (important for SVM)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42)
        
        self.fit(X_train, y_train)
        
        # Results section
        st.markdown("### 📊 Results")
        
        # Basic metrics
        train_metrics = self.get_metrics(X_train, y_train)
        test_metrics = self.get_metrics(X_test, y_test)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Training Metrics:**")
            for metric_name, value in train_metrics.items():
                st.metric(metric_name, f"{value:.4f}")
                
        with col2:
            st.markdown("**Test Metrics:**")
            for metric_name, value in test_metrics.items():
                st.metric(metric_name, f"{value:.4f}")
        
        # Support vectors information
        if hasattr(self.model, 'support_vectors_'):
            st.markdown("### 🎯 Support Vector Information")
            col1, col2, col3 = st.columns(3)
            col1.metric("Number of Support Vectors", len(self.model.support_vectors_))
            if hasattr(self.model, 'n_support_'):
                col2.metric("Support Vectors per Class", str(self.model.n_support_))
            if hasattr(self.model, 'dual_coef_'):
                col3.metric("Dual Coefficients Shape", str(self.model.dual_coef_.shape))
        
        # Interpretation
        st.markdown("### 🔍 Interpretation")
        
        if task_type == 'classification':
            if 'Accuracy' in test_metrics:
                accuracy = test_metrics['Accuracy']
                if accuracy > 0.9:
                    st.success(f"**Excellent performance!** Accuracy: {accuracy:.3f}")
                    st.write("The SVM is performing very well on this dataset.")
                elif accuracy > 0.8:
                    st.info(f"**Good performance.** Accuracy: {accuracy:.3f}")
                    st.write("The SVM is performing well.")
                elif accuracy > 0.7:
                    st.warning(f"**Moderate performance.** Accuracy: {accuracy:.3f}")
                    st.write("Consider tuning hyperparameters or trying different kernels.")
                else:
                    st.error(f"**Poor performance.** Accuracy: {accuracy:.3f}")
                    st.write("The dataset might not be suitable for SVM or needs different preprocessing.")
        else:  # regression
            if 'R² Score' in test_metrics:
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
        col1.metric("Kernel", kernel.upper())
        col2.metric("Regularization (C)", f"{C}")
        col3.metric("Task", task_type.capitalize())


def main():
    """Main function for testing Support Vector Machine."""
    svm = SupportVectorMachine()
    svm.streamlit_interface()


if __name__ == "__main__":
    main()