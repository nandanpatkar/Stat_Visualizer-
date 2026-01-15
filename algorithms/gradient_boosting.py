"""
Gradient Boosting Algorithm Implementation

Gradient Boosting builds models sequentially, where each new model corrects
the errors made by previous models, resulting in a strong ensemble predictor.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report,
                           mean_squared_error, r2_score, mean_absolute_error)
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_regression
import streamlit as st


class GradientBoosting:
    """
    Gradient Boosting implementation with educational explanations.
    
    Builds models in a stage-wise fashion, optimizing for any differentiable
    loss function by fitting new models to residual errors.
    """
    
    def __init__(self, n_estimators=100, learning_rate=0.1, task_type='classification'):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.task_type = task_type
        self.model = None
        self.is_fitted = False
        
    @staticmethod
    def get_theory():
        """Return comprehensive theoretical explanation of Gradient Boosting."""
        return {
            'name': 'Gradient Boosting',
            'type': 'Supervised Learning - Ensemble Method',
            
            # 1. What the algorithm is
            'definition': """
            🔹 **What is Gradient Boosting?**
            Gradient Boosting is like having a team of students where each student learns from 
            the mistakes of the previous ones. It builds models sequentially, with each new 
            model focusing on correcting the errors of the previous ensemble.
            """,
            
            # 2. Why the algorithm is used
            'motivation': """
            🔹 **Why Use Gradient Boosting?**
            • 📈 **High Accuracy**: Often wins machine learning competitions
            • 🎯 **Web Ranking**: Powers Google search and recommendation systems
            • 💰 **Financial Modeling**: Risk assessment and fraud detection
            • 🔍 **Feature Importance**: Automatically identifies important variables
            • 📊 **Marketing Analytics**: Customer behavior prediction
            • 🏥 **Healthcare**: Disease diagnosis and treatment optimization
            """,
            
            # 3. Intuition with real-life analogy
            'intuition': """
            🔹 **Real-Life Analogy: The Iterative Student Team**
            
            Imagine a math competition where your team tries to solve a complex problem:
            
            🎯 **The Problem**: Predict house prices with 90% accuracy
            👥 **Your Strategy**: Use a series of students, each learning from previous mistakes
            
            **Gradient Boosting is like this smart team approach:**
            
            **Round 1**: 🎆 Student 1 (Simple Model)
            - Makes basic prediction: "Average house price is $200K"
            - Achieves 60% accuracy
            - **Mistakes**: Underestimates luxury homes, overestimates small homes
            
            **Round 2**: 🎆 Student 2 (Learning from Student 1)
            - Focuses ONLY on Student 1's mistakes
            - "Luxury homes need +$100K, small homes need -$50K"
            - Combined prediction: Student 1 + 0.1 × Student 2
            - New accuracy: 75%
            
            **Round 3**: 🎆 Student 3 (Learning from Combined Team)
            - Studies remaining errors from Students 1+2
            - "Waterfront properties need +$75K adjustment"
            - Combined: Student 1 + 0.1×Student 2 + 0.1×Student 3
            - New accuracy: 85%
            
            **Continue for 100 rounds...** 🔄
            - Each student specializes in fixing remaining errors
            - Learning rate (0.1) prevents overconfidence
            - Final ensemble reaches 90% accuracy!
            
            **Key Insights:**
            • **Sequential Learning**: Each model builds on previous knowledge
            • **Error Focus**: New models target remaining mistakes
            • **Controlled Learning**: Learning rate prevents overfitting
            • **Ensemble Power**: Team is stronger than individuals
            
            🎯 **In data terms**: 
            - Students = Weak Learners (usually decision trees)
            - Mistakes = Residuals/Gradients
            - Learning Rate = How much to trust each new student
            - Final Answer = Weighted sum of all predictions
            """,
            
            # 4. Mathematical foundation
            'math_foundation': """
            🔹 **Mathematical Foundation (Step-by-Step)**
            
            **Core Algorithm (Functional Gradient Descent):**
            
            **1. Initialize with constant prediction:**
            ```
            F₀(x) = argminᵣ Σᵢ₌₁ⁿ L(yᵢ, γ)
            ```
            For regression: F₀(x) = mean(y)
            For classification: F₀(x) = log(odds)
            
            **2. For m = 1 to M iterations:**
            
            **Step A: Compute pseudo-residuals (gradients)**
            ```
            rᵢₘ = -[∂L(yᵢ, F(xᵢ))/∂F(xᵢ)]_{F=F_{m-1}}  for i = 1,...,n
            ```
            
            **Step B: Fit weak learner to pseudo-residuals**
            ```
            hₘ(x) = argmin_h Σᵢ₌₁ⁿ (rᵢₘ - h(xᵢ))²
            ```
            
            **Step C: Find optimal step size**
            ```
            ρₘ = argminᵰc Σᵢ₌₁ⁿ L(yᵢ, F_{m-1}(xᵢ) + ρ hₘ(xᵢ))
            ```
            
            **Step D: Update ensemble**
            ```
            Fₘ(x) = F_{m-1}(x) + η × ρₘ × hₘ(x)
            ```
            Where η is the learning rate
            
            **3. Final prediction:**
            ```
            F(x) = F₀(x) + η Σₘ₌₁ᴹ ρₘ hₘ(x)
            ```
            
            **Loss Functions:**
            
            **Regression (Squared Error):**
            ```
            L(y, F(x)) = (y - F(x))²
            Gradient: -∂L/∂F = 2(y - F(x)) = residual
            ```
            
            **Classification (Deviance):**
            ```
            L(y, F(x)) = log(1 + exp(-2yF(x)))  # y ∈ {-1, +1}
            Gradient: -∂L/∂F = 2y / (1 + exp(2yF(x)))
            ```
            
            **Regularization:**
            ```
            Fₘ(x) = F_{m-1}(x) + η × hₘ(x)  # η < 1 for shrinkage
            ```
            
            **Tree-specific Updates (GBDT):**
            ```
            For leaf j in tree hₘ:
            γⱼₘ = argminᵣ Σ_{xᵢ ∈ Rⱼₘ} L(yᵢ, F_{m-1}(xᵢ) + γ)
            ```
            """,
            
            # 5. Step-by-step working
            'algorithm_steps': """
            🔹 **How Gradient Boosting Works (Step-by-Step)**
            
            **Step 1: Initialize Prediction** 🎆
            • Start with simple baseline prediction
            • Regression: F₀(x) = mean(y_train)
            • Classification: F₀(x) = log(p/(1-p)) where p = class proportion
            • This becomes your initial "team score"
            
            **Step 2: Calculate Initial Residuals** 📊
            • For each training sample, compute error:
            • Residuals = Actual - Current_Prediction
            • These residuals show where current model fails
            
            **Step 3: Train Weak Learner on Residuals** 🌳
            • Train decision tree to predict residuals (not original target!)
            • Tree learns patterns in current mistakes
            • Usually shallow trees (depth 3-6) to avoid overfitting
            
            **Step 4: Add New Predictions with Learning Rate** ➕
            • New_Prediction = Old_Prediction + learning_rate × Tree_Prediction
            • Learning rate (0.01-0.3) controls how much to trust new tree
            • Prevents any single tree from dominating
            
            **Step 5: Update Ensemble** 🔄
            • Current model now includes all trees built so far
            • F(x) = F₀(x) + η×h₁(x) + η×h₂(x) + ... + η×hₘ(x)
            • Each tree contributes small improvement
            
            **Step 6: Repeat Until Convergence** 🔁
            • Calculate new residuals from updated ensemble
            • Train next tree on these new residuals
            • Continue for n_estimators iterations (typically 100-1000)
            
            **Step 7: Make Final Predictions** 🎯
            • For new sample: pass through all trees in sequence
            • Sum all weighted predictions
            • Apply inverse link function for classification
            
            **Step 8: Monitor for Overfitting** 🚨
            • Track validation error during training
            • Stop early if validation error starts increasing
            • Use regularization techniques (shrinkage, subsampling)
            
            **Key Principles:**
            • **Additive Model**: Each tree adds to the ensemble
            • **Gradient Descent**: Minimize loss function step by step
            • **Weak Learners**: Each tree is simple but improves overall model
            • **Sequential Learning**: Order matters - can't parallelize easily
            """,
            
            # 6. Pseudocode
            'pseudocode': """
            🔹 **Pseudocode (Easy to Understand)**
            
            ```
            ALGORITHM: Gradient Boosting
            
            INPUT: 
                - X: feature matrix (n_samples × n_features)
                - y: target values (n_samples × 1)
                - n_estimators: number of boosting rounds
                - learning_rate: shrinkage parameter (η)
                - max_depth: maximum depth of each tree
                - loss_function: loss function to optimize
            
            OUTPUT:
                - ensemble: collection of weak learners with weights
            
            BEGIN
                1. INITIALIZE base prediction:
                   IF regression:
                       F₀ = mean(y)
                   ELIF classification:
                       F₀ = log(count(class=1) / count(class=0))
                   
                   ensemble = [F₀]
                
                2. FOR m = 1 to n_estimators:
                   
                   a. CALCULATE pseudo-residuals (gradients):
                      residuals = []
                      FOR each sample i:
                          current_pred = predict_current_ensemble(X[i])
                          
                          IF regression:
                              gradient = y[i] - current_pred
                          ELIF classification:
                              prob = 1 / (1 + exp(-current_pred))
                              gradient = y[i] - prob
                          
                          residuals.append(gradient)
                   
                   b. TRAIN weak learner on residuals:
                      tree_m = DecisionTree(max_depth=max_depth)
                      tree_m.fit(X, residuals)
                   
                   c. CALCULATE optimal step size (optional):
                      # For simple implementation, use fixed learning_rate
                      step_size = learning_rate
                   
                   d. UPDATE ensemble:
                      ensemble.append((step_size, tree_m))
                
                3. RETURN ensemble
            END
            
            PREDICTION:
            BEGIN
                1. FUNCTION predict(X_new):
                   predictions = []
                   
                   FOR each sample x in X_new:
                       # Start with base prediction
                       pred = F₀
                       
                       # Add contribution from each tree
                       FOR each (weight, tree) in ensemble[1:]:
                           tree_pred = tree.predict(x)
                           pred += weight * tree_pred
                       
                       # Apply inverse link function if needed
                       IF classification:
                           prob = 1 / (1 + exp(-pred))
                           final_pred = 1 if prob > 0.5 else 0
                       ELSE:
                           final_pred = pred
                       
                       predictions.append(final_pred)
                   
                   RETURN predictions
            END
            
            HELPER FUNCTIONS:
            BEGIN
                FUNCTION predict_current_ensemble(x):
                    pred = F₀
                    FOR each (weight, tree) in ensemble[1:]:
                        pred += weight * tree.predict(x)
                    RETURN pred
                
                FUNCTION calculate_loss(y_true, y_pred):
                    IF regression:
                        RETURN mean((y_true - y_pred)²)
                    ELIF classification:
                        RETURN -mean(y_true * log(sigmoid(y_pred)) + 
                                   (1-y_true) * log(1-sigmoid(y_pred)))
            END
            ```
            """,
            
            # 7. Python implementation
            'python_implementation': """
            🔹 **Python Implementation**
            
            **From Scratch (Simplified Gradient Boosting):**
            ```python
            import numpy as np
            from sklearn.tree import DecisionTreeRegressor
            from sklearn.base import BaseEstimator, RegressorMixin
            
            class GradientBoostingFromScratch(BaseEstimator, RegressorMixin):
                def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, 
                           random_state=None):
                    self.n_estimators = n_estimators
                    self.learning_rate = learning_rate
                    self.max_depth = max_depth
                    self.random_state = random_state
                    self.estimators_ = []
                    self.train_score_ = []
                    self.feature_importances_ = None
                
                def fit(self, X, y):
                    \"\"\"Train gradient boosting model.\"\"\"
                    # Initialize with mean prediction
                    self.init_prediction_ = np.mean(y)
                    
                    # Current predictions start with mean
                    current_predictions = np.full(len(y), self.init_prediction_)
                    
                    self.estimators_ = []
                    self.train_score_ = []
                    
                    for i in range(self.n_estimators):
                        # Calculate residuals (gradients for squared loss)
                        residuals = y - current_predictions
                        
                        # Train tree to predict residuals
                        tree = DecisionTreeRegressor(
                            max_depth=self.max_depth,
                            random_state=self.random_state
                        )
                        tree.fit(X, residuals)
                        
                        # Get tree predictions
                        tree_predictions = tree.predict(X)
                        
                        # Update current predictions with learning rate
                        current_predictions += self.learning_rate * tree_predictions
                        
                        # Store tree and calculate training score
                        self.estimators_.append(tree)
                        mse = np.mean((y - current_predictions) ** 2)
                        self.train_score_.append(mse)
                        
                        # Early stopping if error increases (optional)
                        if i > 10 and self.train_score_[i] > self.train_score_[i-5]:
                            print(f"Early stopping at iteration {i}")
                            break
                    
                    # Calculate feature importances
                    self._calculate_feature_importances(X)
                    
                    return self
                
                def predict(self, X):
                    \"\"\"Make predictions using trained model.\"\"\"
                    # Start with initial prediction
                    predictions = np.full(X.shape[0], self.init_prediction_)
                    
                    # Add predictions from each tree
                    for tree in self.estimators_:
                        tree_pred = tree.predict(X)
                        predictions += self.learning_rate * tree_pred
                    
                    return predictions
                
                def _calculate_feature_importances(self, X):
                    \"\"\"Calculate feature importances by averaging across trees.\"\"\"
                    if len(self.estimators_) == 0:
                        return
                    
                    importances = np.zeros(X.shape[1])
                    for tree in self.estimators_:
                        if hasattr(tree, 'feature_importances_'):
                            importances += tree.feature_importances_
                    
                    # Average and normalize
                    importances /= len(self.estimators_)
                    self.feature_importances_ = importances / np.sum(importances)
                
                def staged_predict(self, X):
                    \"\"\"Return predictions at each stage for plotting.\"\"\"
                    predictions = np.full(X.shape[0], self.init_prediction_)
                    
                    # Yield prediction at each stage
                    yield predictions.copy()
                    
                    for tree in self.estimators_:
                        tree_pred = tree.predict(X)
                        predictions += self.learning_rate * tree_pred
                        yield predictions.copy()
            
            # Example usage
            from sklearn.datasets import make_regression
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error, r2_score
            
            # Generate data
            X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            gb = GradientBoostingFromScratch(n_estimators=100, learning_rate=0.1, max_depth=3)
            gb.fit(X_train, y_train)
            
            # Make predictions
            y_pred = gb.predict(X_test)
            
            print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
            print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
            
            # Plot feature importances
            import matplotlib.pyplot as plt
            plt.bar(range(len(gb.feature_importances_)), gb.feature_importances_)
            plt.title('Feature Importances')
            plt.show()
            ```
            
            **Using Scikit-learn:**
            ```python
            from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
            from sklearn.model_selection import GridSearchCV, validation_curve
            from sklearn.metrics import accuracy_score, mean_squared_error
            
            # Classification
            gb_clf = GradientBoostingClassifier(
                n_estimators=100,        # Number of boosting stages
                learning_rate=0.1,       # Shrinkage parameter
                max_depth=3,            # Maximum depth of trees
                subsample=0.8,          # Fraction of samples for each tree
                max_features='sqrt',    # Number of features for each split
                random_state=42,
                validation_fraction=0.1, # For early stopping
                n_iter_no_change=10     # Early stopping rounds
            )
            
            gb_clf.fit(X_train, y_train)
            y_pred = gb_clf.predict(X_test)
            y_proba = gb_clf.predict_proba(X_test)
            
            # Feature importance
            importance = gb_clf.feature_importances_
            
            # Training progress
            train_scores = gb_clf.train_score_
            validation_scores = gb_clf.validation_scores_
            
            # Regression
            gb_reg = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                loss='squared_error',   # or 'absolute_error', 'huber', 'quantile'
                alpha=0.9,             # For quantile/huber loss
                subsample=1.0,         # Bootstrap sampling fraction
                criterion='friedman_mse',
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42
            )
            
            gb_reg.fit(X_train, y_train)
            y_pred_reg = gb_reg.predict(X_test)
            
            # Hyperparameter tuning
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 4, 5],
                'subsample': [0.8, 0.9, 1.0]
            }
            
            grid_search = GridSearchCV(
                GradientBoostingClassifier(random_state=42),
                param_grid,
                cv=5,
                scoring='accuracy',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            best_gb = grid_search.best_estimator_
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
            
            # Staged predictions for plotting learning curves
            staged_preds = list(gb_reg.staged_predict(X_test))
            staged_scores = [mean_squared_error(y_test, pred) for pred in staged_preds]
            
            plt.plot(staged_scores)
            plt.title('Validation Error vs Number of Estimators')
            plt.xlabel('Number of Estimators')
            plt.ylabel('Mean Squared Error')
            plt.show()
            ```
            
            **Advanced Features:**
            ```python
            # Different loss functions for regression
            gb_huber = GradientBoostingRegressor(loss='huber', alpha=0.9)
            gb_quantile = GradientBoostingRegressor(loss='quantile', alpha=0.9)
            gb_absolute = GradientBoostingRegressor(loss='absolute_error')
            
            # Early stopping with validation
            gb_early = GradientBoostingClassifier(
                validation_fraction=0.2,
                n_iter_no_change=5,
                tol=1e-4
            )
            
            # Monitoring training progress
            test_scores = []
            for i, pred in enumerate(gb_clf.staged_predict(X_test)):
                test_scores.append(accuracy_score(y_test, pred))
            
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(gb_clf.train_score_, label='Training Score')
            plt.plot(test_scores, label='Test Score')
            plt.legend()
            plt.title('Learning Curves')
            
            plt.subplot(1, 2, 2)
            plt.bar(range(len(gb_clf.feature_importances_)), gb_clf.feature_importances_)
            plt.title('Feature Importances')
            plt.show()
            ```
            """,
            
            # 8. Example with sample input/output
            'example': """
            🔹 **Complete Example: Customer Purchase Prediction**
            
            **Input Data (Customer Features):**
            ```
            Customer | Age | Income | Pages_Viewed | Time_Spent | Previous_Purchases | Will_Purchase?
            A        | 25  | 50000  | 5           | 10         | 0                 | No
            B        | 35  | 75000  | 15          | 25         | 2                 | Yes
            C        | 45  | 90000  | 8           | 15         | 1                 | Yes
            D        | 30  | 60000  | 20          | 30         | 3                 | Yes
            E        | 22  | 40000  | 3           | 5          | 0                 | No
            F        | 50  | 100000 | 12          | 20         | 4                 | Yes
            ```
            
            **Step-by-Step Gradient Boosting Training:**
            ```
            Initial Setup:
            - Convert target: No=0, Yes=1
            - Base prediction: F₀ = log(4/2) = log(2) = 0.693 (log odds)
            
            Iteration 1:
            ─────────────
            1. Current probabilities from F₀=0.693:
               P = 1/(1+e^(-0.693)) = 0.667 for all customers
            
            2. Calculate gradients (residuals):
               Customer A: gradient = 0 - 0.667 = -0.667
               Customer B: gradient = 1 - 0.667 = +0.333
               Customer C: gradient = 1 - 0.667 = +0.333
               Customer D: gradient = 1 - 0.667 = +0.333
               Customer E: gradient = 0 - 0.667 = -0.667
               Customer F: gradient = 1 - 0.667 = +0.333
            
            3. Train Tree 1 to predict gradients:
               Tree finds: "If Pages_Viewed > 7, predict +0.3, else predict -0.6"
               
            4. Update predictions (learning_rate = 0.1):
               F₁ = F₀ + 0.1 × Tree1_predictions
               
            Iteration 2:
            ─────────────
            1. New probabilities from F₁:
               Customers A,E: lower probability (Pages_Viewed ≤ 7)
               Customers B,C,D,F: higher probability (Pages_Viewed > 7)
            
            2. New gradients focus on remaining errors
            
            3. Train Tree 2 on new gradients:
               Tree finds: "If Income > 65000 AND Time_Spent > 15, predict +0.4"
               
            Continue for 100 iterations...
            ```
            
            **Final Model Structure:**
            ```
            F(x) = 0.693 + 0.1×Tree1(x) + 0.1×Tree2(x) + ... + 0.1×Tree100(x)
            
            Where each tree contributes specialized knowledge:
            - Tree 1: Basic engagement (pages viewed)
            - Tree 2: High-value customers (income + time)
            - Tree 3: Loyal customers (previous purchases)
            - Tree 4: Age-specific patterns
            - ...
            - Tree 100: Fine-grained edge cases
            ```
            
            **New Customer Prediction:**
            ```
            New Customer: Age=28, Income=65000, Pages=12, Time=18, Previous=1
            
            Pass through each tree:
            F₀: 0.693 (base prediction)
            Tree1: +0.25 (Pages > 7)
            Tree2: +0.15 (Income > 65k AND Time > 15)
            Tree3: +0.10 (Previous purchases = 1)
            ...
            Tree100: +0.02 (minor adjustment)
            
            Final score: F(x) = 0.693 + 0.1×(0.25 + 0.15 + 0.10 + ... + 0.02) = 1.85
            
            Probability: P = 1/(1+e^(-1.85)) = 0.864 (86.4%)
            
            Prediction: WILL PURCHASE! 🛒 (86.4% confidence)
            ```
            
            **Feature Importance Results:**
            ```
            Pages_Viewed: 35% (most important)
            Time_Spent: 25%
            Previous_Purchases: 20%
            Income: 15%
            Age: 5%
            ```
            """,
            
            # 9. Visualization explanation
            'visualization': """
            🔹 **Understanding Through Visualizations**
            
            **Learning Curves:**
            📈 Training and validation error vs number of estimators
            • Training error continuously decreases (overfitting risk)
            • Validation error initially decreases then may increase
            • Optimal stopping point where validation error minimizes
            • Shows bias-variance trade-off over iterations
            
            **Feature Importance Plot:**
            📋 Bar chart showing relative feature contributions
            • Calculated by averaging importance across all trees
            • Height shows cumulative contribution to model
            • Helps identify most predictive features
            • Useful for feature selection and interpretation
            
            **Partial Dependence Plots:**
            📉 Shows effect of individual features on predictions
            • X-axis: Feature values
            • Y-axis: Average prediction when feature has that value
            • Reveals non-linear relationships captured by ensemble
            • Marginal effect plots for feature understanding
            
            **Residual Analysis:**
            🎯 Plots of prediction errors vs iterations
            • Shows how different types of errors are corrected over time
            • Early trees fix major patterns
            • Later trees fix subtle patterns and noise
            • Helps understand model behavior
            
            **Tree Structure Evolution:**
            🔄 Side-by-side trees from different iterations
            • Early trees: focus on major splits
            • Middle trees: focus on medium-importance patterns
            • Late trees: focus on edge cases and fine-tuning
            • Shows progressive refinement
            
            **Prediction Decomposition:**
            📊 Waterfall chart showing contribution of each tree
            • Base prediction + Tree1 + Tree2 + ... = Final prediction
            • Shows how ensemble builds up prediction
            • Useful for explaining individual predictions
            """,
            
            # 10. Time and space complexity
            'complexity': """
            🔹 **Time & Space Complexity**
            
            **Time Complexity:**
            • **Training**: O(M × T × n × log(n) × d) where:
              - M = number of estimators
              - T = tree construction time
              - n = number of samples
              - d = number of features
            • **Single Tree**: O(n × log(n) × d) for building
            • **Prediction**: O(M × depth) per sample
            • **Sequential Nature**: Cannot parallelize across estimators
            
            **Space Complexity:**
            • **Model Storage**: O(M × nodes) where nodes ≈ 2^depth
            • **Training Memory**: O(n) for gradients/residuals
            • **Tree Storage**: Each tree stores split conditions and values
            • **Prediction Memory**: O(1) per prediction
            
            **Computational Characteristics:**
            • ❌ **Sequential Training**: Must train trees one by one
            • ✅ **Parallelizable Tree Building**: Each tree can use multiple cores
            • ❌ **Memory Intensive**: Stores many trees
            • ✅ **Fast Prediction**: Linear in number of estimators
            • ⚠️ **Early Stopping**: Can reduce actual complexity
            
            **Scalability Issues:**
            • Training time grows linearly with number of estimators
            • Model size grows with ensemble size
            • Prediction time acceptable for most applications
            • Memory usage can be significant for large ensembles
            
            **Optimization Strategies:**
            • **Early stopping**: Prevent overfitting and reduce training time
            • **Subsampling**: Use fraction of data for each tree
            • **Feature subsampling**: Use random subset of features
            • **Learning rate scheduling**: Reduce rate over time
            • **Tree pruning**: Limit tree complexity
            """,
            
            # 11. Advantages and disadvantages
            'pros_cons': """
            🔹 **Advantages** ✅
            • **High Accuracy**: Often wins ML competitions and achieves SOTA results
            • **Feature Interactions**: Automatically captures complex feature interactions
            • **Feature Importance**: Provides natural feature ranking
            • **Mixed Data Types**: Handles numerical and categorical features well
            • **Missing Values**: Can handle missing data naturally
            • **No Feature Scaling**: Tree-based, doesn't require normalization
            • **Flexible Loss Functions**: Can optimize any differentiable loss
            • **Robust to Outliers**: Tree splits are robust to extreme values
            • **Progressive Learning**: Each iteration improves the model
            • **Interpretable**: Can trace prediction through trees
            
            🔹 **Disadvantages** ❌
            • **Overfitting Prone**: Can memorize training data if not regularized
            • **Computationally Expensive**: Slow training, especially with many estimators
            • **Sequential Nature**: Cannot parallelize across boosting rounds
            • **Hyperparameter Sensitive**: Requires careful tuning for optimal performance
            • **Memory Intensive**: Stores many trees, can use significant RAM
            • **Sensitive to Noise**: Later trees may fit noise in residuals
            • **Black Box Nature**: Complex ensemble hard to interpret globally
            • **Early Stopping Required**: Need validation set for optimal stopping
            • **Extrapolation Poor**: Cannot predict beyond training data range
            • **Class Imbalance Issues**: May be biased toward majority class
            """,
            
            # 12. When to use and when NOT to use
            'usage_guide': """
            🔹 **When TO Use Gradient Boosting** ✅
            
            **Perfect for:**
            • 🏆 **Kaggle Competitions**: When accuracy is paramount
            • 📈 **Structured/Tabular Data**: Works excellently with traditional datasets
            • 💰 **Financial Modeling**: Risk assessment, fraud detection
            • 🎯 **Ranking Problems**: Search ranking, recommendation scoring
            • 📊 **Marketing Analytics**: Customer behavior prediction
            • 🔍 **Feature-Rich Problems**: When you have many potentially relevant features
            
            **Good when:**
            • Have sufficient training data (>1000 samples)
            • Accuracy is more important than interpretability
            • Features have complex interactions
            • Can afford longer training time
            • Need automatic feature selection
            • Have mixed data types (numerical + categorical)
            
            🔹 **When NOT to Use Gradient Boosting** ❌
            
            **Avoid when:**
            • ⚡ **Real-time Training**: Need to retrain model frequently
            • 📏 **Simple Linear Relationships**: Linear models would work fine
            • 📈 **Small Datasets**: < 500 samples (prone to overfitting)
            • 💾 **Memory Constraints**: Limited RAM for model storage
            • 🔍 **High Interpretability**: Need to explain every decision
            • 🖼️ **Image/Text Data**: Deep learning usually better
            • 🐌 **Very Noisy Data**: May overfit to noise
            
            **Use instead:**
            • **Linear Relationships**: Linear/Logistic Regression
            • **Speed Critical**: Random Forest (parallelizable)
            • **Interpretability**: Single Decision Tree, Linear models
            • **Small Data**: Naive Bayes, KNN, simple models
            • **Image/NLP**: Convolutional/Recurrent Neural Networks
            • **Real-time**: Pre-computed lookup tables, simple heuristics
            """,
            
            # 13. Common interview questions
            'interview_questions': """
            🔹 **Common Interview Questions & Answers**
            
            **Q1: How does Gradient Boosting differ from Random Forest?**
            A: Key differences:
            • **GB**: Sequential, each tree corrects previous errors, higher accuracy potential
            • **RF**: Parallel, independent trees, more robust, faster training
            • **GB**: Prone to overfitting, needs more tuning
            • **RF**: Less overfitting, easier to tune
            • **Use GB**: When accuracy is critical and you have time to tune
            • **Use RF**: When you need robust baseline quickly
            
            **Q2: What is the role of learning rate in Gradient Boosting?**
            A: Learning rate controls the contribution of each tree:
            • **High learning rate (0.3)**: Fast learning, but may overfit
            • **Low learning rate (0.01)**: Slow learning, better generalization, needs more trees
            • **Trade-off**: Lower rate + more estimators often gives better results
            • **Best practice**: Start with 0.1, then try 0.05-0.2 range
            
            **Q3: How do you prevent overfitting in Gradient Boosting?**
            A: Multiple strategies:
            • **Early stopping**: Monitor validation error, stop when it increases
            • **Learning rate**: Use lower rates (0.01-0.1)
            • **Subsampling**: Use 50-80% of data for each tree
            • **Tree constraints**: Limit max_depth (3-6), min_samples_leaf
            • **Regularization**: L1/L2 penalties on leaf values
            • **Cross-validation**: For hyperparameter tuning
            
            **Q4: What are pseudo-residuals and why are they important?**
            A: Pseudo-residuals are negative gradients of the loss function:
            • For squared loss: residuals = actual - predicted (regular residuals)
            • For other losses: gradients point toward optimal direction
            • **Importance**: Allow GB to optimize any differentiable loss function
            • **Example**: For classification, gradients help adjust class probabilities
            
            **Q5: How do you choose the number of estimators?**
            A: Best practices:
            • **Start**: 100-200 estimators as baseline
            • **Monitor**: Use validation curves to see where error plateaus
            • **Early stopping**: Let algorithm decide automatically
            • **More trees**: Needed with lower learning rates
            • **Computational budget**: Balance accuracy vs training time
            
            **Q6: Can Gradient Boosting handle missing values?**
            A: Yes, tree-based GB handles missing values naturally:
            • **Tree splits**: Can learn optimal direction for missing values
            • **XGBoost**: Has sophisticated missing value handling
            • **Default direction**: Learns which way to send missing values
            • **No preprocessing**: Often no need to impute missing values
            """,
            
            # 14. Common mistakes
            'common_mistakes': """
            🔹 **Common Beginner Mistakes & How to Avoid**
            
            **Mistake 1: Using too high learning rate** 🚫
            ❌ Setting learning_rate=1.0 or 0.5 (too aggressive)
            ✅ **Fix**: Start with 0.1, try 0.01-0.3 range, lower is usually better
            
            **Mistake 2: Not using early stopping** 🚫
            ❌ Training for fixed number of iterations without validation monitoring
            ✅ **Fix**: Use validation_fraction and n_iter_no_change for automatic stopping
            
            **Mistake 3: Ignoring overfitting signs** 🚫
            ❌ Training accuracy 99% but test accuracy 70%
            ✅ **Fix**: Monitor validation score, use regularization techniques
            
            **Mistake 4: Too many estimators with high learning rate** 🚫
            ❌ n_estimators=1000 with learning_rate=0.3
            ✅ **Fix**: Use fewer estimators (100-300) with moderate learning rate (0.05-0.15)
            
            **Mistake 5: Not tuning tree depth** 🚫
            ❌ Using default max_depth without considering data complexity
            ✅ **Fix**: Try max_depth 3-6 for most problems, deeper for complex patterns
            
            **Mistake 6: Forgetting about class imbalance** 🚫
            ❌ Using default settings with 90%-10% class split
            ✅ **Fix**: Use class_weight='balanced' or stratified sampling
            
            **Mistake 7: No hyperparameter tuning** 🚫
            ❌ Using default parameters without experimentation
            ✅ **Fix**: Use GridSearchCV or RandomizedSearchCV for optimization
            
            **Mistake 8: Training on entire dataset** 🚫
            ❌ No validation split for monitoring overfitting
            ✅ **Fix**: Always reserve validation set or use cross-validation
            """,
            
            # 15. Comparison with similar algorithms
            'comparisons': """
            🔹 **Gradient Boosting vs Similar Algorithms**
            
            **Gradient Boosting vs XGBoost/LightGBM:**
            • **Gradient Boosting**: Original algorithm, simpler, slower
            • **XGBoost/LightGBM**: Optimized implementations, faster, more features
            • **Use XGB/LGB**: For production and when performance matters
            
            **Gradient Boosting vs AdaBoost:**
            • **Gradient Boosting**: Optimizes any loss function, more flexible
            • **AdaBoost**: Only exponential loss, simpler, specific to classification
            • **Use Gradient Boosting**: More general purpose and robust
            
            **Gradient Boosting vs Random Forest:**
            • **Gradient Boosting**: Sequential, higher accuracy potential, more overfitting risk
            • **Random Forest**: Parallel, faster training, more robust
            • **Use GB**: When accuracy is critical and you can tune properly
            • **Use RF**: When you need quick, robust baseline
            
            **Gradient Boosting vs Neural Networks:**
            • **Gradient Boosting**: Structured data, automatic feature engineering
            • **Neural Networks**: Unstructured data, manual feature engineering
            • **Use GB**: For tabular data with clear features
            • **Use NN**: For images, text, audio, video
            
            **Gradient Boosting vs Stacking:**
            • **Gradient Boosting**: Homogeneous weak learners (usually trees)
            • **Stacking**: Heterogeneous models (different algorithms)
            • **Use Stacking**: When you want to combine very different approaches
            """,
            
            # 16. Real-world applications
            'real_world_applications': """
            🔹 **Real-World Applications & Industry Use Cases**
            
            **🔍 Web Search & Information Retrieval:**
            • Google's RankBrain uses gradient boosting for search ranking
            • Yahoo's web search ranking algorithms
            • Document relevance scoring in search engines
            • Ad placement and click-through rate prediction
            • Content recommendation systems
            
            **💰 Finance & Banking:**
            • Credit scoring and loan default prediction
            • Algorithmic trading strategies and market prediction
            • Fraud detection in transactions and insurance claims
            • Risk assessment and portfolio optimization
            • High-frequency trading signal generation
            
            **🛒 E-commerce & Marketing:**
            • Customer lifetime value prediction
            • Conversion rate optimization
            • Price optimization and dynamic pricing
            • Demand forecasting and inventory management
            • Personalized product recommendations
            
            **🏥 Healthcare & Life Sciences:**
            • Disease diagnosis from medical imaging
            • Drug discovery and molecular property prediction
            • Clinical trial optimization and patient stratification
            • Electronic health record analysis
            • Epidemic modeling and public health planning
            
            **🏭 Manufacturing & Operations:**
            • Predictive maintenance and equipment failure prediction
            • Quality control and defect detection
            • Supply chain optimization
            • Energy consumption forecasting
            • Process optimization in chemical plants
            
            **📱 Technology & Software:**
            • Click-through rate prediction for online advertising
            • User engagement and churn prediction
            • A/B testing result analysis
            • Software bug prediction and code quality assessment
            • Network performance optimization
            
            **🌱 Environmental & Scientific Research:**
            • Climate change modeling and weather prediction
            • Species distribution modeling in ecology
            • Air quality prediction and pollution monitoring
            • Agricultural yield prediction
            • Astronomical object classification
            
            **💡 Key Success Factors:**
            • Proper hyperparameter tuning (learning rate, depth, estimators)
            • Early stopping to prevent overfitting
            • Feature engineering and selection
            • Validation strategy for model selection
            • Understanding of business problem and loss function choice
            • Computational resources for training and deployment
            """
        }
    
    def generate_sample_data(self, task_type, n_samples=500, n_features=6):
        """Generate sample data for demonstration."""
        if task_type == 'classification':
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=n_features,
                n_redundant=0,
                n_classes=3,
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
        """Fit the Gradient Boosting model."""
        if self.task_type == 'classification':
            self.model = GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                random_state=42
            )
        else:
            self.model = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                random_state=42
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
    
    def get_feature_importance(self):
        """Get feature importance from the model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        return self.model.feature_importances_
    
    def streamlit_interface(self):
        """Create comprehensive Streamlit interface for Gradient Boosting."""
        st.subheader("🚀 Gradient Boosting")
        
        theory = self.get_theory()
        
        # Main tabs for comprehensive coverage
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🎯 Overview", "📚 Deep Dive", "💻 Implementation", 
            "🧪 Interactive Demo", "❓ Q&A", "🏢 Applications"
        ])
        
        with tab1:
            # Overview Tab - Essential Information
            st.markdown("### 🎯 What is Gradient Boosting?")
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
            st.markdown("### 🧪 Try Gradient Boosting Yourself!")
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
            n_estimators = st.slider("Number of estimators:", 50, 300, 100, 10)
        with col3:
            learning_rate = st.slider("Learning rate:", 0.01, 0.3, 0.1, 0.01)
        with col4:
            n_samples = st.slider("Samples:", 300, 800, 500)
        
        # Update parameters
        self.task_type = task_type
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        
        # Generate data and train model
        X, y = self.generate_sample_data(task_type, n_samples)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
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
        
        # Feature importance visualization
        st.markdown("### 📈 Feature Importance")
        importances = self.get_feature_importance()
        fig, ax = plt.subplots(figsize=(10, 6))
        feature_names = [f'Feature {i+1}' for i in range(len(importances))]
        bars = ax.bar(feature_names, importances, alpha=0.7, color='orange', edgecolor='black')
        ax.set_title('Feature Importance in Gradient Boosting')
        ax.set_ylabel('Importance')
        
        # Add value labels on bars
        for bar, importance in zip(bars, importances):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{importance:.3f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()
        
        # Interpretation
        st.markdown("### 🔍 Interpretation")
        
        if task_type == 'classification':
            if 'Accuracy' in test_metrics:
                accuracy = test_metrics['Accuracy']
                if accuracy > 0.9:
                    st.success(f"**Excellent performance!** Accuracy: {accuracy:.3f}")
                    st.write("The Gradient Boosting model is performing very well on this dataset.")
                elif accuracy > 0.8:
                    st.info(f"**Good performance.** Accuracy: {accuracy:.3f}")
                    st.write("The model is performing well.")
                elif accuracy > 0.7:
                    st.warning(f"**Moderate performance.** Accuracy: {accuracy:.3f}")
                    st.write("Consider tuning hyperparameters or feature engineering.")
                else:
                    st.error(f"**Poor performance.** Accuracy: {accuracy:.3f}")
                    st.write("The dataset might need more preprocessing or different approach.")
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
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Estimators", f"{n_estimators}")
        col2.metric("Learning Rate", f"{learning_rate}")
        col3.metric("Task", task_type.capitalize())
        col4.metric("Most Important Feature", feature_names[np.argmax(importances)] if len(importances) > 0 else "N/A")
        
        # Recommendations
        st.markdown("**Recommendations:**")
        if learning_rate > 0.2:
            st.write("• High learning rate might lead to overfitting. Consider reducing it.")
        if n_estimators < 100:
            st.write("• Low number of estimators might underfit. Consider increasing.")
        if task_type == 'classification' and 'Accuracy' in test_metrics:
            train_acc = train_metrics.get('Accuracy', 0)
            test_acc = test_metrics.get('Accuracy', 0)
            if train_acc - test_acc > 0.1:
                st.write("• Large gap between training and test accuracy suggests overfitting.")


def main():
    """Main function for testing Gradient Boosting."""
    gb = GradientBoosting()
    gb.streamlit_interface()


if __name__ == "__main__":
    main()