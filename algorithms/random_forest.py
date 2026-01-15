"""
Random Forest Algorithm Implementation

Random Forest is an ensemble learning method that combines multiple decision trees
to create a more robust and accurate model for classification and regression tasks.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report,
                           mean_squared_error, r2_score, mean_absolute_error)
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_regression
import streamlit as st
import seaborn as sns


class RandomForest:
    """
    Random Forest implementation with educational explanations.
    
    Random Forest builds multiple decision trees and combines their predictions
    through voting (classification) or averaging (regression). It introduces
    randomness in two ways:
    1. Bootstrap sampling of training data for each tree
    2. Random subset of features considered at each split
    """
    
    def __init__(self, n_estimators=100, task_type='classification', max_depth=None):
        self.n_estimators = n_estimators
        self.task_type = task_type
        self.max_depth = max_depth
        self.model = None
        self.is_fitted = False
        
    @staticmethod
    def get_theory():
        """Return comprehensive theoretical explanation of Random Forest."""
        return {
            'name': 'Random Forest',
            'type': 'Supervised Learning - Ensemble Method',
            
            # 1. What the algorithm is
            'definition': """
            🔹 **What is Random Forest?**
            Random Forest is like asking 100 experts for their opinion and taking the majority vote.
            Instead of relying on one decision tree (one expert), it creates many trees and combines 
            their predictions to make better, more reliable decisions.
            """,
            
            # 2. Why the algorithm is used
            'motivation': """
            🔹 **Why Use Random Forest?**
            • 🎯 **Better Accuracy**: Outperforms single decision trees
            • 🛡️ **Reduces Overfitting**: Multiple trees prevent memorizing training data
            • 📊 **Feature Ranking**: Automatically identifies important features
            • 🔍 **Handles Missing Data**: Works even with incomplete datasets
            • ⚡ **Fast Training**: Can train trees in parallel
            • 🎲 **Robust Predictions**: Less sensitive to outliers and noise
            """,
            
            # 3. Intuition with real-life analogy
            'intuition': """
            🔹 **Real-Life Analogy: The Expert Committee Decision**
            
            Imagine you're buying a house and need to estimate its value:
            
            🏠 **The Problem**: What's the fair price for this house?
            🤔 **Single Expert**: One appraiser gives estimate (Decision Tree)
            👥 **Expert Committee**: 100 appraisers give estimates (Random Forest)
            
            **Random Forest is like having a wise committee that:**
            
            **Step 1**: 🎲 Each expert sees different house examples (Bootstrap sampling)
            **Step 2**: 🔍 Each expert focuses on different features (Random feature selection)
            - Expert 1: Size, location, age
            - Expert 2: Bedrooms, garage, schools
            - Expert 3: Kitchen, bathrooms, yard
            
            **Step 3**: 🗳️ Each expert makes independent estimate
            **Step 4**: 📊 Committee averages all estimates for final price
            
            **Why it works better:**
            • One expert might focus too much on one feature (overfitting)
            • Committee reduces individual biases and errors
            • Diverse perspectives lead to better overall decision
            
            🎯 **In data terms**: 
            - Experts = Individual Trees
            - House features = Data Features
            - Final estimate = Ensemble Prediction
            - Committee wisdom = Reduced overfitting
            """,
            
            # 4. Mathematical foundation
            'math_foundation': """
            🔹 **Mathematical Foundation (Step-by-Step)**
            
            **Ensemble Prediction Formula:**
            ```
            Classification: ŷ = mode{T₁(x), T₂(x), ..., Tₙ(x)}
            Regression: ŷ = (1/n) × Σᵢ₌₁ⁿ Tᵢ(x)
            ```
            
            **Where:**
            • `Tᵢ(x)` = Prediction from i-th tree
            • `n` = Number of trees (n_estimators)
            • `mode{}` = Most frequent class (majority vote)
            • `Σ` = Average of all tree predictions
            
            **Bootstrap Sampling:**
            ```
            For each tree i:
                Dᵢ = Bootstrap_sample(D, size=|D|)
                # Sample with replacement from original dataset D
            ```
            
            **Random Feature Selection:**
            ```
            At each node split:
                m = √p (classification) or p/3 (regression)
                # Randomly select m features out of p total features
                best_split = find_best_split(selected_features)
            ```
            
            **Out-of-Bag (OOB) Error:**
            ```
            For each sample x:
                OOB_trees = {trees where x was not in bootstrap sample}
                OOB_prediction = average(OOB_trees(x))
            OOB_error = error(OOB_predictions, true_values)
            ```
            
            **Feature Importance:**
            ```
            For each feature j:
                Importance(j) = Σᵢ₌₁ⁿ (decrease_in_impurity_by_feature_j_in_tree_i)
            Normalize: Importance(j) = Importance(j) / Σⱼ Importance(j)
            ```
            """,
            
            # 5. Step-by-step working
            'algorithm_steps': """
            🔹 **How Random Forest Works (Step-by-Step)**
            
            **Step 1: Bootstrap Sampling** 🎲
            • For each tree (1 to n_estimators):
            • Create bootstrap sample of training data
            • Sample WITH replacement (same size as original)
            • Some samples appear multiple times, others not at all
            
            **Step 2: Random Feature Selection** 🎯
            • At each node split in each tree:
            • Randomly select √p features (classification) or p/3 (regression)
            • Only consider these features for best split
            • This adds randomness and reduces correlation between trees
            
            **Step 3: Build Individual Trees** 🌳
            • Train decision tree on bootstrap sample
            • Use only selected features at each split
            • Grow tree to maximum depth or stopping criteria
            • No pruning needed (randomness prevents overfitting)
            
            **Step 4: Repeat for All Trees** 🔄
            • Repeat Steps 1-3 for all trees (n_estimators)
            • Each tree is different due to:
            • Different bootstrap samples
            • Different random feature selections
            
            **Step 5: Make Predictions** 🗳️
            • For new sample, pass through all trees
            • Classification: Take majority vote of all tree predictions
            • Regression: Average all tree predictions
            
            **Step 6: Calculate Confidence** 📊
            • Classification: Probability = votes/total_trees
            • Regression: Variance indicates prediction uncertainty
            """,
            
            # 6. Pseudocode
            'pseudocode': """
            🔹 **Pseudocode (Easy to Understand)**
            
            ```
            ALGORITHM: Random Forest
            
            INPUT: 
                - X: feature matrix (n_samples × p_features)
                - y: target values (n_samples × 1)
                - n_estimators: number of trees
                - max_features: features to consider at each split
            
            OUTPUT:
                - forest: collection of trained trees
            
            BEGIN
                1. INITIALIZE empty forest
                
                2. FOR i = 1 to n_estimators:
                   a. BOOTSTRAP_SAMPLE:
                      X_boot, y_boot = sample_with_replacement(X, y, size=n_samples)
                   
                   b. BUILD_TREE:
                      tree_i = DecisionTree(max_features=max_features)
                      tree_i.fit(X_boot, y_boot)
                   
                   c. ADD to forest: forest.append(tree_i)
                
                3. RETURN forest
            END
            
            PREDICTION:
            BEGIN
                1. predictions = []
                
                2. FOR each tree in forest:
                   prediction = tree.predict(X_new)
                   predictions.append(prediction)
                
                3. IF classification:
                   final_prediction = majority_vote(predictions)
                   
                4. IF regression:
                   final_prediction = average(predictions)
                
                5. RETURN final_prediction
            END
            
            FEATURE_IMPORTANCE:
            BEGIN
                1. total_importance = zeros(p_features)
                
                2. FOR each tree in forest:
                   tree_importance = tree.feature_importances_
                   total_importance += tree_importance
                
                3. average_importance = total_importance / n_estimators
                
                4. RETURN normalize(average_importance)
            END
            ```
            """,
            
            # 7. Python implementation
            'python_implementation': """
            🔹 **Python Implementation**
            
            **From Scratch (Simplified):**
            ```python
            import numpy as np
            from collections import Counter
            from decision_tree import SimpleDecisionTree  # From previous example
            
            class SimpleRandomForest:
                def __init__(self, n_estimators=100, max_features='sqrt'):
                    self.n_estimators = n_estimators
                    self.max_features = max_features
                    self.trees = []
                    self.feature_importances_ = None
                
                def bootstrap_sample(self, X, y):
                    \\\"\\\"\\\"Create bootstrap sample with replacement.\\\"\\\"\\\"
                    n_samples = X.shape[0]
                    indices = np.random.choice(n_samples, size=n_samples, replace=True)
                    return X[indices], y[indices]
                
                def fit(self, X, y):
                    \\\"\\\"\\\"Train the random forest.\\\"\\\"\\\"
                    self.trees = []
                    n_features = X.shape[1]
                    
                    # Determine max_features
                    if self.max_features == 'sqrt':
                        max_feat = int(np.sqrt(n_features))
                    elif self.max_features == 'log2':
                        max_feat = int(np.log2(n_features))
                    else:
                        max_feat = n_features // 3
                    
                    for _ in range(self.n_estimators):
                        # Bootstrap sampling
                        X_boot, y_boot = self.bootstrap_sample(X, y)
                        
                        # Train decision tree
                        tree = SimpleDecisionTree(max_features=max_feat)
                        tree.fit(X_boot, y_boot)
                        
                        self.trees.append(tree)
                    
                    # Calculate feature importances
                    self._calculate_feature_importances(X)
                
                def _calculate_feature_importances(self, X):
                    \\\"\\\"\\\"Calculate average feature importance across all trees.\\\"\\\"\\\"
                    importances = np.zeros(X.shape[1])
                    for tree in self.trees:
                        if hasattr(tree, 'feature_importances_'):
                            importances += tree.feature_importances_
                    self.feature_importances_ = importances / len(self.trees)
                
                def predict(self, X):
                    \\\"\\\"\\\"Make predictions using all trees.\\\"\\\"\\\"
                    # Get predictions from all trees
                    tree_predictions = np.array([tree.predict(X) for tree in self.trees])
                    
                    # For classification: majority vote
                    # For regression: average
                    predictions = []
                    for i in range(X.shape[0]):
                        sample_predictions = tree_predictions[:, i]
                        
                        # Majority vote (classification)
                        most_common = Counter(sample_predictions).most_common(1)
                        predictions.append(most_common[0][0])
                    
                    return np.array(predictions)
                
                def predict_proba(self, X):
                    \\\"\\\"\\\"Predict class probabilities.\\\"\\\"\\\"
                    tree_predictions = np.array([tree.predict(X) for tree in self.trees])
                    
                    probabilities = []
                    for i in range(X.shape[0]):
                        sample_predictions = tree_predictions[:, i]
                        counter = Counter(sample_predictions)
                        total = len(sample_predictions)
                        
                        prob_dict = {k: v/total for k, v in counter.items()}
                        probabilities.append(prob_dict)
                    
                    return probabilities
            
            # Example usage
            X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            
            rf = SimpleRandomForest(n_estimators=100)
            rf.fit(X_train, y_train)
            predictions = rf.predict(X_test)
            ```
            
            **Using Scikit-learn:**
            ```python
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, mean_squared_error
            
            # Classification
            rf_clf = RandomForestClassifier(
                n_estimators=100,      # Number of trees
                max_depth=10,          # Maximum depth of trees
                max_features='sqrt',   # Features to consider at each split
                random_state=42,       # For reproducibility
                n_jobs=-1             # Use all CPU cores
            )
            rf_clf.fit(X_train, y_train)
            y_pred = rf_clf.predict(X_test)
            
            # Feature importance
            importance = rf_clf.feature_importances_
            
            # Regression
            rf_reg = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            rf_reg.fit(X_train, y_train)
            y_pred = rf_reg.predict(X_test)
            ```
            """,
            
            # 8. Example with sample input/output
            'example': """
            🔹 **Complete Example: Email Spam Detection**
            
            **Input Data (Email Features):**
            ```
            Email | Word_Count | Exclamation | Links | Capitals | Spam?
            1     | 50         | 0          | 1     | 5        | No
            2     | 200        | 8          | 10    | 50       | Yes  
            3     | 100        | 1          | 2     | 10       | No
            4     | 300        | 15         | 20    | 80       | Yes
            5     | 75         | 0          | 0     | 3        | No
            ```
            
            **Step-by-Step Random Forest Training:**
            ```
            Tree 1 Bootstrap Sample: [Email 1, Email 1, Email 3, Email 4, Email 5]
            - Random Features: [Word_Count, Links, Capitals]
            - Best Split: Links > 5? → Spam=Yes, else consider Word_Count...
            
            Tree 2 Bootstrap Sample: [Email 2, Email 2, Email 3, Email 4, Email 1]
            - Random Features: [Exclamation, Links, Word_Count]
            - Best Split: Exclamation > 3? → Spam=Yes, else consider Links...
            
            Tree 3 Bootstrap Sample: [Email 1, Email 2, Email 4, Email 4, Email 5]
            - Random Features: [Capitals, Word_Count, Exclamation]
            - Best Split: Capitals > 20? → Spam=Yes, else consider Word_Count...
            
            ... (97 more trees with different samples and features)
            ```
            
            **New Email Prediction:**
            ```
            New Email: Word_Count=150, Exclamation=5, Links=8, Capitals=25
            
            Tree 1 prediction: Spam (because Links > 5)
            Tree 2 prediction: Spam (because Exclamation > 3)  
            Tree 3 prediction: Spam (because Capitals > 20)
            Tree 4 prediction: No Spam
            Tree 5 prediction: Spam
            ...
            
            Final Vote: 78 trees say "Spam", 22 trees say "No Spam"
            Final Prediction: SPAM (78% confidence)
            ```
            
            **Feature Importance Results:**
            ```
            Links: 35% importance (most discriminative)
            Capitals: 25% importance
            Exclamation: 20% importance  
            Word_Count: 20% importance
            ```
            """,
            
            # 9. Visualization explanation
            'visualization': """
            🔹 **Understanding Through Visualizations**
            
            **Forest Structure:**
            📊 Multiple decision trees side by side
            • Each tree has different structure (due to randomness)
            • Each tree sees different bootstrap sample
            • Final prediction combines all trees
            
            **Feature Importance Plot:**
            📈 Bar chart showing average importance across all trees
            • Taller bars = more important features
            • Calculated by averaging decrease in impurity across all trees
            • Helps identify which features matter most
            
            **Out-of-Bag (OOB) Error:**
            📉 Error rate using samples not seen during training
            • Each tree makes predictions on samples not in its bootstrap
            • Provides unbiased estimate of model performance
            • No need for separate validation set
            
            **Partial Dependence Plots:**
            📊 Show how predictions change with individual features
            • X-axis: Feature values
            • Y-axis: Average prediction across all trees
            • Reveals feature relationships and interactions
            
            **Tree Depth Distribution:**
            📊 Histogram showing depth of individual trees
            • Shows diversity in forest structure
            • Helps diagnose overfitting vs underfitting
            """,
            
            # 10. Time and space complexity
            'complexity': """
            🔹 **Time & Space Complexity**
            
            **Time Complexity:**
            • **Training**: O(n × log(n) × m × k) where:
              - n = number of samples
              - m = number of features
              - k = number of trees
            • **Prediction**: O(log(n) × k) average case
            • **Bootstrap Sampling**: O(n) per tree
            • **Feature Selection**: O(m) per split
            
            **Space Complexity:**
            • **Model Storage**: O(nodes × k) where nodes ≈ 2^depth
            • **Training Memory**: O(n × m) for dataset + tree structures
            • **Prediction Memory**: O(k) to store intermediate predictions
            
            **Scalability:**
            • ✅ **Parallelizable**: Trees can train independently
            • ✅ **Large Datasets**: Handles millions of samples efficiently
            • ⚠️ **Memory Usage**: Increases linearly with number of trees
            • ✅ **Feature Scaling**: Not required (tree-based splits)
            • ⚠️ **Many Trees**: Diminishing returns beyond 100-500 trees
            """,
            
            # 11. Advantages and disadvantages
            'pros_cons': """
            🔹 **Advantages** ✅
            • **High Accuracy**: Usually outperforms single decision trees
            • **Robust**: Less prone to overfitting than individual trees
            • **Feature Importance**: Provides automatic feature ranking
            • **Handles Missing Values**: Can work with incomplete data
            • **No Scaling Required**: Works with features of different scales
            • **Parallel Training**: Trees can be trained simultaneously
            • **Out-of-Bag Validation**: Built-in performance estimation
            • **Versatile**: Works for both classification and regression
            • **Outlier Resistant**: Random sampling reduces outlier impact
            
            🔹 **Disadvantages** ❌
            • **Less Interpretable**: Cannot easily trace decision logic
            • **Memory Intensive**: Stores many trees (large model size)
            • **Prediction Speed**: Slower than single tree for real-time use
            • **Hyperparameter Tuning**: More parameters to optimize
            • **Linear Relationships**: Poor at modeling simple linear patterns
            • **Bias**: Can be biased toward features with more levels
            • **Extrapolation**: Cannot predict beyond training data range
            • **Overfitting**: Can still overfit with very noisy data
            """,
            
            # 12. When to use and when NOT to use
            'usage_guide': """
            🔹 **When TO Use Random Forest** ✅
            
            **Perfect for:**
            • 🎯 **Complex Patterns**: Non-linear relationships in data
            • 📊 **Feature Selection**: Need to identify important variables
            • 🔍 **Mixed Data Types**: Numerical and categorical features together
            • 🛡️ **Robust Predictions**: Need stable, reliable model
            • 📈 **Moderate Datasets**: 1K to 1M samples work well
            • 🎲 **Noisy Data**: Dataset has outliers or missing values
            
            **Good when:**
            • Accuracy is more important than interpretability
            • You have sufficient training data (>1000 samples)
            • Features have complex interactions
            • Need confidence estimates for predictions
            • Want automatic feature importance ranking
            
            🔹 **When NOT to Use Random Forest** ❌
            
            **Avoid when:**
            • 📏 **Linear Relationships**: Simple linear patterns (use Linear Regression)
            • ⚡ **Real-time Predictions**: Need millisecond response times
            • 🔍 **Interpretability Required**: Need to explain every decision
            • 💾 **Memory Constraints**: Limited storage for model
            • 📊 **Small Datasets**: Less than 100 samples per class
            • 🎯 **Simple Patterns**: Basic relationships can be captured by simpler models
            
            **Use instead:**
            • Linear models (for linear relationships)
            • Single Decision Tree (for interpretability)
            • Gradient Boosting (for maximum accuracy)
            • Logistic Regression (for probability estimates)
            • Neural Networks (for very complex patterns)
            """,
            
            # 13. Common interview questions
            'interview_questions': """
            🔹 **Common Interview Questions & Answers**
            
            **Q1: How does Random Forest reduce overfitting compared to a single decision tree?**
            A: Random Forest reduces overfitting through:
            • Bootstrap aggregating (bagging) - averages predictions from multiple models
            • Random feature selection - prevents trees from being too similar
            • Ensemble effect - individual tree errors cancel out
            • Out-of-bag validation - provides unbiased performance estimate
            
            **Q2: What's the difference between Random Forest and Gradient Boosting?**
            A:
            • Random Forest: Trees trained in parallel, independent of each other
            • Gradient Boosting: Trees trained sequentially, each corrects previous errors
            • Random Forest: Less prone to overfitting, more robust
            • Gradient Boosting: Can achieve higher accuracy but more sensitive to noise
            
            **Q3: How do you choose the optimal number of trees (n_estimators)?**
            A:
            • Start with 100 trees (good default)
            • Plot OOB error vs number of trees
            • Choose point where OOB error plateaus
            • Consider computational constraints
            • Usually 100-500 trees is sufficient
            
            **Q4: What is Out-of-Bag (OOB) error and why is it useful?**
            A: OOB error uses samples not included in each tree's bootstrap sample for validation:
            • Provides unbiased performance estimate
            • No need for separate validation set
            • Helps detect overfitting
            • Approximately equals cross-validation error
            
            **Q5: How does feature importance work in Random Forest?**
            A: Feature importance measures average decrease in impurity when feature is used for splits across all trees. Features that create better splits (higher information gain) get higher importance scores.
            """,
            
            # 14. Common mistakes
            'common_mistakes': """
            🔹 **Common Beginner Mistakes & How to Avoid**
            
            **Mistake 1: Using too many trees** 🚫
            ❌ Setting n_estimators=1000+ without checking performance
            ✅ **Fix**: Monitor OOB error; usually 100-500 trees sufficient
            
            **Mistake 2: Not handling class imbalance** 🚫
            ❌ Ignoring imbalanced datasets leads to biased predictions
            ✅ **Fix**: Use class_weight='balanced' or balanced sampling
            
            **Mistake 3: Using default max_features** 🚫
            ❌ Not tuning max_features for dataset characteristics
            ✅ **Fix**: Try sqrt(n_features), log2(n_features), or grid search
            
            **Mistake 4: Ignoring memory constraints** 🚫
            ❌ Training huge forests that don't fit in memory
            ✅ **Fix**: Monitor memory usage, reduce n_estimators or max_depth
            
            **Mistake 5: Over-interpreting feature importance** 🚫
            ❌ "This feature is unimportant because RF ranked it low"
            ✅ **Fix**: Consider correlation with other features, try permutation importance
            
            **Mistake 6: Not using OOB score** 🚫
            ❌ Splitting data for validation when OOB is available
            ✅ **Fix**: Use OOB score for model evaluation and hyperparameter tuning
            """,
            
            # 15. Comparison with similar algorithms
            'comparisons': """
            🔹 **Random Forest vs Similar Algorithms**
            
            **Random Forest vs Single Decision Tree:**
            • **Random Forest**: Higher accuracy, less interpretable
            • **Decision Tree**: More interpretable, prone to overfitting
            • **Use Decision Tree**: When you need full interpretability
            
            **Random Forest vs Gradient Boosting:**
            • **Random Forest**: Parallel training, more robust, less tuning
            • **Gradient Boosting**: Sequential training, higher accuracy potential
            • **Use Gradient Boosting**: When maximum accuracy is needed
            
            **Random Forest vs Extra Trees:**
            • **Random Forest**: Finds best split among random features
            • **Extra Trees**: Uses random thresholds, faster training
            • **Use Extra Trees**: When training speed is critical
            
            **Random Forest vs Neural Networks:**
            • **Random Forest**: Requires less data, automatic feature selection
            • **Neural Networks**: Better for very complex patterns, needs more data
            • **Use Neural Networks**: With large datasets and complex patterns
            
            **Random Forest vs SVM:**
            • **Random Forest**: Handles non-linear patterns naturally
            • **SVM**: Better for high-dimensional data, memory efficient
            • **Use SVM**: With high-dimensional, sparse data
            """,
            
            # 16. Real-world applications
            'real_world_applications': """
            🔹 **Real-World Applications & Industry Use Cases**
            
            **🏥 Healthcare & Medicine:**
            • Medical diagnosis from symptoms and test results
            • Drug discovery and molecular analysis
            • Epidemiological studies and disease prediction
            • Medical image analysis and radiology
            • Personalized treatment recommendations
            
            **💰 Finance & Banking:**
            • Credit scoring and loan default prediction
            • Fraud detection in transactions
            • Algorithmic trading strategies
            • Risk assessment and portfolio management
            • Insurance claim processing
            
            **🛒 E-commerce & Marketing:**
            • Product recommendation systems
            • Customer segmentation and targeting
            • Price optimization and demand forecasting
            • Churn prediction and retention strategies
            • A/B testing and conversion optimization
            
            **🌱 Agriculture & Environment:**
            • Crop yield prediction and optimization
            • Species classification and biodiversity studies
            • Climate change modeling and prediction
            • Remote sensing and satellite imagery analysis
            • Precision agriculture and resource management
            
            **📱 Technology & Software:**
            • Feature selection for machine learning pipelines
            • Software bug prediction and quality assurance
            • User behavior analysis and engagement
            • Network intrusion detection
            • Search ranking and information retrieval
            
            **🏭 Manufacturing & Operations:**
            • Quality control and defect detection
            • Predictive maintenance and equipment monitoring
            • Supply chain optimization
            • Process optimization and automation
            • Safety monitoring and risk assessment
            
            **🎓 Education & Research:**
            • Student performance prediction
            • Educational content recommendation
            • Research data analysis and pattern discovery
            • Academic intervention systems
            • Curriculum optimization
            
            **💡 Key Success Factors:**
            • Sufficient training data (>1000 samples)
            • Proper hyperparameter tuning
            • Feature engineering and selection
            • Regular model updates and monitoring
            • Understanding of domain-specific requirements
            """
        }
    
    def generate_sample_data(self, task_type, n_samples=500, n_features=6):
        """Generate sample data for demonstration."""
        if task_type == 'classification':
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=max(3, n_features // 2),
                n_redundant=1,
                n_clusters_per_class=2,
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
        """Fit the Random Forest model."""
        if self.task_type == 'classification':
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=42
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
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
    
    def get_feature_importance(self):
        """Get feature importance from the model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        return self.model.feature_importances_
    
    def get_metrics(self, X, y):
        """Calculate performance metrics for the model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calculating metrics")
        
        y_pred = self.predict(X)
        metrics = {}
        
        if self.task_type == 'classification':
            metrics['Accuracy'] = accuracy_score(y, y_pred)
            metrics['Precision'] = precision_score(y, y_pred, average='weighted', zero_division=0)
            metrics['Recall'] = recall_score(y, y_pred, average='weighted', zero_division=0)
            metrics['F1 Score'] = f1_score(y, y_pred, average='weighted', zero_division=0)
        else:
            metrics['MSE'] = mean_squared_error(y, y_pred)
            metrics['RMSE'] = np.sqrt(mean_squared_error(y, y_pred))
            metrics['MAE'] = mean_absolute_error(y, y_pred)
            metrics['R² Score'] = r2_score(y, y_pred)
        
        return metrics
    
    def streamlit_interface(self):
        """Create comprehensive Streamlit interface for Random Forest."""
        st.subheader("🌲 Random Forest")
        
        theory = self.get_theory()
        
        # Main tabs for comprehensive coverage
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🎯 Overview", "📚 Deep Dive", "💻 Implementation", 
            "🧪 Interactive Demo", "❓ Q&A", "🏢 Applications"
        ])
        
        with tab1:
            # Overview Tab - Essential Information
            st.markdown("### 🎯 What is Random Forest?")
            st.markdown(theory['definition'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 🌟 Why Use It?")
                st.markdown(theory['motivation'])
                
            with col2:
                st.markdown("### 👥 Simple Analogy")
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
            st.markdown("### 🧪 Try Random Forest Yourself!")
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
        # Parameters
        st.markdown("### 🔧 Parameters")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            task_type = st.selectbox("Task Type:", ['classification', 'regression'])
        with col2:
            n_estimators = st.slider("Number of trees:", 10, 200, 100, 10)
        with col3:
            max_depth = st.slider("Max depth:", 3, 20, 10)
        with col4:
            n_samples = st.slider("Samples:", 200, 1000, 500)
        
        # Update parameters
        self.task_type = task_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        
        # Generate data and train model
        X, y = self.generate_sample_data(task_type, n_samples)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.fit(X_train, y_train)
        
        # Results section
        st.markdown("### 📊 Results")
        
        # Metrics
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
        
        # Feature importance
        st.markdown("### 📊 Feature Importance")
        importances = self.get_feature_importance()
        fig, ax = plt.subplots(figsize=(10, 6))
        feature_names = [f'Feature {i+1}' for i in range(len(importances))]
        ax.bar(feature_names, importances, alpha=0.7, color='green', edgecolor='black')
        ax.set_title('Feature Importance in Random Forest')
        ax.set_ylabel('Importance')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        plt.close()
        
        # Interpretation
        st.markdown("### 🔍 Interpretation")
        
        if task_type == 'classification':
            accuracy = test_metrics.get('Accuracy', 0)
            if accuracy > 0.9:
                st.success(f"**Excellent performance!** Accuracy: {accuracy:.1%}")
            elif accuracy > 0.8:
                st.info(f"**Good performance.** Accuracy: {accuracy:.1%}")
            elif accuracy > 0.7:
                st.warning(f"**Moderate performance.** Accuracy: {accuracy:.1%}")
            else:
                st.error(f"**Poor performance.** Accuracy: {accuracy:.1%}")
        else:
            r2_score_val = test_metrics.get('R² Score', 0)
            if r2_score_val > 0.8:
                st.success(f"**Excellent fit!** R² Score: {r2_score_val:.3f}")
            elif r2_score_val > 0.6:
                st.info(f"**Good fit.** R² Score: {r2_score_val:.3f}")
            elif r2_score_val > 0.3:
                st.warning(f"**Moderate fit.** R² Score: {r2_score_val:.3f}")
            else:
                st.error(f"**Poor fit.** R² Score: {r2_score_val:.3f}")
        
        # Forest statistics
        st.markdown("**Forest Statistics:**")
        col1, col2, col3 = st.columns(3)
        col1.metric("Number of Trees", self.n_estimators)
        col2.metric("Max Depth", self.max_depth)
        col3.metric("Number of Features", X.shape[1])


def main():
    """Main function for testing Random Forest."""
    rf = RandomForest()
    rf.streamlit_interface()


if __name__ == "__main__":
    main()