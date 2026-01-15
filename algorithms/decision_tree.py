"""
Decision Tree Algorithm Implementation

Decision trees are a non-parametric supervised learning method used for 
classification and regression. They create a model that predicts the target 
variable by learning simple decision rules inferred from the data features.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree, export_text
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report,
                           mean_squared_error, r2_score, mean_absolute_error)
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_regression
import streamlit as st
import seaborn as sns


class DecisionTree:
    """
    Decision Tree implementation with educational explanations.
    
    Decision trees work by recursively partitioning the feature space into 
    regions and making predictions based on the majority class (classification)
    or mean value (regression) in each region.
    
    Key Concepts:
    - Root Node: Top node representing the entire dataset
    - Internal Nodes: Nodes with conditions/splits
    - Leaf Nodes: Terminal nodes with predictions
    - Splitting Criteria: Rules to divide data (Gini, Entropy, MSE)
    """
    
    def __init__(self, task_type='classification', max_depth=3, min_samples_split=2):
        self.task_type = task_type
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.model = None
        self.is_fitted = False
        
    @staticmethod
    def get_theory():
        """Return comprehensive theoretical explanation of Decision Trees."""
        return {
            'name': 'Decision Tree',
            'type': 'Supervised Learning - Classification/Regression',
            
            # 1. What the algorithm is
            'definition': """
            🔹 **What is Decision Tree?**
            Decision Tree is like a flowchart that asks yes/no questions to make decisions.
            It starts with one question, then branches into more specific questions until 
            it reaches a final answer. Think of it as a "choose your own adventure" book for data!
            """,
            
            # 2. Why the algorithm is used
            'motivation': """
            🔹 **Why Use Decision Trees?**
            • 🏥 **Medical Diagnosis**: "Is fever > 101°F? → Is cough present?" → Diagnosis
            • 💳 **Loan Approval**: Income, credit score, age → Approve/reject decision
            • 🎯 **Marketing**: Customer behavior → Target specific campaigns
            • 🌦️ **Weather Prediction**: Temperature, humidity, pressure → Rain/shine
            • 🎮 **Game AI**: Player moves → Best counter-strategy
            • 📊 **Feature Selection**: Identifies most important variables automatically
            """,
            
            # 3. Intuition with real-life analogy
            'intuition': """
            🔹 **Real-Life Analogy: The Animal Guessing Game**
            
            Imagine playing "20 Questions" to guess an animal:
            
            🎯 **The Game**: Guess any animal with yes/no questions
            🧠 **Your Strategy**: Ask questions that split animals into groups
            
            **Decision Tree is like a master player who:**
            
            **Level 1**: 🦅 "Does it fly?" 
            ├─ Yes → Birds, bats, insects
            └─ No → Land/water animals
            
            **Level 2**: 🐾 "Does it have fur?"
            ├─ Yes → Mammals (cats, dogs, bears)
            └─ No → Reptiles, fish, etc.
            
            **Level 3**: 🏠 "Is it a pet?"
            ├─ Yes → Cat, Dog, Hamster
            └─ No → Wild animals
            
            **Final Guess**: "It's a DOG!" 🐕
            
            🎯 **In data terms**: 
            - Questions = Feature Tests
            - Animal Categories = Classes
            - Final Guess = Prediction
            - Question Strategy = Splitting Algorithm
            """,
            
            # 4. Mathematical foundation
            'math_foundation': """
            🔹 **Mathematical Foundation (Step-by-Step)**
            
            **Core Concept: Information Gain**
            ```
            Information Gain = Entropy(Parent) - Weighted_Average(Entropy(Children))
            ```
            
            **Where:**
            • `Entropy` = Measure of disorder/uncertainty
            • `Parent` = Current node before split
            • `Children` = Nodes after split
            • `Weighted_Average` = Based on number of samples in each child
            
            **Entropy Formula (Classification):**
            ```
            Entropy(S) = -Σ(pi × log2(pi))
            ```
            Where `pi` = proportion of samples in class i
            
            **Gini Impurity (Alternative):**
            ```
            Gini(S) = 1 - Σ(pi²)
            ```
            
            **Best Split Selection:**
            ```
            For each feature:
                For each possible threshold:
                    Calculate Information Gain
            Choose feature + threshold with highest gain
            ```
            
            **Regression (MSE):**
            ```
            MSE = (1/n) × Σ(yi - ȳ)²
            MSE_Reduction = MSE(Parent) - Weighted_Average(MSE(Children))
            ```
            """,
            
            # 5. Step-by-step working
            'algorithm_steps': """
            🔹 **How Decision Tree Works (Step-by-Step)**
            
            **Step 1: Start with Root** 🌱
            • Begin with entire dataset at root node
            • Calculate current impurity (entropy/gini/MSE)
            • Set current node as "best guess" (majority class or mean)
            
            **Step 2: Find Best Split** 🔍
            • For EVERY feature:
            • For EVERY possible threshold:
            • Calculate information gain or MSE reduction
            • Choose feature + threshold with maximum gain
            
            **Step 3: Create Children** 👶👶
            • Split data into left and right child nodes
            • Left child: samples satisfying condition
            • Right child: samples NOT satisfying condition
            
            **Step 4: Recursive Splitting** 🔄
            • Repeat Steps 2-3 for each child node
            • Continue until stopping criteria met:
            
            **Step 5: Stopping Criteria** ✋
            • Maximum depth reached
            • Minimum samples per node reached
            • No more information gain possible
            • Pure nodes achieved (all same class)
            
            **Step 6: Make Predictions** 🎯
            • Start at root with new sample
            • Follow decision path down tree
            • Reach leaf node → return prediction
            """,
            
            # 6. Pseudocode
            'pseudocode': """
            🔹 **Pseudocode (Easy to Understand)**
            
            ```
            ALGORITHM: Decision Tree
            
            INPUT: 
                - X: feature matrix (n_samples × n_features)
                - y: target values (n_samples × 1)
                - max_depth: maximum tree depth
                - min_samples: minimum samples to split
            
            OUTPUT:
                - trained_tree: decision tree model
            
            BEGIN
                1. CREATE root node with all data
                
                2. FUNCTION build_tree(node, depth):
                   a. IF stopping_criteria_met(node, depth):
                      RETURN make_leaf(node)  # majority class or mean
                   
                   b. best_gain = 0
                   c. best_split = None
                   
                   d. FOR each feature in X:
                      FOR each threshold in unique_values(feature):
                          left_data, right_data = split_data(feature, threshold)
                          gain = calculate_information_gain(left_data, right_data)
                          IF gain > best_gain:
                              best_gain = gain
                              best_split = (feature, threshold)
                   
                   e. IF best_gain == 0:
                      RETURN make_leaf(node)
                   
                   f. CREATE left_child, right_child from best_split
                   g. left_subtree = build_tree(left_child, depth+1)
                   h. right_subtree = build_tree(right_child, depth+1)
                   
                   i. RETURN internal_node(best_split, left_subtree, right_subtree)
                
                3. tree = build_tree(root, 0)
                4. RETURN tree
            END
            
            PREDICTION:
            BEGIN
                1. START at root node
                2. WHILE current_node is not leaf:
                   IF sample[feature] <= threshold:
                       current_node = left_child
                   ELSE:
                       current_node = right_child
                3. RETURN leaf_prediction
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
            
            class SimpleDecisionTree:
                def __init__(self, max_depth=3, min_samples=2):
                    self.max_depth = max_depth
                    self.min_samples = min_samples
                    self.tree = None
                
                def entropy(self, y):
                    \"\"\"Calculate entropy for classification.\"\"\"
                    counts = Counter(y)
                    total = len(y)
                    entropy = 0
                    for count in counts.values():
                        p = count / total
                        if p > 0:
                            entropy -= p * np.log2(p)
                    return entropy
                
                def information_gain(self, y_parent, y_left, y_right):
                    \"\"\"Calculate information gain from a split.\"\"\"
                    n = len(y_parent)
                    n_left, n_right = len(y_left), len(y_right)
                    
                    if n_left == 0 or n_right == 0:
                        return 0
                    
                    gain = self.entropy(y_parent)
                    gain -= (n_left/n) * self.entropy(y_left)
                    gain -= (n_right/n) * self.entropy(y_right)
                    return gain
                
                def best_split(self, X, y):
                    \"\"\"Find best feature and threshold to split.\"\"\"
                    best_gain = 0
                    best_feature = None
                    best_threshold = None
                    
                    for feature in range(X.shape[1]):
                        thresholds = np.unique(X[:, feature])
                        for threshold in thresholds:
                            left_mask = X[:, feature] <= threshold
                            y_left = y[left_mask]
                            y_right = y[~left_mask]
                            
                            gain = self.information_gain(y, y_left, y_right)
                            if gain > best_gain:
                                best_gain = gain
                                best_feature = feature
                                best_threshold = threshold
                    
                    return best_feature, best_threshold, best_gain
                
                def build_tree(self, X, y, depth=0):
                    \"\"\"Recursively build the decision tree.\"\"\"
                    # Stopping criteria
                    if (depth >= self.max_depth or 
                        len(y) < self.min_samples or 
                        len(np.unique(y)) == 1):
                        return Counter(y).most_common(1)[0][0]  # majority class
                    
                    # Find best split
                    feature, threshold, gain = self.best_split(X, y)
                    if gain == 0:
                        return Counter(y).most_common(1)[0][0]
                    
                    # Create child nodes
                    left_mask = X[:, feature] <= threshold
                    left_tree = self.build_tree(X[left_mask], y[left_mask], depth+1)
                    right_tree = self.build_tree(X[~left_mask], y[~left_mask], depth+1)
                    
                    return {
                        'feature': feature,
                        'threshold': threshold,
                        'left': left_tree,
                        'right': right_tree
                    }
                
                def fit(self, X, y):
                    \"\"\"Train the decision tree.\"\"\"
                    self.tree = self.build_tree(X, y)
                
                def predict_sample(self, sample):
                    \"\"\"Predict single sample.\"\"\"
                    node = self.tree
                    while isinstance(node, dict):
                        if sample[node['feature']] <= node['threshold']:
                            node = node['left']
                        else:
                            node = node['right']
                    return node
                
                def predict(self, X):
                    \"\"\"Predict multiple samples.\"\"\"
                    return [self.predict_sample(sample) for sample in X]
            ```
            
            **Using Scikit-learn:**
            ```python
            from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score
            
            # Classification
            clf = DecisionTreeClassifier(max_depth=3, random_state=42)
            clf.fit(X_train, y_train)
            predictions = clf.predict(X_test)
            
            # Regression
            reg = DecisionTreeRegressor(max_depth=3, random_state=42)
            reg.fit(X_train, y_train)
            predictions = reg.predict(X_test)
            
            # View tree structure
            from sklearn.tree import export_text
            tree_rules = export_text(clf, feature_names=['feature_1', 'feature_2'])
            print(tree_rules)
            ```
            """,
            
            # 8. Example with sample input/output
            'example': """
            🔹 **Complete Example: Tennis Playing Decision**
            
            **Input Data (Should I play tennis?):**
            ```
            Weather    | Temperature | Humidity | Wind   | Play?
            Sunny      | Hot         | High     | Weak   | No
            Sunny      | Hot         | High     | Strong | No  
            Overcast   | Hot         | High     | Weak   | Yes
            Rain       | Mild        | High     | Weak   | Yes
            Rain       | Cool        | Normal   | Weak   | Yes
            Rain       | Cool        | Normal   | Strong | No
            Overcast   | Cool        | Normal   | Strong | Yes
            Sunny      | Mild        | High     | Weak   | No
            Sunny      | Cool        | Normal   | Weak   | Yes
            Rain       | Mild        | Normal   | Weak   | Yes
            ```
            
            **Step-by-Step Tree Building:**
            ```
            1. Root Node: 14 samples (9 Yes, 5 No) → Entropy = 0.94
            
            2. Best Split Found: Weather = "Sunny"?
               - Left (Sunny): 5 samples (2 Yes, 3 No) → Entropy = 0.97
               - Right (Not Sunny): 9 samples (7 Yes, 2 No) → Entropy = 0.76
               - Information Gain = 0.94 - (5/14)×0.97 - (9/14)×0.76 = 0.25
            
            3. Split Sunny Branch: Humidity = "High"?
               - Left (High): 3 samples (0 Yes, 3 No) → Pure! → Predict: NO
               - Right (Normal): 2 samples (2 Yes, 0 No) → Pure! → Predict: YES
            
            4. Split Non-Sunny Branch: Weather = "Rain"?
               - Rain + Wind Strong: NO
               - Rain + Wind Weak: YES  
               - Overcast: YES
            ```
            
            **Final Decision Tree:**
            ```
            Weather = Sunny?
            ├─ Yes: Humidity = High?
            │   ├─ Yes: DON'T PLAY ❌
            │   └─ No: PLAY ✅
            └─ No: Weather = Rain?
                ├─ Yes: Wind = Strong?
                │   ├─ Yes: DON'T PLAY ❌
                │   └─ No: PLAY ✅
                └─ No (Overcast): PLAY ✅
            ```
            
            **New Prediction:**
            ```
            New Day: Sunny, Mild, Normal, Weak
            Path: Sunny? → Yes → Humidity High? → No → PLAY! ✅
            ```
            """,
            
            # 9. Visualization explanation
            'visualization': """
            🔹 **Understanding Through Visualizations**
            
            **Tree Structure Plot:**
            📊 Shows complete decision flow
            • Root at top, leaves at bottom
            • Each box shows: splitting condition, samples, predicted class
            • Colors indicate class purity (darker = more pure)
            • Path from root to leaf = decision sequence
            
            **Feature Importance Chart:**
            📈 Bar chart showing feature contributions
            • Height = how much feature reduces impurity
            • Features used higher in tree = more important
            • Sum of all importances = 1.0
            
            **Decision Boundaries (2D features):**
            🎯 Rectangular regions in feature space
            • Each region = one leaf node
            • Parallel lines to axes (axis-aligned splits)
            • Different colors = different predicted classes
            
            **Confusion Matrix (Classification):**
            📋 Actual vs Predicted class counts
            • Diagonal = correct predictions
            • Off-diagonal = errors
            • Perfect tree = only diagonal values
            
            **Residual Plots (Regression):**
            📉 Prediction errors vs predicted values
            • Good tree = random scatter around zero
            • Patterns indicate underfitting or overfitting
            """,
            
            # 10. Time and space complexity
            'complexity': """
            🔹 **Time & Space Complexity**
            
            **Time Complexity:**
            • **Training**: O(n × m × log(n)) where n=samples, m=features
            • **Prediction**: O(log(n)) average, O(n) worst case (unbalanced tree)
            • **Tree Building**: O(n × m × d) where d=depth
            • **Split Finding**: O(n × log(n)) per feature per node
            
            **Space Complexity:**
            • **Model Storage**: O(nodes) = O(2^d) worst case, O(log(n)) average
            • **Training Memory**: O(n × m) to store dataset
            • **Recursion Stack**: O(d) for tree building
            
            **Scalability:**
            • ✅ **Fast Prediction**: Logarithmic time for most trees
            • ⚠️ **Training Time**: Can be slow with many features
            • ✅ **Memory Efficient**: Only stores split conditions
            • ❌ **Unbalanced Trees**: Can degrade to linear time
            """,
            
            # 11. Advantages and disadvantages
            'pros_cons': """
            🔹 **Advantages** ✅
            • **Highly Interpretable**: Easy to understand decision rules
            • **No Assumptions**: Works with any data distribution
            • **Handles Mixed Data**: Numerical and categorical features
            • **Feature Selection**: Automatically identifies important features
            • **No Scaling Needed**: Robust to different feature scales
            • **Missing Values**: Can handle missing data naturally
            • **Non-linear**: Captures complex interactions and patterns
            • **Fast Predictions**: Logarithmic time complexity
            • **Rule Extraction**: Provides explicit if-then rules
            
            🔹 **Disadvantages** ❌
            • **Overfitting**: Creates overly complex trees on training data
            • **Instability**: Small data changes create completely different trees
            • **Bias**: Favors features with more distinct values
            • **Linear Relationships**: Poor at modeling simple linear patterns
            • **Probability Estimates**: Provides poor class probabilities
            • **Memory Growth**: Tree size can grow exponentially
            • **Extrapolation**: Cannot predict beyond training data range
            """,
            
            # 12. When to use and when NOT to use
            'usage_guide': """
            🔹 **When TO Use Decision Trees** ✅
            
            **Perfect for:**
            • 🎯 **Rule-based Problems**: Need clear, interpretable rules
            • 🏥 **Medical Diagnosis**: Doctors need explainable decisions
            • 💼 **Business Rules**: Convert decisions into business logic
            • 🔍 **Feature Exploration**: Understanding which features matter
            • 📊 **Mixed Data Types**: Combination of numerical and categorical
            • 🚀 **Prototyping**: Quick baseline for any classification/regression
            
            **Good when:**
            • Interpretability is more important than accuracy
            • Data has complex non-linear interactions
            • Features have different scales and types
            • You need automatic feature selection
            • Training data is limited
            
            🔹 **When NOT to Use Decision Trees** ❌
            
            **Avoid when:**
            • 📏 **Linear Relationships**: Simple linear patterns (use Linear Regression)
            • 🎯 **High Accuracy Needed**: Better algorithms available (Random Forest)
            • 📊 **Small Datasets**: Prone to overfitting with limited data
            • 🎲 **High Noise**: Unstable with very noisy data
            • 📈 **Continuous Smooth Functions**: Cannot model smooth curves
            • ⚡ **Real-time Learning**: Need to update model frequently
            
            **Use instead:**
            • Linear models (for linear relationships)
            • Random Forest (for better accuracy)
            • Neural Networks (for complex patterns)
            • Ensemble methods (for stability)
            """,
            
            # 13. Common interview questions
            'interview_questions': """
            🔹 **Common Interview Questions & Answers**
            
            **Q1: How do you prevent overfitting in decision trees?**
            A: 
            • Pre-pruning: Set max_depth, min_samples_split, min_samples_leaf
            • Post-pruning: Build full tree then remove unnecessary branches
            • Cross-validation: Use validation data to select optimal parameters
            • Ensemble methods: Use Random Forest instead of single tree
            
            **Q2: What's the difference between Gini and Entropy?**
            A:
            • Gini: Faster to compute, range [0, 0.5], prefers largest class
            • Entropy: More theoretically grounded, range [0, 1], better balanced splits
            • In practice: Very similar results, Gini slightly faster
            
            **Q3: Why are decision trees unstable?**
            A: Small changes in training data can create completely different trees because:
            • Greedy algorithm: Makes locally optimal decisions
            • Hierarchical structure: Early split changes affect entire subtree
            • Solution: Use ensemble methods like Random Forest
            
            **Q4: How do decision trees handle continuous features?**
            A: 
            • Sort unique values of feature
            • Try all possible thresholds as split points
            • Choose threshold that maximizes information gain
            • Creates binary splits: ≤ threshold vs > threshold
            
            **Q5: Can decision trees do feature selection?**
            A: Yes! Features not used in any split have zero importance. Features used higher in tree or in more nodes have higher importance. Tree automatically ignores irrelevant features.
            """,
            
            # 14. Common mistakes
            'common_mistakes': """
            🔹 **Common Beginner Mistakes & How to Avoid**
            
            **Mistake 1: Not limiting tree depth** 🚫
            ❌ Letting tree grow too deep and memorizing training data
            ✅ **Fix**: Set max_depth=5-10, use validation to find optimal depth
            
            **Mistake 2: Ignoring class imbalance** 🚫
            ❌ Tree biased toward majority class with imbalanced data
            ✅ **Fix**: Use class_weight='balanced' or stratified sampling
            
            **Mistake 3: Using single tree for production** 🚫
            ❌ Single trees are unstable and overfit easily
            ✅ **Fix**: Use Random Forest or Gradient Boosting instead
            
            **Mistake 4: Not validating splits** 🚫
            ❌ Trusting tree performance on training data only
            ✅ **Fix**: Always use cross-validation or separate test set
            
            **Mistake 5: Over-interpreting feature importance** 🚫
            ❌ "This feature is unimportant because tree didn't use it"
            ✅ **Fix**: Consider feature interactions, try multiple tree configurations
            
            **Mistake 6: Expecting smooth predictions** 🚫
            ❌ Trees create step-wise predictions, not smooth curves
            ✅ **Fix**: Use ensemble methods or other algorithms for smooth functions
            """,
            
            # 15. Comparison with similar algorithms
            'comparisons': """
            🔹 **Decision Trees vs Similar Algorithms**
            
            **Decision Tree vs Random Forest:**
            • **Decision Tree**: Single tree, interpretable, unstable
            • **Random Forest**: Multiple trees, more accurate, less interpretable
            • **Use Random Forest**: When accuracy > interpretability
            
            **Decision Tree vs Logistic Regression:**
            • **Decision Tree**: Captures non-linear patterns, no assumptions
            • **Logistic Regression**: Linear decision boundary, probabilistic
            • **Use Logistic**: When relationship is roughly linear
            
            **Decision Tree vs K-Nearest Neighbors:**
            • **Decision Tree**: Creates explicit rules, fast prediction
            • **KNN**: Instance-based, no training phase, smooth boundaries
            • **Use KNN**: When local patterns matter more than global rules
            
            **Decision Tree vs Neural Networks:**
            • **Decision Tree**: Interpretable, handles mixed data easily
            • **Neural Networks**: More flexible, handles complex patterns
            • **Use Neural Networks**: When you have lots of data and need high accuracy
            
            **Decision Tree vs Naive Bayes:**
            • **Decision Tree**: No independence assumptions, handles interactions
            • **Naive Bayes**: Assumes feature independence, probabilistic
            • **Use Naive Bayes**: When features are mostly independent
            """,
            
            # 16. Real-world applications
            'real_world_applications': """
            🔹 **Real-World Applications & Industry Use Cases**
            
            **🏥 Healthcare & Medicine:**
            • Medical diagnosis decision support systems
            • Drug dosage determination based on patient factors
            • Treatment path recommendations
            • Clinical trial patient stratification
            • Epidemic outbreak prediction
            
            **💰 Finance & Banking:**
            • Credit approval and loan default prediction
            • Fraud detection in transactions
            • Investment portfolio risk assessment
            • Insurance claim processing
            • Algorithmic trading rule generation
            
            **🛒 E-commerce & Retail:**
            • Customer segmentation and targeting
            • Product recommendation systems
            • Inventory management decisions
            • Price optimization strategies
            • Supply chain optimization
            
            **🏭 Manufacturing & Operations:**
            • Quality control and defect detection
            • Predictive maintenance scheduling
            • Production planning optimization
            • Equipment failure diagnosis
            • Process control automation
            
            **📱 Technology & Software:**
            • User behavior analysis
            • Feature flagging and A/B testing
            • Content recommendation engines
            • Cybersecurity threat detection
            • Resource allocation in cloud services
            
            **🎓 Education & Research:**
            • Student performance prediction
            • Curriculum design optimization
            • Learning path personalization
            • Research data mining
            • Academic intervention systems
            
            **🌱 Agriculture & Environment:**
            • Crop yield prediction
            • Pest and disease identification
            • Weather pattern analysis
            • Environmental monitoring
            • Species classification
            
            **💡 Key Success Factors:**
            • Domain expertise for feature engineering
            • Proper tree pruning to prevent overfitting
            • Regular model validation and updating
            • Ensemble methods for production systems
            • Clear documentation of decision rules
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
        """Fit the decision tree model."""
        if self.task_type == 'classification':
            self.model = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=42
            )
        else:
            self.model = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
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
    
    def plot_tree_structure(self, feature_names=None):
        """Plot the decision tree structure."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting tree")
            
        if feature_names is None:
            n_features = self.model.n_features_in_
            feature_names = [f'Feature {i+1}' for i in range(n_features)]
        
        fig, ax = plt.subplots(figsize=(20, 12))
        
        plot_tree(
            self.model,
            feature_names=feature_names,
            filled=True,
            rounded=True,
            fontsize=10,
            ax=ax
        )
        
        ax.set_title(f'Decision Tree Structure (max_depth={self.max_depth})', 
                    fontsize=16, fontweight='bold')
        
        return fig
    
    def get_tree_rules(self, feature_names=None):
        """Extract decision rules as text."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before extracting rules")
            
        if feature_names is None:
            n_features = self.model.n_features_in_
            feature_names = [f'Feature {i+1}' for i in range(n_features)]
        
        return export_text(self.model, feature_names=feature_names)
    
    def plot_feature_importance(self, feature_names=None):
        """Plot feature importance from the decision tree."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting feature importance")
            
        importances = self.model.feature_importances_
        
        if feature_names is None:
            feature_names = [f'Feature {i+1}' for i in range(len(importances))]
        
        # Sort features by importance
        indices = np.argsort(importances)[::-1]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(range(len(importances)), importances[indices], 
                     alpha=0.7, color='lightgreen', edgecolor='black')
        
        ax.set_xlabel('Features')
        ax.set_ylabel('Importance')
        ax.set_title('Feature Importance in Decision Tree')
        ax.set_xticks(range(len(importances)))
        ax.set_xticklabels([feature_names[i] for i in indices], rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, importance in zip(bars, importances[indices]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{importance:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def plot_classification_results(self, X, y):
        """Create classification visualization (classification only)."""
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
        """Create regression visualization (regression only)."""
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
        
        # Feature importance (if tree is fitted)
        if hasattr(self.model, 'feature_importances_'):
            feature_names = [f'Feature {i+1}' for i in range(X.shape[1])]
            importances = self.model.feature_importances_
            
            bars = axes[1, 1].bar(feature_names, importances, alpha=0.7, 
                                color='lightgreen', edgecolor='black')
            axes[1, 1].set_xlabel('Features')
            axes[1, 1].set_ylabel('Importance')
            axes[1, 1].set_title('Feature Importance')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Rotate x-axis labels if many features
            if len(feature_names) > 5:
                plt.setp(axes[1, 1].get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        return fig
    
    def streamlit_interface(self):
        """Create comprehensive Streamlit interface for Decision Trees."""
        st.subheader("🌳 Decision Tree")
        
        theory = self.get_theory()
        
        # Main tabs for comprehensive coverage
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🎯 Overview", "📚 Deep Dive", "💻 Implementation", 
            "🧪 Interactive Demo", "❓ Q&A", "🏢 Applications"
        ])
        
        with tab1:
            # Overview Tab - Essential Information
            st.markdown("### 🎯 What is Decision Tree?")
            st.markdown(theory['definition'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 🌟 Why Use It?")
                st.markdown(theory['motivation'])
                
            with col2:
                st.markdown("### 🎮 Simple Analogy")
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
            st.markdown("### 🧪 Try Decision Tree Yourself!")
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
            max_depth = st.slider("Max depth:", 1, 10, 3)
        
        # Update task type
        self.task_type = task_type
        self.max_depth = max_depth
        
        # Generate and split data
        X, y = self.generate_sample_data(task_type, n_samples, n_features)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
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
        
        # Tree structure visualization
        st.markdown("### 🌲 Tree Structure")
        
        # Limit tree size for better visualization
        if max_depth <= 4 and n_features <= 6:
            fig_tree = self.plot_tree_structure()
            st.pyplot(fig_tree)
            plt.close()
        else:
            st.warning("Tree too large for visualization. Showing text rules instead.")
        
        # Decision rules
        with st.expander("📋 Decision Rules", expanded=False):
            rules = self.get_tree_rules()
            st.text(rules)
        
        # Feature importance
        st.markdown("### 📊 Feature Importance")
        fig_importance = self.plot_feature_importance()
        st.pyplot(fig_importance)
        plt.close()
        
        # Task-specific visualizations
        st.markdown("### 📈 Model Performance")
        
        if task_type == 'classification':
            fig_results = self.plot_classification_results(X_test, y_test)
            st.pyplot(fig_results)
            plt.close()
            
            # Classification report
            y_pred_test = self.predict(X_test)
            report = classification_report(y_test, y_pred_test, output_dict=True)
            
            st.markdown("**Detailed Classification Report:**")
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.round(4))
            
        else:  # regression
            fig_results = self.plot_regression_results(X_test, y_test)
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
        
        # Tree statistics
        st.markdown("**Tree Statistics:**")
        col1, col2, col3 = st.columns(3)
        col1.metric("Tree Depth", self.model.get_depth())
        col2.metric("Number of Leaves", self.model.get_n_leaves())
        col3.metric("Number of Features Used", np.sum(self.model.feature_importances_ > 0))


def main():
    """Main function for testing Decision Tree."""
    dt = DecisionTree()
    dt.streamlit_interface()


if __name__ == "__main__":
    main()