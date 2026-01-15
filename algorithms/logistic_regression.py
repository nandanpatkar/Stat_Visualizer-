"""
Logistic Regression Algorithm Implementation

Logistic regression is a statistical method used for binary classification tasks.
It uses the logistic function to model the probability of binary outcomes.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report,
                           roc_curve, auc, log_loss)
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import streamlit as st
import seaborn as sns


class LogisticRegression:
    """
    Logistic Regression implementation with educational explanations.
    
    Logistic regression uses the sigmoid function to map any real number 
    to a value between 0 and 1, making it ideal for probability estimation
    and binary classification.
    
    Mathematical Formula: p = 1 / (1 + e^(-(β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ)))
    
    Where:
    - p: probability of positive class
    - e: Euler's number (≈2.718)
    - β₀: intercept
    - β₁, β₂, ..., βₙ: coefficients
    - x₁, x₂, ..., xₙ: features
    """
    
    def __init__(self, max_iter=1000):
        self.model = None
        self.is_fitted = False
        self.max_iter = max_iter
        
    @staticmethod
    def get_theory():
        """Return comprehensive theoretical explanation of Logistic Regression."""
        return {
            'name': 'Logistic Regression',
            'type': 'Supervised Learning - Classification',
            
            # 1. What the algorithm is
            'definition': """
            🔹 **What is Logistic Regression?**
            Logistic Regression is like a smart yes/no decision maker. Instead of predicting 
            exact numbers (like Linear Regression), it predicts the probability of something 
            belonging to a category - like "Will this email be spam?" or "Will this customer buy?"
            """,
            
            # 2. Why the algorithm is used
            'motivation': """
            🔹 **Why Use Logistic Regression?**
            • 📧 **Spam Detection**: Is this email spam or not?
            • 🏥 **Medical Diagnosis**: Does patient have disease or not?
            • 💳 **Credit Approval**: Approve or reject loan application?
            • 🛒 **Marketing**: Will customer buy this product?
            • 🎯 **A/B Testing**: Which version performs better?
            • 📊 **Risk Assessment**: High risk or low risk investment?
            """,
            
            # 3. Intuition with real-life analogy
            'intuition': """
            🔹 **Real-Life Analogy: The College Admission Officer**
            
            Imagine you're a college admission officer deciding whether to admit students:
            
            🎓 **The Problem**: Admit or reject each applicant?
            📊 **Input factors**: GPA, SAT score, extracurriculars, essays
            
            **Logistic Regression is like an experienced officer who:**
            
            **Step 1**: 📏 Looks at all factors and calculates a "readiness score"
            **Step 2**: 🎯 Uses the sigmoid function: "The higher the score, the more likely to admit"
            **Step 3**: 📈 Converts score to probability: "This student has 85% chance of admission"
            **Step 4**: ✅ Makes decision: If probability > 50%, admit; otherwise, reject
            
            **The Magic**: Instead of saying "score = 7.5", it says "85% chance of success"
            
            🎯 **In data terms**: 
            - Students = Data Points
            - Factors = Features  
            - Admission Decision = Class Label (0 or 1)
            - Officer's Experience = Trained Model
            """,
            
            # 4. Mathematical foundation
            'math_foundation': """
            🔹 **Mathematical Foundation (Step-by-Step)**
            
            **Core Formula (Sigmoid Function):**
            ```
            p = 1 / (1 + e^(-(β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ)))
            ```
            
            **Where:**
            • `p` = Probability of positive class (0 to 1)
            • `e` = Euler's number (≈ 2.718)
            • `β₀` = Intercept (bias term)
            • `β₁, β₂, ...` = Coefficients (feature weights)
            • `x₁, x₂, ...` = Input features
            
            **Logit (Log-Odds):**
            ```
            logit(p) = ln(p/(1-p)) = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
            ```
            
            **Odds Ratio:**
            ```
            odds = p / (1-p)
            ```
            
            **Cost Function (Maximum Likelihood):**
            ```
            Cost = -Σ[y*log(p) + (1-y)*log(1-p)]
            ```
            *Translation: Penalize wrong predictions heavily*
            
            **Decision Rule:**
            ```
            If p ≥ 0.5: Predict Class 1
            If p < 0.5: Predict Class 0
            ```
            """,
            
            # 5. Step-by-step working
            'algorithm_steps': """
            🔹 **How Logistic Regression Works (Step-by-Step)**
            
            **Step 1: Data Preparation** 📋
            • Collect training data with features (X) and binary labels (y)
            • Clean data and handle missing values
            • Split into training and testing sets
            
            **Step 2: Initialize Coefficients** 🎲
            • Start with random weights (β₀, β₁, β₂, ...)
            • These will be learned during training
            
            **Step 3: Calculate Linear Combination** 📐
            • For each data point: z = β₀ + β₁x₁ + β₂x₂ + ...
            • This gives us the "raw score" for each example
            
            **Step 4: Apply Sigmoid Function** 🎯
            • Transform z into probability: p = 1/(1 + e^(-z))
            • Now we have probabilities between 0 and 1
            
            **Step 5: Calculate Cost** 📏
            • Compare predictions with actual labels
            • Use log-likelihood to measure how wrong we are
            • High cost = bad predictions, low cost = good predictions
            
            **Step 6: Update Coefficients** 🔄
            • Use gradient descent to minimize cost
            • Adjust weights to make better predictions
            • Repeat until cost stops decreasing
            
            **Step 7: Make Predictions** ✅
            • For new data: Calculate z, apply sigmoid, get probability
            • If p ≥ 0.5: Predict positive class
            • If p < 0.5: Predict negative class
            """,
            
            # 6. Pseudocode
            'pseudocode': """
            🔹 **Pseudocode (Easy to Understand)**
            
            ```
            ALGORITHM: Logistic Regression
            
            INPUT: 
                - X: features matrix (n_samples × n_features)
                - y: binary labels (n_samples × 1)
                - learning_rate: step size for updates
                - max_iterations: stopping condition
            
            OUTPUT:
                - coefficients: β₀, β₁, β₂, ..., βₙ
                - prediction function
            
            BEGIN
                1. INITIALIZE coefficients randomly
                   coefficients = random_small_values(n_features + 1)
                
                2. FOR iteration = 1 to max_iterations:
                   
                   a. CALCULATE linear combination:
                      z = X × coefficients  // Matrix multiplication
                   
                   b. APPLY sigmoid function:
                      probabilities = 1 / (1 + exp(-z))
                   
                   c. CALCULATE cost (log-likelihood):
                      cost = -mean(y*log(p) + (1-y)*log(1-p))
                   
                   d. CALCULATE gradients:
                      gradients = X^T × (probabilities - y) / n_samples
                   
                   e. UPDATE coefficients:
                      coefficients = coefficients - learning_rate × gradients
                   
                   f. CHECK convergence:
                      IF cost barely changed: BREAK
                
                3. PREDICT function:
                   FOR new_data:
                       z = new_data × coefficients
                       probability = 1 / (1 + exp(-z))
                       IF probability ≥ 0.5: RETURN 1
                       ELSE: RETURN 0
            END
            ```
            """,
            
            # 7. Python implementation
            'python_implementation': """
            🔹 **Python Implementation**
            
            **From Scratch:**
            ```python
            import numpy as np
            import matplotlib.pyplot as plt
            
            class LogisticRegressionFromScratch:
                def __init__(self, learning_rate=0.01, max_iterations=1000):
                    self.learning_rate = learning_rate
                    self.max_iterations = max_iterations
                    self.coefficients = None
                
                def sigmoid(self, z):
                    # Clip z to prevent overflow
                    z = np.clip(z, -250, 250)
                    return 1 / (1 + np.exp(-z))
                
                def fit(self, X, y):
                    # Add bias term (intercept)
                    X_with_bias = np.column_stack([np.ones(len(X)), X])
                    
                    # Initialize coefficients
                    self.coefficients = np.random.normal(0, 0.01, X_with_bias.shape[1])
                    
                    # Gradient descent
                    for i in range(self.max_iterations):
                        # Forward pass
                        z = X_with_bias @ self.coefficients
                        predictions = self.sigmoid(z)
                        
                        # Calculate cost
                        cost = -np.mean(y*np.log(predictions + 1e-15) + 
                                      (1-y)*np.log(1-predictions + 1e-15))
                        
                        # Calculate gradients
                        gradients = X_with_bias.T @ (predictions - y) / len(y)
                        
                        # Update coefficients
                        self.coefficients -= self.learning_rate * gradients
                        
                        # Check convergence (optional)
                        if i % 100 == 0:
                            print(f"Iteration {i}, Cost: {cost:.4f}")
                
                def predict_proba(self, X):
                    X_with_bias = np.column_stack([np.ones(len(X)), X])
                    z = X_with_bias @ self.coefficients
                    return self.sigmoid(z)
                
                def predict(self, X):
                    return (self.predict_proba(X) >= 0.5).astype(int)
            
            # Example usage
            from sklearn.datasets import make_classification
            X, y = make_classification(n_samples=1000, n_features=2, 
                                     n_redundant=0, random_state=42)
            
            model = LogisticRegressionFromScratch(learning_rate=0.1, max_iterations=1000)
            model.fit(X, y)
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)
            ```
            
            **Using Scikit-learn:**
            ```python
            from sklearn.linear_model import LogisticRegression
            from sklearn.datasets import make_classification
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, classification_report
            
            # Generate sample data
            X, y = make_classification(n_samples=1000, n_features=4, 
                                     n_classes=2, random_state=42)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)
            
            # Create and train model
            model = LogisticRegression(random_state=42)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            
            # Evaluate
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Accuracy: {accuracy:.3f}")
            print(f"Coefficients: {model.coef_}")
            print(f"Intercept: {model.intercept_}")
            ```
            """,
            
            # 8. Example with sample input/output
            'example': """
            🔹 **Complete Example: Email Spam Detection**
            
            **Input Data:**
            ```
            Email | Word Count | Link Count | Capital Ratio | Spam (0/1)
            1     | 50         | 0          | 0.1          | 0
            2     | 200        | 5          | 0.8          | 1
            3     | 30         | 1          | 0.2          | 0
            4     | 150        | 3          | 0.6          | 1
            5     | 80         | 0          | 0.3          | 0
            ```
            
            **Step-by-Step Calculation:**
            ```
            1. Features: X = [[50,0,0.1], [200,5,0.8], [30,1,0.2], [150,3,0.6], [80,0,0.3]]
            2. Labels: y = [0, 1, 0, 1, 0]
            
            3. After training, we get coefficients:
               - Intercept (β₀) = -2.5
               - Word Count (β₁) = 0.01
               - Link Count (β₂) = 0.8  
               - Capital Ratio (β₃) = 3.0
            
            4. Formula: p = 1 / (1 + e^(-(−2.5 + 0.01×words + 0.8×links + 3.0×capitals)))
            ```
            
            **New Email Prediction:**
            ```
            New email: 100 words, 2 links, 0.5 capital ratio
            
            z = -2.5 + 0.01×100 + 0.8×2 + 3.0×0.5
              = -2.5 + 1.0 + 1.6 + 1.5
              = 1.6
            
            p = 1 / (1 + e^(-1.6))
              = 1 / (1 + 0.202)
              = 0.83
            
            Since p = 0.83 > 0.5 → Predict SPAM
            Confidence: 83%
            ```
            """,
            
            # 9. Visualization explanation
            'visualization': """
            🔹 **Understanding Through Visualizations**
            
            **Sigmoid Curve:**
            📈 S-shaped curve showing probability transformation
            • X-axis: Linear combination (z = β₀ + β₁x₁ + ...)
            • Y-axis: Probability (0 to 1)
            • Shape: Smooth transition from 0 to 1
            • Decision boundary at p = 0.5 (z = 0)
            
            **Decision Boundary (2D):**
            📊 Line separating two classes
            • Points above line: Predicted as Class 1
            • Points below line: Predicted as Class 0
            • Line equation: β₀ + β₁x₁ + β₂x₂ = 0
            
            **Probability Contours:**
            🎯 Colored regions showing probability levels
            • Red regions: High probability of Class 1
            • Blue regions: High probability of Class 0
            • Gradient: Smooth transition between classes
            
            **ROC Curve:**
            📈 Performance evaluation plot
            • X-axis: False Positive Rate
            • Y-axis: True Positive Rate
            • Good model: Curve towards top-left corner
            • AUC: Area under curve (higher = better)
            """,
            
            # 10. Time and space complexity
            'complexity': """
            🔹 **Time & Space Complexity**
            
            **Time Complexity:**
            • **Training**: O(n × p × i) 
              - n = number of samples
              - p = number of features
              - i = number of iterations (usually 100-1000)
            • **Prediction**: O(p) per prediction
            • **Gradient Calculation**: O(n × p) per iteration
            
            **Space Complexity:**
            • **Model Storage**: O(p) - store coefficients only
            • **Training Memory**: O(n × p) - store dataset
            • **Prediction Memory**: O(1) per prediction
            • **Gradient Storage**: O(p) - temporary gradients
            
            **Scalability:**
            • ✅ **Large n (samples)**: Handles millions of samples well
            • ✅ **Large p (features)**: Efficient with thousands of features
            • ✅ **Streaming**: Can update model incrementally
            • ⚡ **Speed**: Very fast training and prediction
            """,
            
            # 11. Advantages and disadvantages
            'pros_cons': """
            🔹 **Advantages** ✅
            • **Probabilistic output**: Gives confidence scores (0-1)
            • **Interpretable**: Coefficients show feature importance
            • **Fast**: Quick training and prediction
            • **No hyperparameters**: Works well with default settings
            • **Robust**: Less prone to overfitting than complex models
            • **Versatile**: Works with any number of features
            • **Statistical foundation**: Well-understood mathematical basis
            
            🔹 **Disadvantages** ❌
            • **Linear assumption**: Only finds linear decision boundaries
            • **Outlier sensitive**: Extreme values can skew results
            • **Feature scaling needed**: Performance depends on feature scales
            • **Large sample requirement**: Needs sufficient data for stability
            • **Complex relationships**: Can't capture feature interactions
            • **Perfect separation**: Problems when classes are perfectly separable
            """,
            
            # 12. When to use and when NOT to use
            'usage_guide': """
            🔹 **When TO Use Logistic Regression** ✅
            
            **Perfect for:**
            • 📧 **Binary classification**: Spam/not spam, fraud/legitimate
            • 🎯 **Probability estimation**: Need confidence scores
            • 📊 **Baseline models**: Quick first model for classification
            • 🔍 **Feature analysis**: Understanding which features matter
            • ⚡ **Real-time prediction**: Fast inference required
            • 📈 **Linear relationships**: When decision boundary is roughly linear
            
            **Good when:**
            • Need interpretable results
            • Have clean, preprocessed data
            • Classes are roughly balanced
            • Features are not highly correlated
            • Sample size is adequate (100+ per feature)
            
            🔹 **When NOT to Use Logistic Regression** ❌
            
            **Avoid when:**
            • 🌀 **Non-linear relationships**: Complex curved decision boundaries
            • 🔢 **Multi-class problems**: Many categories (use multinomial instead)
            • 📊 **Feature interactions**: Complex relationships between features
            • 🎲 **Perfect separation**: One feature perfectly separates classes
            • 📈 **High-dimensional**: More features than samples
            • 🎯 **Imbalanced classes**: Very uneven class distribution
            
            **Use instead:**
            • Random Forest (for non-linear relationships)
            • SVM (for high-dimensional data)
            • Neural Networks (for complex patterns)
            • XGBoost (for feature interactions)
            """,
            
            # 13. Common interview questions
            'interview_questions': """
            🔹 **Common Interview Questions & Answers**
            
            **Q1: Why is it called "Logistic Regression" when it's used for classification?**
            A: Historical reasons! It uses the logistic (sigmoid) function to model probabilities, and the math is similar to linear regression. The "regression" refers to the method of finding coefficients, even though the output is used for classification.
            
            **Q2: What's the difference between Linear and Logistic Regression?**
            A:
            • **Linear**: Predicts continuous values, uses straight line
            • **Logistic**: Predicts probabilities (0-1), uses S-shaped curve
            • **Output**: Linear gives any number, Logistic gives 0-1 probability
            • **Use**: Linear for "how much?", Logistic for "yes/no?"
            
            **Q3: Why do we use the sigmoid function?**
            A: Three key reasons:
            • **Bounds output**: Always between 0 and 1 (perfect for probabilities)
            • **Smooth**: Differentiable everywhere (needed for gradient descent)
            • **Interpretable**: S-shape matches many real-world probability patterns
            
            **Q4: How do you handle imbalanced classes?**
            A: Several approaches:
            • **Class weights**: Penalize mistakes on minority class more
            • **Resampling**: Over-sample minority or under-sample majority
            • **Threshold tuning**: Change decision threshold from 0.5
            • **Different metrics**: Use F1-score, AUC instead of accuracy
            
            **Q5: What causes the "perfect separation" problem?**
            A: When one feature perfectly separates the classes, coefficients become infinite. Solutions:
            • **Regularization**: Add L1/L2 penalty terms
            • **More data**: Collect examples that break the pattern
            • **Feature engineering**: Combine or transform features
            """,
            
            # 14. Common mistakes
            'common_mistakes': """
            🔹 **Common Beginner Mistakes & How to Avoid**
            
            **Mistake 1: Not scaling features** 🚫
            ❌ Using features with very different scales (age vs income)
            ✅ **Fix**: Standardize features using StandardScaler or MinMaxScaler
            
            **Mistake 2: Ignoring class imbalance** 🚫
            ❌ Treating 95% vs 5% class split like 50% vs 50%
            ✅ **Fix**: Use class_weight='balanced' or adjust threshold
            
            **Mistake 3: Using accuracy for imbalanced data** 🚫
            ❌ "My model is 95% accurate!" (when 95% of data is one class)
            ✅ **Fix**: Use precision, recall, F1-score, or AUC-ROC
            
            **Mistake 4: Not checking for multicollinearity** 🚫
            ❌ Including highly correlated features that confuse the model
            ✅ **Fix**: Calculate correlation matrix, remove redundant features
            
            **Mistake 5: Wrong interpretation of coefficients** 🚫
            ❌ "This coefficient is bigger, so this feature is more important"
            ✅ **Fix**: Only compare coefficients after feature scaling
            
            **Mistake 6: Forgetting to validate assumptions** 🚫
            ❌ Assuming logistic regression will work without checking linearity
            ✅ **Fix**: Plot logit vs features, check for linear relationship
            """,
            
            # 15. Comparison with similar algorithms
            'comparisons': """
            🔹 **Logistic Regression vs Similar Algorithms**
            
            **Logistic Regression vs Linear Regression:**
            • **Logistic**: Classification, outputs probabilities, sigmoid function
            • **Linear**: Regression, outputs continuous values, straight line
            • **Use Logistic**: When predicting categories (yes/no, spam/not spam)
            
            **Logistic Regression vs Decision Trees:**
            • **Logistic**: Linear boundaries, probabilistic, fast
            • **Decision Trees**: Non-linear boundaries, rule-based, interpretable
            • **Use Decision Trees**: When relationships are non-linear or rule-based
            
            **Logistic Regression vs Naive Bayes:**
            • **Logistic**: Discriminative model, learns decision boundary
            • **Naive Bayes**: Generative model, learns class distributions
            • **Use Naive Bayes**: With small datasets or when features are independent
            
            **Logistic Regression vs SVM:**
            • **Logistic**: Probabilistic output, all data affects model
            • **SVM**: Margin-based, only support vectors matter
            • **Use SVM**: With high-dimensional data or when margin is important
            
            **Logistic Regression vs Random Forest:**
            • **Logistic**: Fast, simple, linear boundaries
            • **Random Forest**: Slower, complex, non-linear boundaries
            • **Use Random Forest**: When you have complex feature interactions
            """,
            
            # 16. Real-world applications
            'real_world_applications': """
            🔹 **Real-World Applications & Industry Use Cases**
            
            **📧 Technology & Communications:**
            • **Email Spam Detection**: Gmail, Outlook spam filters
            • **Click-Through Prediction**: Google Ads, Facebook advertising
            • **Fraud Detection**: Credit card transaction monitoring
            • **Sentiment Analysis**: Social media monitoring, review classification
            
            **🏥 Healthcare & Medicine:**
            • **Disease Diagnosis**: Cancer screening, diabetes prediction
            • **Clinical Trials**: Treatment success probability
            • **Drug Discovery**: Compound effectiveness prediction
            • **Emergency Triage**: Priority assignment in hospitals
            
            **📈 Finance & Banking:**
            • **Credit Scoring**: Loan approval decisions
            • **Risk Assessment**: Investment risk classification
            • **Algorithmic Trading**: Buy/sell signal generation
            • **Insurance Claims**: Fraud detection, claim approval
            
            **🛒 E-commerce & Marketing:**
            • **Customer Churn**: Will customer cancel subscription?
            • **Purchase Prediction**: Likelihood to buy specific product
            • **A/B Testing**: Which version performs better?
            • **Recommendation Systems**: Interest classification
            
            **🏭 Manufacturing & Operations:**
            • **Quality Control**: Defect detection in products
            • **Predictive Maintenance**: Machine failure prediction
            • **Supply Chain**: Demand forecasting (high/low)
            • **Safety Monitoring**: Accident risk assessment
            
            **🎓 Education & Research:**
            • **Student Success**: Graduation probability prediction
            • **Course Recommendation**: Student interest classification
            • **Research Analysis**: Hypothesis testing in studies
            • **Admissions**: Application approval likelihood
            
            **🚗 Transportation & Logistics:**
            • **Route Optimization**: Traffic condition classification
            • **Delivery Prediction**: On-time delivery probability
            • **Vehicle Maintenance**: Service need prediction
            • **Driver Safety**: Risk assessment
            
            **💡 Key Success Factors:**
            • Clean, well-preprocessed data
            • Adequate sample size (100+ examples per feature)
            • Balanced or properly weighted classes
            • Linear separability of classes
            • Regular model retraining with new data
            """
        }
    
    def generate_sample_data(self, n_samples=100, n_features=2, n_clusters_per_class=1):
        """Generate sample classification data."""
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_redundant=0,
            n_informative=n_features,
            n_clusters_per_class=n_clusters_per_class,
            random_state=42
        )
        return X, y
    
    def fit(self, X, y):
        """Fit the logistic regression model."""
        self.model = SklearnLogisticRegression(max_iter=self.max_iter, random_state=42)
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Make predictions using the fitted model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict_proba(X)
    
    def get_metrics(self, X, y):
        """Calculate and return classification metrics."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calculating metrics")
            
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]  # Probability of positive class
        
        metrics = {
            'Accuracy': accuracy_score(y, y_pred),
            'Precision': precision_score(y, y_pred, average='binary'),
            'Recall': recall_score(y, y_pred, average='binary'),
            'F1-Score': f1_score(y, y_pred, average='binary'),
            'Log Loss': log_loss(y, y_proba),
            'AUC-ROC': auc(*roc_curve(y, y_proba)[:2])
        }
        
        return metrics
    
    def plot_sigmoid_function(self):
        """Plot the sigmoid function."""
        x = np.linspace(-6, 6, 100)
        y = 1 / (1 + np.exp(-x))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, y, 'b-', linewidth=2, label='Sigmoid Function')
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Decision Boundary (p=0.5)')
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Linear Combination (z = β₀ + β₁x₁ + ... + βₙxₙ)')
        ax.set_ylabel('Probability')
        ax.set_title('Sigmoid (Logistic) Function')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add annotations
        ax.annotate('Class 0\n(p < 0.5)', xy=(-3, 0.25), fontsize=12, ha='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
        ax.annotate('Class 1\n(p > 0.5)', xy=(3, 0.75), fontsize=12, ha='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        return fig
    
    def plot_decision_boundary(self, X, y, title="Logistic Regression Decision Boundary"):
        """Plot decision boundary for 2D data."""
        if X.shape[1] != 2:
            raise ValueError("Decision boundary plot only available for 2D data")
            
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
        Z = self.model.predict_proba(mesh_points)[:, 1]  # Probability of class 1
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary and probability contours
        contour = ax.contourf(xx, yy, Z, levels=50, alpha=0.6, cmap=plt.cm.RdYlBu)
        ax.contour(xx, yy, Z, levels=[0.5], colors='black', linestyles='--', linewidths=2)
        
        # Plot data points
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
        
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title(title)
        
        # Add colorbar
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Probability of Class 1')
        
        # Add legend for data points
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor='red', markersize=10, label='Class 0'),
                          plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor='blue', markersize=10, label='Class 1')]
        ax.legend(handles=legend_elements)
        
        return fig
    
    def plot_classification_results(self, X, y):
        """Create comprehensive classification visualization."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting")
            
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Confusion Matrix
        cm = confusion_matrix(y, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        axes[0, 0].set_title('Confusion Matrix')
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y, y_proba)
        roc_auc = auc(fpr, tpr)
        axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, 
                       label=f'ROC Curve (AUC = {roc_auc:.2f})')
        axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Predicted Probabilities Distribution
        axes[1, 0].hist(y_proba[y == 0], bins=30, alpha=0.5, label='Class 0', color='red')
        axes[1, 0].hist(y_proba[y == 1], bins=30, alpha=0.5, label='Class 1', color='blue')
        axes[1, 0].axvline(x=0.5, color='black', linestyle='--', 
                          label='Decision Threshold')
        axes[1, 0].set_xlabel('Predicted Probability')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Predicted Probabilities')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Feature Importance
        if hasattr(self.model, 'coef_'):
            feature_names = [f'Feature {i+1}' for i in range(X.shape[1])]
            coefficients = self.model.coef_[0]
            
            bars = axes[1, 1].bar(feature_names, coefficients, alpha=0.7, 
                                color='skyblue', edgecolor='black')
            axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.8)
            axes[1, 1].set_xlabel('Features')
            axes[1, 1].set_ylabel('Coefficient Value')
            axes[1, 1].set_title('Feature Coefficients')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, coef in zip(bars, coefficients):
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                               f'{coef:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def streamlit_interface(self):
        """Create comprehensive Streamlit interface for Logistic Regression."""
        st.subheader("🎯 Logistic Regression")
        
        theory = self.get_theory()
        
        # Main tabs for comprehensive coverage
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🎯 Overview", "📚 Deep Dive", "💻 Implementation", 
            "🧪 Interactive Demo", "❓ Q&A", "🏢 Applications"
        ])
        
        with tab1:
            # Overview Tab - Essential Information
            st.markdown("### 🎯 What is Logistic Regression?")
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
            
            # Sigmoid function visualization
            st.markdown("### 📈 Sigmoid Function")
            fig_sigmoid = self.plot_sigmoid_function()
            st.pyplot(fig_sigmoid)
            plt.close()
        
        with tab4:
            # Interactive Demo Tab
            st.markdown("### 🧪 Try Logistic Regression Yourself!")
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
        col1, col2, col3 = st.columns(3)
        
        with col1:
            n_samples = st.slider("Number of samples:", 50, 500, 200)
            
        with col2:
            n_features = st.slider("Number of features:", 2, 5, 2)
            
        with col3:
            n_clusters = st.slider("Clusters per class:", 1, 3, 1)
        
        # Generate and split data
        X, y = self.generate_sample_data(n_samples, n_features, n_clusters)
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
        
        # Visualizations
        st.markdown("### 📈 Visualizations")
        
        # Decision boundary (only for 2D data)
        if n_features == 2:
            fig_boundary = self.plot_decision_boundary(X_test, y_test, 
                                                     "Logistic Regression - Decision Boundary")
            st.pyplot(fig_boundary)
            plt.close()
        
        # Classification results
        fig_results = self.plot_classification_results(X_test, y_test)
        st.pyplot(fig_results)
        plt.close()
        
        # Interpretation
        st.markdown("### 🔍 Interpretation")
        
        accuracy = test_metrics['Accuracy']
        auc_score = test_metrics['AUC-ROC']
        
        col1, col2 = st.columns(2)
        
        with col1:
            if accuracy > 0.9:
                st.success(f"**Excellent performance!** Accuracy: {accuracy:.1%}")
            elif accuracy > 0.8:
                st.info(f"**Good performance.** Accuracy: {accuracy:.1%}")
            elif accuracy > 0.7:
                st.warning(f"**Moderate performance.** Accuracy: {accuracy:.1%}")
            else:
                st.error(f"**Poor performance.** Accuracy: {accuracy:.1%}")
                
        with col2:
            if auc_score > 0.9:
                st.success(f"**Excellent discrimination!** AUC: {auc_score:.3f}")
            elif auc_score > 0.8:
                st.info(f"**Good discrimination.** AUC: {auc_score:.3f}")
            elif auc_score > 0.7:
                st.warning(f"**Moderate discrimination.** AUC: {auc_score:.3f}")
            else:
                st.error(f"**Poor discrimination.** AUC: {auc_score:.3f}")
        
        # Classification report
        y_pred_test = self.predict(X_test)
        report = classification_report(y_test, y_pred_test, output_dict=True)
        
        st.markdown("**Detailed Classification Report:**")
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.round(4))


def main():
    """Main function for testing Logistic Regression."""
    lr = LogisticRegression()
    lr.streamlit_interface()


if __name__ == "__main__":
    main()