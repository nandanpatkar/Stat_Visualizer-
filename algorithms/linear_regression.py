"""
Linear Regression Algorithm Implementation

Linear regression is a fundamental supervised learning algorithm used for 
predicting continuous target variables based on linear relationships between 
features and the target.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import streamlit as st


class LinearRegression:
    """
    Linear Regression implementation with educational explanations.
    
    Linear regression finds the best-fitting straight line through data points
    by minimizing the sum of squared residuals.
    
    Mathematical Formula: y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
    
    Where:
    - y: target variable (dependent variable)
    - β₀: intercept (y-intercept)
    - β₁, β₂, ..., βₙ: coefficients (slopes)
    - x₁, x₂, ..., xₙ: features (independent variables)
    - ε: error term
    """
    
    def __init__(self):
        self.model = None
        self.is_fitted = False
        
    @staticmethod
    def get_theory():
        """Return comprehensive theoretical explanation of Linear Regression."""
        return {
            'name': 'Linear Regression',
            'type': 'Supervised Learning - Regression',
            
            # 1. What the algorithm is
            'definition': """
            🔹 **What is Linear Regression?**
            Linear Regression is like drawing the "best-fit line" through a scatter plot of data points.
            It predicts a continuous target value by finding a straight line relationship between 
            input features and the output.
            """,
            
            # 2. Why the algorithm is used
            'motivation': """
            🔹 **Why Use Linear Regression?**
            • 📈 **Prediction**: Predict house prices, stock prices, sales revenue
            • 📊 **Understanding**: See which factors most affect the outcome
            • ⚡ **Speed**: Very fast to train and make predictions
            • 🎯 **Baseline**: Good starting point before trying complex algorithms
            • 📉 **Trends**: Identify trends and relationships in data
            """,
            
            # 3. Intuition with real-life analogy
            'intuition': """
            🔹 **Real-Life Analogy: The Pizza Delivery Problem**
            
            Imagine you're a pizza delivery manager trying to predict delivery time:
            
            🍕 **The Problem**: How long will delivery take?
            🎯 **Input factors**: Distance (km), traffic level (1-10), weather (1-10)
            
            **Linear Regression is like having an experienced driver who says:**
            "Delivery time = 15 minutes base time + 2 minutes per km + 1 minute per traffic point + 0.5 minutes per weather point"
            
            📐 **The line**: This formula is your "line" through all past delivery data
            🎯 **New prediction**: For a new delivery (5km, traffic=6, weather=3), 
            Time = 15 + 2×5 + 1×6 + 0.5×3 = 32.5 minutes
            """,
            
            # 4. Mathematical foundation
            'math_foundation': """
            🔹 **Mathematical Foundation (Step-by-Step)**
            
            **Core Formula:**
            ```
            y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
            ```
            
            **Where:**
            • `y` = Target variable (what we want to predict)
            • `β₀` = Intercept (y-value when all x's are 0)
            • `β₁, β₂, ...` = Coefficients (slopes for each feature)
            • `x₁, x₂, ...` = Input features
            • `ε` = Error term (what the model can't explain)
            
            **Matrix Form:**
            ```
            Y = X × β + ε
            ```
            
            **Finding Best Coefficients (Normal Equation):**
            ```
            β = (X^T × X)^(-1) × X^T × Y
            ```
            
            **Cost Function (Mean Squared Error):**
            ```
            MSE = (1/n) × Σ(actual - predicted)²
            ```
            """,
            
            # 5. Step-by-step working
            'algorithm_steps': """
            🔹 **How Linear Regression Works (Step-by-Step)**
            
            **Step 1: Data Preparation** 📋
            • Collect training data with input features (X) and target values (y)
            • Clean data (handle missing values, outliers)
            • Split into training and testing sets
            
            **Step 2: Initialize** 🎲
            • Start with random coefficient values (β₀, β₁, β₂, ...)
            • Or use the normal equation for direct calculation
            
            **Step 3: Make Predictions** 🎯
            • For each data point: predicted_y = β₀ + β₁×x₁ + β₂×x₂ + ...
            • Calculate predictions for all training data
            
            **Step 4: Calculate Error** 📏
            • Find difference between actual and predicted values
            • Square the differences to avoid negative cancellation
            • Average all squared errors (Mean Squared Error)
            
            **Step 5: Optimize** 🔧
            • Adjust coefficients to minimize error
            • Use gradient descent or normal equation
            • Repeat until error stops improving
            
            **Step 6: Test** ✅
            • Use final model on test data
            • Check performance metrics (R², RMSE, MAE)
            """,
            
            # 6. Pseudocode
            'pseudocode': """
            🔹 **Pseudocode (Easy to Understand)**
            
            ```
            ALGORITHM: Linear Regression
            
            INPUT: 
                - X: matrix of features (n_samples × n_features)
                - y: target values (n_samples × 1)
            
            OUTPUT:
                - coefficients: β₀, β₁, β₂, ..., βₙ
            
            BEGIN
                1. ADD intercept column of 1s to X → X_with_intercept
                
                2. CALCULATE coefficients using Normal Equation:
                   coefficients = (X^T × X)^(-1) × X^T × y
                
                3. FOR each new prediction:
                   prediction = coefficient[0] + coefficient[1]*x1 + ... + coefficient[n]*xn
                
                4. RETURN coefficients and prediction function
            END
            ```
            
            **Alternative (Gradient Descent):**
            ```
            BEGIN
                1. INITIALIZE coefficients randomly
                2. SET learning_rate = 0.01
                3. REPEAT until convergence:
                   a. CALCULATE predictions = X × coefficients
                   b. CALCULATE error = predictions - actual
                   c. CALCULATE gradient = X^T × error / n
                   d. UPDATE coefficients = coefficients - learning_rate × gradient
            END
            ```
            """,
            
            # 7. Python implementation
            'python_implementation': """
            🔹 **Python Implementation**
            
            **From Scratch:**
            ```python
            import numpy as np
            
            class LinearRegressionFromScratch:
                def __init__(self):
                    self.coefficients = None
                    self.intercept = None
                
                def fit(self, X, y):
                    # Add intercept column
                    X_with_intercept = np.column_stack([np.ones(len(X)), X])
                    
                    # Normal equation: β = (X^T × X)^(-1) × X^T × y
                    XtX = X_with_intercept.T @ X_with_intercept
                    XtX_inv = np.linalg.inv(XtX)
                    Xty = X_with_intercept.T @ y
                    
                    coeffs = XtX_inv @ Xty
                    
                    self.intercept = coeffs[0]
                    self.coefficients = coeffs[1:]
                
                def predict(self, X):
                    return self.intercept + X @ self.coefficients
            
            # Example usage
            X = np.array([[1], [2], [3], [4], [5]])  # Features
            y = np.array([2, 4, 6, 8, 10])           # Target
            
            model = LinearRegressionFromScratch()
            model.fit(X, y)
            predictions = model.predict(X)
            ```
            
            **Using Scikit-learn:**
            ```python
            from sklearn.linear_model import LinearRegression
            
            # Create and train model
            model = LinearRegression()
            model.fit(X, y)
            
            # Make predictions
            predictions = model.predict(X)
            
            # Get coefficients
            print(f"Intercept: {model.intercept_}")
            print(f"Coefficients: {model.coef_}")
            ```
            """,
            
            # 8. Example with sample input/output
            'example': """
            🔹 **Complete Example: Predicting House Prices**
            
            **Input Data:**
            ```
            House Size (sqft) | Bedrooms | Age (years) | Price ($1000s)
            1000              | 2        | 5           | 150
            1500              | 3        | 10          | 200
            2000              | 4        | 2           | 300
            2500              | 4        | 15          | 350
            ```
            
            **Step-by-Step Calculation:**
            ```
            1. Features: X = [[1000, 2, 5], [1500, 3, 10], [2000, 4, 2], [2500, 4, 15]]
            2. Target: y = [150, 200, 300, 350]
            
            3. After training, we get:
               - Intercept (β₀) = 50
               - Coefficients: β₁=0.1 (size), β₂=20 (bedrooms), β₃=-2 (age)
            
            4. Formula: Price = 50 + 0.1×Size + 20×Bedrooms - 2×Age
            ```
            
            **New Prediction:**
            ```
            New house: 1800 sqft, 3 bedrooms, 8 years old
            Predicted Price = 50 + 0.1×1800 + 20×3 - 2×8
                           = 50 + 180 + 60 - 16
                           = $274,000
            ```
            """,
            
            # 9. Visualization explanation
            'visualization': """
            🔹 **Understanding Through Visualizations**
            
            **1D Case (One Feature):**
            📊 Scatter plot with best-fit line
            • X-axis: Feature (e.g., house size)
            • Y-axis: Target (e.g., price)
            • Line: Shows relationship y = mx + b
            
            **2D Case (Two Features):**
            📈 3D plot with best-fit plane
            • X-axis: Feature 1, Y-axis: Feature 2, Z-axis: Target
            • Plane: Shows relationship z = ax + by + c
            
            **Residual Plots:**
            📉 Show prediction errors
            • Good model: Random scatter around zero
            • Bad model: Clear patterns in residuals
            
            **Feature Importance:**
            📊 Bar chart of coefficients
            • Taller bars = more important features
            • Sign shows positive/negative relationship
            """,
            
            # 10. Time and space complexity
            'complexity': """
            🔹 **Time & Space Complexity**
            
            **Time Complexity:**
            • **Training**: O(n × p²) where n=samples, p=features
            • **Prediction**: O(p) per prediction
            • **Normal Equation**: O(p³) for matrix inversion
            • **Gradient Descent**: O(n × p × iterations)
            
            **Space Complexity:**
            • **Model Storage**: O(p) - just store coefficients
            • **Training Memory**: O(n × p) - store dataset
            • **Prediction Memory**: O(1) per prediction
            
            **Scalability:**
            • ✅ **Large n (samples)**: Handles millions of data points
            • ⚠️ **Large p (features)**: Slows down with thousands of features
            • ✅ **Streaming**: Can update model incrementally
            """,
            
            # 11. Advantages and disadvantages
            'pros_cons': """
            🔹 **Advantages** ✅
            • **Simple**: Easy to understand and explain
            • **Fast**: Quick training and prediction
            • **Interpretable**: Coefficients show feature importance
            • **No hyperparameters**: Works out of the box
            • **Baseline**: Good starting point for any regression problem
            • **Probabilistic**: Can provide prediction intervals
            • **Memory efficient**: Only stores p coefficients
            
            🔹 **Disadvantages** ❌
            • **Linear only**: Can't capture complex relationships
            • **Outlier sensitive**: One bad point affects whole model
            • **Feature scaling**: Performance depends on feature scales
            • **Multicollinearity**: Problems when features are correlated
            • **Overfitting**: Can memorize noise with many features
            • **Assumptions**: Requires several statistical assumptions
            """,
            
            # 12. When to use and when NOT to use
            'usage_guide': """
            🔹 **When TO Use Linear Regression** ✅
            
            **Perfect for:**
            • 📊 **Relationship exploration**: Understanding feature impact
            • ⚡ **Quick baselines**: Fast first model for regression
            • 🎯 **Simple relationships**: When relationship is roughly linear
            • 📈 **Trend analysis**: Identifying trends over time
            • 🔍 **Feature selection**: Finding important variables
            • 📚 **Educational**: Learning ML fundamentals
            
            **Good when:**
            • Dataset is small to medium (< 100K samples)
            • Number of features is small (< 100)
            • Need interpretable results
            • Real-time predictions required
            
            🔹 **When NOT to Use Linear Regression** ❌
            
            **Avoid when:**
            • 🌀 **Non-linear relationships**: Curved, exponential patterns
            • 🎯 **Classification**: Predicting categories (use logistic regression)
            • 📊 **Complex interactions**: Features interact in complex ways
            • 🎲 **High noise**: Too much randomness in data
            • 🔢 **Many features**: More features than samples
            • 📈 **Non-constant variance**: Errors change with prediction value
            
            **Use instead:**
            • Polynomial Regression (for curves)
            • Random Forest (for interactions)
            • Neural Networks (for complex patterns)
            • Regularized models (for many features)
            """,
            
            # 13. Common interview questions
            'interview_questions': """
            🔹 **Common Interview Questions & Answers**
            
            **Q1: What's the difference between correlation and causation in linear regression?**
            A: Correlation shows relationship strength, causation shows cause-effect. Linear regression finds correlation but doesn't prove causation. Just because X predicts Y doesn't mean X causes Y.
            
            **Q2: How do you handle multicollinearity?**
            A: 
            • Remove highly correlated features
            • Use regularization (Ridge/Lasso)
            • Principal Component Analysis (PCA)
            • Variance Inflation Factor (VIF) to detect
            
            **Q3: What are the assumptions of linear regression?**
            A: LINNE
            • **L**inearity: Relationship is linear
            • **I**ndependence: Observations are independent
            • **N**ormality: Residuals are normally distributed
            • **N**o multicollinearity: Features aren't highly correlated
            • **E**qual variance: Constant variance of residuals
            
            **Q4: How to evaluate a linear regression model?**
            A:
            • R² Score: Proportion of variance explained (higher better)
            • RMSE: Root Mean Square Error (lower better)
            • MAE: Mean Absolute Error (lower better)
            • Residual plots: Check for patterns
            
            **Q5: Normal equation vs Gradient descent?**
            A:
            • Normal equation: Direct calculation, O(p³), exact solution
            • Gradient descent: Iterative, O(n×p×iter), handles large data
            """,
            
            # 14. Common mistakes
            'common_mistakes': """
            🔹 **Common Beginner Mistakes & How to Avoid**
            
            **Mistake 1: Not checking linearity** 🚫
            ❌ Assuming relationship is linear without checking
            ✅ **Fix**: Plot features vs target, look for straight lines
            
            **Mistake 2: Forgetting feature scaling** 🚫
            ❌ Using features with different scales (age vs income)
            ✅ **Fix**: Standardize features (mean=0, std=1)
            
            **Mistake 3: Ignoring outliers** 🚫
            ❌ Including extreme values that skew the model
            ✅ **Fix**: Use box plots, remove or cap outliers
            
            **Mistake 4: Overfitting with many features** 🚫
            ❌ Including too many features for small datasets
            ✅ **Fix**: Feature selection, regularization, more data
            
            **Mistake 5: Not validating assumptions** 🚫
            ❌ Skipping residual analysis and assumption checking
            ✅ **Fix**: Plot residuals, test normality, check variance
            
            **Mistake 6: Wrong interpretation** 🚫
            ❌ "Feature X causes Y because coefficient is large"
            ✅ **Fix**: "Feature X is associated with Y" (correlation ≠ causation)
            """,
            
            # 15. Comparison with similar algorithms
            'comparisons': """
            🔹 **Linear Regression vs Similar Algorithms**
            
            **Linear vs Polynomial Regression:**
            • **Linear**: y = mx + b (straight line)
            • **Polynomial**: y = ax² + bx + c (curves)
            • **Use polynomial**: When scatter plot shows curves
            
            **Linear vs Ridge/Lasso Regression:**
            • **Linear**: No penalty, can overfit
            • **Ridge**: L2 penalty, shrinks coefficients
            • **Lasso**: L1 penalty, can make coefficients zero
            • **Use regularized**: When you have many features
            
            **Linear vs Random Forest:**
            • **Linear**: Assumes linear relationship, interpretable
            • **Random Forest**: Captures non-linear relationships
            • **Use Random Forest**: When relationships are complex
            
            **Regression vs Classification:**
            • **Linear Regression**: Predicts continuous values
            • **Logistic Regression**: Predicts probabilities/categories
            • **Use logistic**: When target is yes/no, categories
            """,
            
            # 16. Real-world applications
            'real_world_applications': """
            🔹 **Real-World Applications & Industry Use Cases**
            
            **🏠 Real Estate:**
            • Predicting house prices based on size, location, age
            • Rental price estimation for properties
            • Market trend analysis
            
            **📈 Finance:**
            • Stock price prediction using economic indicators
            • Credit scoring and loan default prediction
            • Portfolio risk assessment
            • Algorithmic trading strategies
            
            **🛒 E-commerce:**
            • Sales forecasting based on marketing spend
            • Customer lifetime value prediction
            • Demand planning and inventory management
            • Pricing optimization
            
            **🏥 Healthcare:**
            • Drug dosage calculation based on patient factors
            • Medical cost prediction
            • Disease progression modeling
            • Clinical trial analysis
            
            **🏭 Manufacturing:**
            • Quality control and defect prediction
            • Energy consumption optimization
            • Predictive maintenance scheduling
            • Supply chain optimization
            
            **📱 Technology:**
            • User engagement prediction
            • Performance monitoring and capacity planning
            • A/B testing result analysis
            • Resource allocation optimization
            
            **🎓 Education:**
            • Student performance prediction
            • Course completion rate forecasting
            • Educational resource optimization
            
            **💡 Key Success Factors:**
            • Clean, relevant data
            • Good feature engineering
            • Regular model updates
            • Domain expertise integration
            """
        }
    
    def generate_sample_data(self, n_samples=100, n_features=1, noise=0.1):
        """Generate sample data for demonstration."""
        np.random.seed(42)
        
        if n_features == 1:
            # Simple linear relationship
            X = np.random.uniform(0, 10, (n_samples, 1))
            y = 2.5 * X.flatten() + 1.5 + np.random.normal(0, noise, n_samples)
        else:
            # Multiple features
            X = np.random.randn(n_samples, n_features)
            # Create linear relationship with different coefficients
            true_coefficients = np.random.uniform(-3, 3, n_features)
            y = X @ true_coefficients + np.random.normal(0, noise, n_samples)
            
        return X, y
    
    def fit(self, X, y):
        """Fit the linear regression model."""
        self.model = SklearnLinearRegression()
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Make predictions using the fitted model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)
    
    def get_metrics(self, X, y):
        """Calculate and return model performance metrics."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calculating metrics")
            
        y_pred = self.predict(X)
        
        metrics = {
            'Mean Squared Error (MSE)': mean_squared_error(y, y_pred),
            'Root Mean Squared Error (RMSE)': np.sqrt(mean_squared_error(y, y_pred)),
            'Mean Absolute Error (MAE)': mean_absolute_error(y, y_pred),
            'R² Score': r2_score(y, y_pred),
            'Adjusted R²': self._calculate_adjusted_r2(y, y_pred, X.shape[1])
        }
        
        return metrics
    
    def _calculate_adjusted_r2(self, y_true, y_pred, n_features):
        """Calculate adjusted R² score."""
        n = len(y_true)
        r2 = r2_score(y_true, y_pred)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
        return adj_r2
    
    def plot_results(self, X, y, title="Linear Regression Results"):
        """Create visualizations for linear regression results."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting")
            
        y_pred = self.predict(X)
        
        if X.shape[1] == 1:
            # Single feature - scatter plot with regression line
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Scatter plot with regression line
            axes[0, 0].scatter(X.flatten(), y, alpha=0.6, color='blue', label='Actual')
            
            # Sort for smooth line
            sort_idx = np.argsort(X.flatten())
            axes[0, 0].plot(X[sort_idx].flatten(), y_pred[sort_idx], 
                          color='red', linewidth=2, label='Predicted')
            
            axes[0, 0].set_xlabel('Feature')
            axes[0, 0].set_ylabel('Target')
            axes[0, 0].set_title('Actual vs Predicted')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
        else:
            # Multiple features - predicted vs actual
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes[0, 0].scatter(y, y_pred, alpha=0.6, color='blue')
            axes[0, 0].plot([y.min(), y.max()], [y.min(), y.max()], 
                          'r--', linewidth=2, label='Perfect Prediction')
            axes[0, 0].set_xlabel('Actual')
            axes[0, 0].set_ylabel('Predicted')
            axes[0, 0].set_title('Actual vs Predicted')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Residual plot
        residuals = y - y_pred
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
        
        # Q-Q plot for normality check
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot (Normality Check)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self):
        """Plot feature coefficients (importance)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting feature importance")
            
        coefficients = self.model.coef_
        intercept = self.model.intercept_
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot coefficients
        feature_names = [f'Feature {i+1}' for i in range(len(coefficients))]
        bars = ax.bar(feature_names, coefficients, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Add value labels on bars
        for bar, coef in zip(bars, coefficients):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{coef:.3f}', ha='center', va='bottom')
        
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.8)
        ax.set_xlabel('Features')
        ax.set_ylabel('Coefficient Value')
        ax.set_title(f'Feature Coefficients (Intercept: {intercept:.3f})')
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig
    
    def streamlit_interface(self):
        """Create comprehensive Streamlit interface for Linear Regression."""
        st.subheader("🔗 Linear Regression")
        
        theory = self.get_theory()
        
        # Main tabs for comprehensive coverage
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🎯 Overview", "📚 Deep Dive", "💻 Implementation", 
            "🧪 Interactive Demo", "❓ Q&A", "🏢 Applications"
        ])
        
        with tab1:
            # Overview Tab - Essential Information
            st.markdown("### 🎯 What is Linear Regression?")
            st.markdown(theory['definition'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 🌟 Why Use It?")
                st.markdown(theory['motivation'])
                
            with col2:
                st.markdown("### 🍕 Simple Analogy")
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
            # Interactive Demo Tab - This will contain the actual working demo
            st.markdown("### 🧪 Try Linear Regression Yourself!")
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
        """Create the interactive demo section (previous implementation)."""
        # Parameters section
        st.markdown("### 🔧 Parameters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            n_samples = st.slider("Number of samples:", 50, 500, 100)
            
        with col2:
            n_features = st.slider("Number of features:", 1, 5, 1)
            
        with col3:
            noise_level = st.slider("Noise level:", 0.1, 2.0, 0.5, 0.1)
        
        # Generate and split data
        X, y = self.generate_sample_data(n_samples, n_features, noise_level)
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
        
        # Main results plot
        fig_results = self.plot_results(X_test, y_test, "Linear Regression - Test Set")
        st.pyplot(fig_results)
        plt.close()
        
        # Feature importance plot
        fig_importance = self.plot_feature_importance()
        st.pyplot(fig_importance)
        plt.close()
        
        # Interpretation
        st.markdown("### 🔍 Interpretation")
        
        r2_score_val = test_metrics['R² Score']
        if r2_score_val > 0.8:
            st.success(f"**Excellent fit!** The model explains {r2_score_val:.1%} of the variance in the data.")
        elif r2_score_val > 0.6:
            st.info(f"**Good fit.** The model explains {r2_score_val:.1%} of the variance in the data.")
        elif r2_score_val > 0.3:
            st.warning(f"**Moderate fit.** The model explains {r2_score_val:.1%} of the variance in the data.")
        else:
            st.error(f"**Poor fit.** The model only explains {r2_score_val:.1%} of the variance in the data.")
        
        # Model equation
        if n_features == 1:
            coef = self.model.coef_[0]
            intercept = self.model.intercept_
            st.write(f"**Model Equation:** y = {intercept:.3f} + {coef:.3f} × x")
        else:
            equation_parts = [f"{self.model.intercept_:.3f}"]
            for i, coef in enumerate(self.model.coef_):
                equation_parts.append(f"{coef:+.3f} × x{i+1}")
            equation = "y = " + " ".join(equation_parts)
            st.write(f"**Model Equation:** {equation}")


def main():
    """Main function for testing Linear Regression."""
    lr = LinearRegression()
    lr.streamlit_interface()


if __name__ == "__main__":
    main()