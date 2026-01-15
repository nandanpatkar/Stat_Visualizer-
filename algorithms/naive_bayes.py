"""
Naive Bayes Algorithm Implementation

Naive Bayes is a family of simple probabilistic classifiers based on Bayes' theorem
with strong independence assumptions between features.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report)
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import streamlit as st


class NaiveBayes:
    """
    Naive Bayes implementation with educational explanations.
    
    Based on Bayes' theorem: P(class|features) = P(features|class) * P(class) / P(features)
    
    Assumes feature independence (hence "naive").
    """
    
    def __init__(self, nb_type='gaussian'):
        self.nb_type = nb_type
        self.model = None
        self.is_fitted = False
        
    @staticmethod
    def get_theory():
        """Return comprehensive theoretical explanation of Naive Bayes."""
        return {
            'name': 'Naive Bayes',
            'type': 'Supervised Learning - Classification',
            
            # 1. What the algorithm is
            'definition': """
            🔹 **What is Naive Bayes?**
            Naive Bayes is like a detective who solves cases by calculating the probability of 
            different suspects based on evidence. It's "naive" because it assumes all evidence 
            pieces are independent (which is rarely true), but surprisingly, it still works well!
            """,
            
            # 2. Why the algorithm is used
            'motivation': """
            🔹 **Why Use Naive Bayes?**
            • 📧 **Spam Detection**: Classic use case for email filtering
            • 📝 **Text Classification**: News categorization, sentiment analysis
            • 🚀 **Real-time Systems**: Extremely fast training and prediction
            • 🏥 **Medical Diagnosis**: Disease prediction from symptoms
            • 👥 **Customer Segmentation**: User behavior classification
            • 📱 **Recommendation Systems**: Content filtering and suggestions
            """,
            
            # 3. Intuition with real-life analogy
            'intuition': """
            🔹 **Real-Life Analogy: The Medical Diagnosis Detective**
            
            Imagine you're a doctor trying to diagnose a patient:
            
            🏥 **The Problem**: Patient has fever, cough, and headache → What disease?
            🤔 **Evidence**: Symptoms are your "features"
            
            **Naive Bayes is like an experienced doctor who:**
            
            **Step 1**: 📊 Knows disease probabilities from past cases
            - Flu: 40% of patients
            - Cold: 35% of patients  
            - COVID: 25% of patients
            
            **Step 2**: 🌡️ Calculates symptom probabilities FOR EACH disease
            - P(Fever | Flu) = 90%
            - P(Fever | Cold) = 30%
            - P(Fever | COVID) = 85%
            
            **Step 3**: 😷 Does the same for ALL symptoms
            - P(Cough | Flu), P(Cough | Cold), P(Cough | COVID)
            - P(Headache | Flu), P(Headache | Cold), P(Headache | COVID)
            
            **Step 4**: 🧮 The "Naive" Assumption
            Assumes symptoms are independent:
            P(Fever, Cough, Headache | Flu) = P(Fever|Flu) × P(Cough|Flu) × P(Headache|Flu)
            
            **Step 5**: 📊 Bayes' Theorem Magic
            For each disease:
            P(Disease | Symptoms) = P(Symptoms | Disease) × P(Disease) / P(Symptoms)
            
            **Step 6**: 🎯 Final Diagnosis
            Choose disease with highest probability!
            
            **Why "Naive"?**
            • Assumes fever doesn't affect probability of cough (often false)
            • Real symptoms often correlate (fever + cough often together)
            • But still works amazingly well in practice!
            
            🎯 **In data terms**: 
            - Diseases = Classes
            - Symptoms = Features
            - Past cases = Training Data
            - New patient = Test Sample
            """,
            
            # 4. Mathematical foundation
            'math_foundation': """
            🔹 **Mathematical Foundation (Step-by-Step)**
            
            **Bayes' Theorem (Foundation):**
            ```
            P(Class | Features) = P(Features | Class) × P(Class) / P(Features)
            ```
            
            **Where:**
            • `P(Class | Features)` = Posterior probability (what we want)
            • `P(Features | Class)` = Likelihood (how likely features given class)
            • `P(Class)` = Prior probability (how common is this class)
            • `P(Features)` = Evidence (normalization constant)
            
            **Naive Independence Assumption:**
            ```
            P(x₁, x₂, ..., xₙ | Class) = ∏ᵢ₌₁ⁿ P(xᵢ | Class)
            ```
            This means: P(fever, cough | flu) = P(fever | flu) × P(cough | flu)
            
            **Classification Decision:**
            ```
            Predicted_Class = argmaxᴄ P(Class = c | x₁, x₂, ..., xₙ)
                            = argmaxᴄ P(Class = c) × ∏ᵢ₌₁ⁿ P(xᵢ | Class = c)
            ```
            
            **Different Naive Bayes Types:**
            
            **1. Gaussian Naive Bayes (Continuous features):**
            ```
            P(xᵢ | Class = c) = (1/√(2πσᴄ²)) × exp(-((xᵢ - μᴄ)²) / (2σᴄ²))
            ```
            Where μᴄ and σᴄ are mean and std deviation for class c
            
            **2. Multinomial Naive Bayes (Count features):**
            ```
            P(xᵢ | Class = c) = (θᴄᵢ)ᵡ₁ × ... × (θᴄᵢ)ᵡₙ / (n!/(n₁!...nₙ!))
            ```
            Where θᴄᵢ is probability of feature i in class c
            
            **3. Bernoulli Naive Bayes (Binary features):**
            ```
            P(xᵢ | Class = c) = pᵢᴄˣᵢ × (1 - pᵢᴄ)ᵧᵡⵡᵢ
            ```
            Where pᵢᴄ is probability of feature i being 1 in class c
            
            **Laplace Smoothing (for zero probabilities):**
            ```
            P(xᵢ | Class = c) = (count(xᵢ, c) + α) / (count(c) + α × |V|)
            ```
            Where α = smoothing parameter (usually 1), |V| = vocabulary size
            """,
            
            # 5. Step-by-step working
            'algorithm_steps': """
            🔹 **How Naive Bayes Works (Step-by-Step)**
            
            **Step 1: Calculate Prior Probabilities** 📈
            • Count frequency of each class in training data
            • P(Class = c) = count(class c) / total_samples
            • Example: P(Spam) = 400/1000 = 0.4, P(Not Spam) = 600/1000 = 0.6
            
            **Step 2: Calculate Feature Likelihoods** 🔢
            • For each class, calculate P(feature | class)
            • Gaussian: Estimate mean and variance for each feature
            • Multinomial: Count feature occurrences in each class
            • Bernoulli: Calculate probability of feature being 1
            
            **Step 3: Handle Feature Types** 🔍
            • **Continuous features**: Use Gaussian assumption
            • **Count features**: Use Multinomial distribution
            • **Binary features**: Use Bernoulli distribution
            • **Mixed types**: Use appropriate distribution per feature
            
            **Step 4: Apply Laplace Smoothing** ✨
            • Add small constant to avoid zero probabilities
            • Prevents issues when feature never seen in training
            • Particularly important for text classification
            
            **Step 5: Make Predictions** 🔮
            • For new sample: calculate posterior for each class
            • P(Class | Features) ∝ P(Class) × ∏ P(feature | Class)
            • Choose class with highest posterior probability
            
            **Step 6: Handle Numerical Stability** 📋
            • Use log probabilities to prevent underflow
            • log P(Class | Features) = log P(Class) + Σ log P(feature | Class)
            • Subtract maximum before exponentiating
            
            **Step 7: Return Probabilities** 📊
            • Normalize predictions to get probability estimates
            • Can return both predicted class and confidence scores
            """,
            
            # 6. Pseudocode
            'pseudocode': """
            🔹 **Pseudocode (Easy to Understand)**
            
            ```
            ALGORITHM: Naive Bayes Training
            
            INPUT: 
                - X: feature matrix (n_samples × n_features)
                - y: class labels (n_samples × 1)
                - nb_type: 'gaussian', 'multinomial', or 'bernoulli'
            
            OUTPUT:
                - priors: P(class) for each class
                - likelihoods: P(feature | class) parameters
            
            BEGIN
                1. CALCULATE class priors:
                   FOR each class c:
                       priors[c] = count(y == c) / len(y)
                
                2. CALCULATE feature likelihoods:
                   IF nb_type == 'gaussian':
                       FOR each class c:
                           FOR each feature f:
                               mean[c][f] = mean(X[y==c, f])
                               std[c][f] = std(X[y==c, f])
                               
                   ELIF nb_type == 'multinomial':
                       FOR each class c:
                           FOR each feature f:
                               # With Laplace smoothing
                               count_cf = sum(X[y==c, f])
                               count_c = sum(X[y==c, :])
                               theta[c][f] = (count_cf + 1) / (count_c + |V|)
                               
                   ELIF nb_type == 'bernoulli':
                       FOR each class c:
                           FOR each feature f:
                               # Probability of feature being 1
                               p[c][f] = (sum(X[y==c, f]) + 1) / (count(y==c) + 2)
                
                3. RETURN (priors, likelihoods)
            END
            
            PREDICTION:
            BEGIN
                1. FOR new sample x:
                   FOR each class c:
                       log_posterior[c] = log(priors[c])
                       
                       FOR each feature f:
                           IF nb_type == 'gaussian':
                               # Gaussian probability density
                               log_likelihood = -0.5 * log(2π * std[c][f]^2)
                               log_likelihood -= (x[f] - mean[c][f])^2 / (2 * std[c][f]^2)
                               
                           ELIF nb_type == 'multinomial':
                               log_likelihood = x[f] * log(theta[c][f])
                               
                           ELIF nb_type == 'bernoulli':
                               IF x[f] == 1:
                                   log_likelihood = log(p[c][f])
                               ELSE:
                                   log_likelihood = log(1 - p[c][f])
                           
                           log_posterior[c] += log_likelihood
                
                2. predicted_class = argmax(log_posterior)
                
                3. # Convert to probabilities (optional)
                   max_log = max(log_posterior)
                   FOR each class c:
                       prob[c] = exp(log_posterior[c] - max_log)
                   normalize prob to sum to 1
                
                4. RETURN predicted_class, prob
            END
            ```
            """,
            
            # 7. Python implementation
            'python_implementation': """
            🔹 **Python Implementation**
            
            **From Scratch (Gaussian Naive Bayes):**
            ```python
            import numpy as np
            from collections import defaultdict
            
            class GaussianNaiveBayes:
                def __init__(self):
                    self.class_priors = {}
                    self.feature_means = defaultdict(dict)
                    self.feature_stds = defaultdict(dict)
                    self.classes = None
                
                def fit(self, X, y):
                    \"\"\"Train Gaussian Naive Bayes.\"\"\"
                    self.classes = np.unique(y)
                    n_samples = len(y)
                    
                    # Calculate class priors
                    for class_label in self.classes:
                        class_count = np.sum(y == class_label)
                        self.class_priors[class_label] = class_count / n_samples
                    
                    # Calculate feature statistics for each class
                    for class_label in self.classes:
                        class_mask = (y == class_label)
                        class_data = X[class_mask]
                        
                        for feature_idx in range(X.shape[1]):
                            feature_values = class_data[:, feature_idx]
                            self.feature_means[class_label][feature_idx] = np.mean(feature_values)
                            self.feature_stds[class_label][feature_idx] = np.std(feature_values) + 1e-9  # Add small value to avoid division by zero
                
                def _gaussian_pdf(self, x, mean, std):
                    \"\"\"Calculate Gaussian probability density function.\"\"\"
                    exponent = -((x - mean) ** 2) / (2 * std ** 2)
                    return (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(exponent)
                
                def predict_proba(self, X):
                    \"\"\"Predict class probabilities.\"\"\"
                    probabilities = []
                    
                    for sample in X:
                        class_probabilities = {}
                        
                        # Calculate posterior for each class
                        for class_label in self.classes:
                            # Start with prior probability
                            log_prob = np.log(self.class_priors[class_label])
                            
                            # Add log likelihood for each feature
                            for feature_idx, feature_value in enumerate(sample):
                                mean = self.feature_means[class_label][feature_idx]
                                std = self.feature_stds[class_label][feature_idx]
                                likelihood = self._gaussian_pdf(feature_value, mean, std)
                                log_prob += np.log(likelihood + 1e-10)  # Add small value to avoid log(0)
                            
                            class_probabilities[class_label] = log_prob
                        
                        # Convert log probabilities to probabilities
                        max_log_prob = max(class_probabilities.values())
                        for class_label in class_probabilities:
                            class_probabilities[class_label] = np.exp(class_probabilities[class_label] - max_log_prob)
                        
                        # Normalize probabilities
                        total_prob = sum(class_probabilities.values())
                        for class_label in class_probabilities:
                            class_probabilities[class_label] /= total_prob
                        
                        probabilities.append(class_probabilities)
                    
                    return probabilities
                
                def predict(self, X):
                    \"\"\"Predict classes.\"\"\"
                    probabilities = self.predict_proba(X)
                    predictions = []
                    
                    for prob_dict in probabilities:
                        predicted_class = max(prob_dict, key=prob_dict.get)
                        predictions.append(predicted_class)
                    
                    return np.array(predictions)
            
            # Example usage
            from sklearn.datasets import make_classification
            from sklearn.model_selection import train_test_split
            
            # Generate data
            X, y = make_classification(n_samples=1000, n_features=4, n_classes=3, random_state=42)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train and predict
            nb = GaussianNaiveBayes()
            nb.fit(X_train, y_train)
            predictions = nb.predict(X_test)
            probabilities = nb.predict_proba(X_test)
            ```
            
            **Using Scikit-learn:**
            ```python
            from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
            from sklearn.feature_extraction.text import CountVectorizer
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import classification_report
            
            # Gaussian Naive Bayes (continuous features)
            gnb = GaussianNB()
            gnb.fit(X_train, y_train)
            y_pred = gnb.predict(X_test)
            y_proba = gnb.predict_proba(X_test)
            
            # Multinomial Naive Bayes (count features, e.g., text)
            # Example with text data
            texts = ["I love this movie", "This movie is terrible", "Great film"]
            labels = [1, 0, 1]  # 1=positive, 0=negative
            
            vectorizer = CountVectorizer()
            X_text = vectorizer.fit_transform(texts)
            
            mnb = MultinomialNB(alpha=1.0)  # alpha is Laplace smoothing parameter
            mnb.fit(X_text, labels)
            
            # Bernoulli Naive Bayes (binary features)
            # Convert features to binary
            X_binary = (X > 0).astype(int)
            
            bnb = BernoulliNB(alpha=1.0)
            bnb.fit(X_binary, y)
            predictions = bnb.predict(X_binary)
            
            # Get feature log probabilities
            feature_log_prob = gnb.feature_log_prob_
            class_log_prior = gnb.class_log_prior_
            ```
            """,
            
            # 8. Example with sample input/output
            'example': """
            🔹 **Complete Example: Movie Review Sentiment Classification**
            
            **Input Data (Movie Reviews):**
            ```
            Review                    | "great" | "terrible" | "love" | "hate" | Sentiment
            "I love this great movie" |    1    |     0      |   1    |   0    | Positive
            "Terrible hate this film" |    0    |     1      |   0    |   1    | Negative  
            "Great acting love it"    |    1    |     0      |   1    |   0    | Positive
            "I hate terrible movies"  |    0    |     1      |   0    |   1    | Negative
            "This is great film"      |    1    |     0      |   0    |   0    | Positive
            ```
            
            **Step-by-Step Naive Bayes Training:**
            ```
            1. Calculate Class Priors:
               P(Positive) = 3/5 = 0.6
               P(Negative) = 2/5 = 0.4
            
            2. Calculate Feature Likelihoods (with Laplace smoothing):
               
               For "great" word:
               P("great"|Positive) = (2 + 1) / (3 + 2) = 3/5 = 0.6
               P("great"|Negative) = (0 + 1) / (2 + 2) = 1/4 = 0.25
               
               For "terrible" word:
               P("terrible"|Positive) = (0 + 1) / (3 + 2) = 1/5 = 0.2
               P("terrible"|Negative) = (2 + 1) / (2 + 2) = 3/4 = 0.75
               
               For "love" word:
               P("love"|Positive) = (2 + 1) / (3 + 2) = 3/5 = 0.6
               P("love"|Negative) = (0 + 1) / (2 + 2) = 1/4 = 0.25
               
               For "hate" word:
               P("hate"|Positive) = (0 + 1) / (3 + 2) = 1/5 = 0.2
               P("hate"|Negative) = (2 + 1) / (2 + 2) = 3/4 = 0.75
            ```
            
            **New Review Prediction:**
            ```
            New Review: "I love great movies" → ["great"=1, "terrible"=0, "love"=1, "hate"=0]
            
            For Positive class:
            P(Positive | features) ∝ P(Positive) × P("great"=1|Positive) × P("terrible"=0|Positive) × P("love"=1|Positive) × P("hate"=0|Positive)
            
            = 0.6 × 0.6 × (1-0.2) × 0.6 × (1-0.2)
            = 0.6 × 0.6 × 0.8 × 0.6 × 0.8
            = 0.1382
            
            For Negative class:
            P(Negative | features) ∝ P(Negative) × P("great"=1|Negative) × P("terrible"=0|Negative) × P("love"=1|Negative) × P("hate"=0|Negative)
            
            = 0.4 × 0.25 × (1-0.75) × 0.25 × (1-0.75)
            = 0.4 × 0.25 × 0.25 × 0.25 × 0.25
            = 0.0039
            
            Normalize:
            Total = 0.1382 + 0.0039 = 0.1421
            P(Positive | features) = 0.1382 / 0.1421 = 0.973 (97.3%)
            P(Negative | features) = 0.0039 / 0.1421 = 0.027 (2.7%)
            
            Prediction: POSITIVE 😊 (97.3% confidence)
            ```
            
            **Why it worked:**
            ```
            Words "love" and "great" are strongly associated with positive sentiment
            Words "terrible" and "hate" are absent (good for positive prediction)
            Even though independence assumption is violated (words often co-occur),
            the overall pattern still leads to correct classification!
            ```
            """,
            
            # 9. Visualization explanation
            'visualization': """
            🔹 **Understanding Through Visualizations**
            
            **Class Prior Probabilities:**
            📊 Pie chart showing distribution of classes in training data
            • Shows baseline probabilities before considering features
            • Larger slices = more common classes
            • Influences final predictions (prior bias)
            
            **Feature Likelihood Distributions:**
            📈 Histograms/density plots for each feature by class
            • Gaussian NB: Bell curves showing mean/std for each class
            • Shows how features discriminate between classes
            • Overlapping distributions = less discriminative features
            
            **Feature Importance (Information Gain):**
            📋 Bar chart showing discriminative power of features
            • Height = how much feature helps distinguish classes
            • Calculated using mutual information or similar metrics
            • Helps identify most useful features
            
            **Decision Boundaries (2D):**
            🎯 Contour plots showing classification regions
            • Linear boundaries (due to independence assumption)
            • Shows how different classes are separated
            • Illustrates effect of naive assumption
            
            **Probability Heatmaps:**
            🌡️ Grid showing P(feature | class) for all combinations
            • Rows = features, columns = classes
            • Color intensity = probability value
            • Reveals feature-class relationships
            
            **Confusion Matrix:**
            📋 Shows actual vs predicted classifications
            • Diagonal = correct predictions
            • Off-diagonal = misclassifications
            • Reveals which classes are confused with each other
            """,
            
            # 10. Time and space complexity
            'complexity': """
            🔹 **Time & Space Complexity**
            
            **Time Complexity:**
            • **Training**: O(n × m) where n=samples, m=features
            • **Prediction**: O(c × m) where c=classes, m=features
            • **Parameter Estimation**: O(n × m × c) for all class-feature combinations
            • **Probability Calculation**: O(m) per sample per class
            
            **Space Complexity:**
            • **Model Storage**: O(c × m) for storing class-feature parameters
            • **Training Memory**: O(n × m) for dataset storage
            • **Gaussian NB**: Store mean and std for each feature-class pair
            • **Multinomial NB**: Store probability for each feature-class pair
            
            **Scalability:**
            • ✅ **Linear Scaling**: Training time scales linearly with data size
            • ✅ **Fast Prediction**: Extremely fast inference
            • ✅ **Memory Efficient**: Model size independent of training data size
            • ✅ **Parallel Friendly**: Feature calculations can be parallelized
            • ✅ **Online Learning**: Can update parameters incrementally
            • ⚠️ **Feature Growth**: Model size grows with number of features
            
            **Performance Characteristics:**
            • One of the fastest ML algorithms for both training and prediction
            • Can handle millions of samples and thousands of features efficiently
            • Ideal for real-time applications requiring instant predictions
            • Memory usage grows slowly with data size
            """,
            
            # 11. Advantages and disadvantages
            'pros_cons': """
            🔹 **Advantages** ✅
            • **Extremely Fast**: Lightning-fast training and prediction
            • **Simple Implementation**: Easy to understand and code
            • **Probabilistic Output**: Natural probability estimates
            • **Multi-class Native**: Handles multiple classes automatically
            • **Small Data Friendly**: Works well with limited training data
            • **Scalable**: Linear time complexity with data size
            • **Feature Independence**: Not affected by irrelevant features
            • **No Hyperparameters**: Minimal tuning required
            • **Memory Efficient**: Compact model representation
            • **Online Learning**: Can update model incrementally
            • **Baseline Performance**: Good starting point for classification
            
            🔹 **Disadvantages** ❌
            • **Naive Assumption**: Features independence rarely holds in practice
            • **Limited Expressiveness**: Cannot capture feature interactions
            • **Poor Probability Estimates**: Often overconfident predictions
            • **Categorical Feature Issues**: Struggles with high-cardinality categories
            • **Continuous Feature Assumptions**: Gaussian assumption may not hold
            • **Correlated Features**: Performance degrades with highly correlated features
            • **Zero Probability Problem**: Needs smoothing for unseen feature combinations
            • **Linear Decision Boundaries**: Cannot capture complex patterns
            • **Feature Engineering Dependent**: Requires good feature preprocessing
            """,
            
            # 12. When to use and when NOT to use
            'usage_guide': """
            🔹 **When TO Use Naive Bayes** ✅
            
            **Perfect for:**
            • 📧 **Text Classification**: Spam detection, sentiment analysis, topic classification
            • ⚡ **Real-time Systems**: When prediction speed is critical
            • 📊 **Small Datasets**: Limited training data available
            • 🚀 **Baseline Models**: Quick first model for new problems
            • 🖼️ **High-Dimensional Data**: Many features, sparse data
            • 📱 **Streaming Data**: Online learning requirements
            
            **Good when:**
            • Features are approximately independent
            • Need probability estimates (with calibration)
            • Interpretability is important
            • Computing resources are limited
            • Multi-class classification needed
            • Quick prototyping required
            
            🔹 **When NOT to Use Naive Bayes** ❌
            
            **Avoid when:**
            • 🔗 **Strong Feature Correlations**: Features are highly dependent
            • 🎯 **Complex Patterns**: Non-linear relationships in data
            • 📊 **Large Datasets**: Have plenty of data for sophisticated models
            • 📈 **Continuous Relationships**: Smooth numerical relationships
            • 🎮 **Feature Interactions**: Important feature combinations
            • 💰 **High Accuracy Requirements**: Need state-of-the-art performance
            
            **Use instead:**
            • Logistic Regression (for linear relationships with interactions)
            • Random Forest (for feature interactions and non-linear patterns)
            • Gradient Boosting (for maximum accuracy)
            • Neural Networks (for complex patterns with lots of data)
            • SVM (for high-dimensional data with complex boundaries)
            """,
            
            # 13. Common interview questions
            'interview_questions': """
            🔹 **Common Interview Questions & Answers**
            
            **Q1: Why is Naive Bayes called "naive"?**
            A: It assumes features are conditionally independent given the class, which is "naive" because:
            • Real-world features often correlate (e.g., height and weight)
            • In text: words like "not" and "good" often appear together
            • Despite this violation, it often works surprisingly well in practice
            
            **Q2: What's the difference between Gaussian, Multinomial, and Bernoulli Naive Bayes?**
            A:
            • **Gaussian**: Continuous features, assumes normal distribution
            • **Multinomial**: Count/frequency features (like word counts in text)
            • **Bernoulli**: Binary features (presence/absence of features)
            
            **Q3: How do you handle the zero probability problem?**
            A: Use Laplace (add-one) smoothing:
            • Add small constant (usually 1) to all counts
            • P(word|class) = (count + 1) / (total_words + vocabulary_size)
            • Ensures no probability is exactly zero
            • Prevents entire probability from becoming zero due to unseen features
            
            **Q4: Can Naive Bayes be used for regression?**
            A: Theoretically yes, but rarely done in practice:
            • Would need to discretize continuous target variable
            • Other algorithms (Linear Regression, Random Forest) are much better for regression
            • Naive Bayes is almost exclusively used for classification
            
            **Q5: How do you handle numerical features in Naive Bayes?**
            A: Options:
            • **Gaussian NB**: Assume features follow normal distribution
            • **Discretization**: Convert to categorical bins
            • **Kernel Density**: Use non-parametric density estimation
            • Most common: Use Gaussian assumption with mean and standard deviation
            """,
            
            # 14. Common mistakes
            'common_mistakes': """
            🔹 **Common Beginner Mistakes & How to Avoid**
            
            **Mistake 1: Not handling zero probabilities** 🚫
            ❌ Allowing zero probabilities which make entire prediction zero
            ✅ **Fix**: Use Laplace smoothing (add-one smoothing) to avoid zeros
            
            **Mistake 2: Wrong Naive Bayes type for data** 🚫
            ❌ Using Gaussian NB for count data or Multinomial NB for continuous data
            ✅ **Fix**: Match NB type to data: Gaussian (continuous), Multinomial (counts), Bernoulli (binary)
            
            **Mistake 3: Not preprocessing text properly** 🚫
            ❌ Using raw text without cleaning, tokenization, or feature extraction
            ✅ **Fix**: Proper text preprocessing: lowercase, remove punctuation, use CountVectorizer/TfidfVectorizer
            
            **Mistake 4: Ignoring feature scaling** 🚫
            ❌ For Gaussian NB, not scaling features can lead to dominance by large-scale features
            ✅ **Fix**: Standardize features for Gaussian NB, especially when features have very different scales
            
            **Mistake 5: Over-interpreting probability outputs** 🚫
            ❌ Treating Naive Bayes probabilities as well-calibrated confidence scores
            ✅ **Fix**: Use probability calibration techniques or focus on relative rankings rather than absolute probabilities
            
            **Mistake 6: Using with highly correlated features** 🚫
            ❌ Applying NB when features are strongly correlated, violating independence assumption
            ✅ **Fix**: Feature selection to remove highly correlated features or use algorithms that handle correlations
            
            **Mistake 7: Not handling class imbalance** 🚫
            ❌ Ignoring skewed class distributions in training data
            ✅ **Fix**: Use stratified sampling, class weights, or resampling techniques
            """,
            
            # 15. Comparison with similar algorithms
            'comparisons': """
            🔹 **Naive Bayes vs Similar Algorithms**
            
            **Naive Bayes vs Logistic Regression:**
            • **Naive Bayes**: Assumes feature independence, generative model
            • **Logistic Regression**: No independence assumption, discriminative model
            • **Use Logistic**: When features are correlated or you need feature interactions
            
            **Naive Bayes vs Random Forest:**
            • **Naive Bayes**: Fast, simple, probabilistic
            • **Random Forest**: Handles feature interactions, non-linear patterns
            • **Use Random Forest**: For tabular data with complex relationships
            
            **Naive Bayes vs SVM:**
            • **Naive Bayes**: Fast training, works with small data
            • **SVM**: Better with high-dimensional data, no probability assumptions
            • **Use SVM**: For text classification with many features
            
            **Naive Bayes vs K-Nearest Neighbors:**
            • **Naive Bayes**: Fast prediction, global model
            • **KNN**: No training, handles local patterns, memory-based
            • **Use KNN**: When local similarities matter more than global patterns
            
            **Naive Bayes vs Neural Networks:**
            • **Naive Bayes**: Simple, interpretable, fast
            • **Neural Networks**: Complex patterns, needs more data
            • **Use Neural Networks**: With large datasets and complex patterns
            
            **Different Naive Bayes Types:**
            • **Gaussian**: Best for continuous/numerical features
            • **Multinomial**: Perfect for text classification, count data
            • **Bernoulli**: Good for binary features, presence/absence data
            """,
            
            # 16. Real-world applications
            'real_world_applications': """
            🔹 **Real-World Applications & Industry Use Cases**
            
            **📧 Email & Communication:**
            • Spam email detection and filtering
            • Email categorization (inbox organization)
            • Phishing detection systems
            • Automatic email routing in customer service
            • Language detection in multilingual platforms
            
            **📝 Text Analytics & NLP:**
            • Sentiment analysis for social media monitoring
            • News article categorization and topic classification
            • Document classification in legal systems
            • Content moderation and toxicity detection
            • Author identification and authorship attribution
            
            **🏥 Healthcare & Medical:**
            • Disease diagnosis from symptoms
            • Medical document classification
            • Drug side effect prediction
            • Patient risk assessment
            • Clinical trial patient selection
            
            **🛒 E-commerce & Marketing:**
            • Product recommendation engines
            • Customer segmentation for targeted marketing
            • Review sentiment classification
            • Fraud detection in transactions
            • Price optimization strategies
            
            **📱 Technology & Software:**
            • Real-time content filtering
            • Search query classification
            • User behavior prediction
            • Network intrusion detection
            • Software bug triage and classification
            
            **💰 Finance & Banking:**
            • Credit scoring and risk assessment
            • Algorithmic trading signal classification
            • Insurance claim processing
            • Anti-money laundering detection
            • Market sentiment analysis
            
            **🔬 Scientific Research:**
            • Biological sequence classification
            • Image classification in astronomy
            • Environmental monitoring and classification
            • Weather pattern prediction
            • Chemical compound activity prediction
            
            **💡 Key Success Factors:**
            • Proper feature selection and engineering
            • Appropriate preprocessing for data type
            • Handling of class imbalance
            • Use of appropriate smoothing techniques
            • Understanding when independence assumption is reasonable
            """
        }
    
    def generate_sample_data(self, n_samples=400, n_features=4):
        """Generate sample classification data."""
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features,
            n_redundant=0,
            n_classes=3,
            random_state=42
        )
        return X, y
    
    def fit(self, X, y):
        """Fit the Naive Bayes model."""
        if self.nb_type == 'gaussian':
            self.model = GaussianNB()
        elif self.nb_type == 'multinomial':
            self.model = MultinomialNB()
        else:  # bernoulli
            self.model = BernoulliNB()
        
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
    
    def streamlit_interface(self):
        """Create comprehensive Streamlit interface for Naive Bayes."""
        st.subheader("🎲 Naive Bayes")
        
        theory = self.get_theory()
        
        # Main tabs for comprehensive coverage
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🎯 Overview", "📚 Deep Dive", "💻 Implementation", 
            "🧪 Interactive Demo", "❓ Q&A", "🏢 Applications"
        ])
        
        with tab1:
            # Overview Tab - Essential Information
            st.markdown("### 🎯 What is Naive Bayes?")
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
            st.markdown("### 🧪 Try Naive Bayes Yourself!")
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
            nb_type = st.selectbox("Naive Bayes Type:", ['gaussian', 'multinomial', 'bernoulli'])
        with col2:
            n_samples = st.slider("Number of samples:", 200, 800, 400)
        with col3:
            n_features = st.slider("Number of features:", 2, 10, 4)
        
        # Update parameters
        self.nb_type = nb_type
        
        # Generate data and train model
        X, y = self.generate_sample_data(n_samples, n_features)
        
        # For multinomial and bernoulli, ensure non-negative features
        if nb_type in ['multinomial', 'bernoulli']:
            X = np.abs(X)
            if nb_type == 'bernoulli':
                X = (X > np.median(X)).astype(int)  # Convert to binary
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.fit(X_train, y_train)
        
        # Results section
        st.markdown("### 📊 Results")
        
        # Basic accuracy
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Additional metrics
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{accuracy:.3f}")
        with col2:
            st.metric("Precision", f"{precision:.3f}")
        with col3:
            st.metric("Recall", f"{recall:.3f}")
        with col4:
            st.metric("F1 Score", f"{f1:.3f}")
        
        # Interpretation
        st.markdown("### 🔍 Interpretation")
        
        if accuracy > 0.9:
            st.success(f"**Excellent performance!** Accuracy: {accuracy:.3f}")
            st.write("The Naive Bayes classifier is performing very well on this dataset.")
        elif accuracy > 0.8:
            st.info(f"**Good performance.** Accuracy: {accuracy:.3f}")
            st.write("The classifier is performing well.")
        elif accuracy > 0.7:
            st.warning(f"**Moderate performance.** Accuracy: {accuracy:.3f}")
            st.write("Consider feature engineering or trying different variants.")
        else:
            st.error(f"**Poor performance.** Accuracy: {accuracy:.3f}")
            st.write("The dataset might not be suitable for Naive Bayes or needs preprocessing.")


def main():
    """Main function for testing Naive Bayes."""
    nb = NaiveBayes()
    nb.streamlit_interface()


if __name__ == "__main__":
    main()