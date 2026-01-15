"""
Neural Network Algorithm Implementation

Neural Networks are computing systems inspired by biological neural networks
that learn to perform tasks by considering examples, generally without
task-specific programming.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report,
                           mean_squared_error, r2_score, mean_absolute_error)
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler
import streamlit as st


class NeuralNetwork:
    """
    Neural Network implementation with educational explanations.
    
    Multi-layer Perceptron (MLP) that uses backpropagation for training.
    Consists of input layer, hidden layers, and output layer.
    """
    
    def __init__(self, hidden_layer_sizes=(100,), task_type='classification', 
                 activation='relu', learning_rate=0.001):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.task_type = task_type
        self.activation = activation
        self.learning_rate = learning_rate
        self.model = None
        self.is_fitted = False
        
    @staticmethod
    def get_theory():
        """Return comprehensive theoretical explanation of Neural Networks."""
        return {
            'name': 'Neural Network (Multi-layer Perceptron)',
            'type': 'Supervised Learning - Classification/Regression',
            
            # 1. What the algorithm is
            'definition': """
            🔹 **What is a Neural Network?**
            A Neural Network is like a simplified brain made of artificial neurons that learn 
            patterns by adjusting connections between them. Think of it as a team of decision-makers 
            where each member processes information and passes it to the next level.
            """,
            
            # 2. Why the algorithm is used
            'motivation': """
            🔹 **Why Use Neural Networks?**
            • 🖼️ **Image Recognition**: Face detection, medical imaging, autonomous vehicles
            • 📝 **Natural Language**: Translation, chatbots, text analysis
            • 🎮 **Game AI**: Chess masters, AlphaGo, game strategy
            • 🔊 **Speech Recognition**: Voice assistants, audio processing
            • 🏥 **Medical Diagnosis**: Disease detection, drug discovery
            • 🎨 **Creative AI**: Art generation, music composition
            """,
            
            # 3. Intuition with real-life analogy
            'intuition': """
            🔹 **Real-Life Analogy: The Corporate Decision-Making Hierarchy**
            
            Imagine a large corporation making complex business decisions:
            
            🏢 **The Problem**: Should we launch a new product?
            📊 **Input**: Market data, customer feedback, financial reports, competitor analysis
            
            **Neural Network is like this corporate hierarchy:**
            
            **Input Layer (Data Collection Department)** 📊
            - Market Research Team: Collects customer data
            - Financial Team: Gathers cost/revenue data  
            - Competition Team: Analyzes competitor moves
            - Technical Team: Assesses feasibility
            
            **Hidden Layer 1 (Middle Management)** 💼
            - **Marketing Manager**: Combines customer + competition data
              Input: Raw market data → Processing: "Strong demand in age 25-40"
            - **Finance Manager**: Combines cost + revenue data
              Input: Financial reports → Processing: "Projected 15% profit margin"
            - **Tech Manager**: Evaluates technical feasibility
              Input: Technical specs → Processing: "95% confidence we can build it"
            
            **Hidden Layer 2 (Senior Management)** 🎯
            - **VP Strategy**: Combines marketing + finance insights
              Input: Demand + profit data → "Market opportunity worth $10M"
            - **VP Operations**: Combines tech + finance insights  
              Input: Feasibility + costs → "Can deliver on time and budget"
            
            **Output Layer (CEO Decision)** 🎆
            - **CEO**: Makes final decision based on all VP recommendations
              Input: All senior insights → **Decision: LAUNCH THE PRODUCT!**
            
            **Key Insights:**
            • **Layered Processing**: Each level adds more sophisticated analysis
            • **Weighted Connections**: Some inputs matter more (CEO trusts finance VP most)
            • **Non-linear Thinking**: Managers don't just add numbers - they make judgment calls
            • **Learning**: Bad decisions lead to adjusting who influences what
            
            **The Magic - Backpropagation Learning:**
            If the product fails:
            - CEO: "I trusted finance VP too much, marketing VP too little"
            - Finance VP: "I should weight risk assessment higher"
            - Marketing Manager: "I should focus more on competitor data"
            
            Everyone adjusts their decision-making process!
            
            🎯 **In data terms**: 
            - Employees = Neurons
            - Departments = Layers  
            - Influence/Trust = Weights
            - Decision Rules = Activation Functions
            - Learning from Mistakes = Backpropagation
            """,
            
            # 4. Mathematical foundation
            'math_foundation': """
            🔹 **Mathematical Foundation (Step-by-Step)**
            
            **Forward Propagation:**
            
            **Layer Computation:**
            ```
            zⲗⁿ = Wⲗ × a^(l-1) + bⲗ
            aⲗ = σ(zⲗ)
            ```
            Where:
            • `zⲗ` = weighted input to layer l
            • `Wⲗ` = weight matrix for layer l
            • `a^(l-1)` = activations from previous layer
            • `bⲗ` = bias vector for layer l
            • `σ()` = activation function
            • `aⲗ` = output activations of layer l
            
            **Activation Functions:**
            
            **1. ReLU (Rectified Linear Unit):**
            ```
            ReLU(x) = max(0, x)
            Derivative: ReLU'(x) = 1 if x > 0, else 0
            ```
            
            **2. Sigmoid:**
            ```
            σ(x) = 1 / (1 + e^(-x))
            Derivative: σ'(x) = σ(x)(1 - σ(x))
            ```
            
            **3. Tanh:**
            ```
            tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
            Derivative: tanh'(x) = 1 - tanh²(x)
            ```
            
            **Loss Functions:**
            
            **Regression (Mean Squared Error):**
            ```
            L = (1/2) × (y - ŷ)²
            ```
            
            **Classification (Cross-Entropy):**
            ```
            L = -Σ yᵢ log(ŷᵢ)
            ```
            
            **Backpropagation Algorithm:**
            
            **Output Layer Error:**
            ```
            δᴿ = (aᴿ - y) ⊙ σ'(zᴿ)
            ```
            
            **Hidden Layer Error:**
            ```
            δⲗ = (W^(l+1)ᵀ δ^(l+1)) ⊙ σ'(zⲗ)
            ```
            Where ⊙ denotes element-wise multiplication
            
            **Weight Updates (Gradient Descent):**
            ```
            Wⲗ := Wⲗ - η × δⲗ × a^(l-1)ᵀ
            bⲗ := bⲗ - η × δⲗ
            ```
            Where η is the learning rate
            
            **Mini-batch Gradient Descent:**
            ```
            For batch of m samples:
            ∇Wⲗ = (1/m) Σᵢ₌₁ᵐ δⲗᵢ × a^(l-1)ᵢ
            Wⲗ := Wⲗ - η × ∇Wⲗ
            ```
            
            **Regularization:**
            
            **L2 Regularization:**
            ```
            L_regularized = L + (λ/2) Σ ||Wⲗ||²
            ```
            
            **Dropout:**
            ```
            During training: aⲗ = aⲗ ⊙ mask (randomly set some neurons to 0)
            During inference: aⲗ = aⲗ × (1 - dropout_rate)
            ```
            """,
            
            # 5. Step-by-step working
            'algorithm_steps': """
            🔹 **How Neural Networks Work (Step-by-Step)**
            
            **Step 1: Network Initialization** 🎆
            • Create layers: input → hidden(s) → output
            • Initialize weights randomly (Xavier/He initialization)
            • Set biases to zero or small random values
            • Choose activation functions for each layer
            
            **Step 2: Forward Propagation** ➡️
            • **Input Layer**: Feed data into first layer
            • **Hidden Layers**: For each layer:
              - Multiply inputs by weights: z = W × a + b
              - Apply activation function: a = σ(z)
              - Pass output to next layer
            • **Output Layer**: Generate final predictions
            
            **Step 3: Loss Calculation** 📊
            • Compare predictions with true labels
            • Calculate loss (MSE for regression, cross-entropy for classification)
            • Add regularization terms if using L1/L2 regularization
            
            **Step 4: Backpropagation** ⬅️
            • **Output Layer**: Calculate error gradients
            • **Hidden Layers**: Propagate errors backward through network
            • **Weight Gradients**: Calculate how much each weight contributed to error
            • **Bias Gradients**: Calculate bias contributions to error
            
            **Step 5: Parameter Updates** ⚙️
            • Update weights: W = W - learning_rate × gradient_W
            • Update biases: b = b - learning_rate × gradient_b
            • Apply regularization (weight decay)
            • Apply momentum or adaptive learning rates if used
            
            **Step 6: Repeat Training** 🔄
            • Process mini-batches of data
            • Repeat steps 2-5 for each batch
            • Complete epochs (full passes through dataset)
            • Monitor training and validation loss
            
            **Step 7: Validation & Early Stopping** ✋
            • Evaluate on validation set after each epoch
            • Track validation loss and accuracy
            • Stop training if validation loss stops improving
            • Prevent overfitting with early stopping
            
            **Step 8: Testing & Evaluation** ✅
            • Evaluate final model on test set
            • Calculate performance metrics
            • Analyze model predictions and errors
            
            **Key Training Techniques:**
            • **Batch Normalization**: Normalize inputs to each layer
            • **Dropout**: Randomly disable neurons during training
            • **Learning Rate Scheduling**: Adjust learning rate over time
            • **Data Augmentation**: Create variations of training data
            """,
            
            # 6. Pseudocode
            'pseudocode': """
            🔹 **Pseudocode (Easy to Understand)**
            
            ```
            ALGORITHM: Neural Network Training
            
            INPUT: 
                - X: feature matrix (n_samples × n_features)
                - y: target values (n_samples × 1)
                - architecture: [input_size, hidden1, hidden2, ..., output_size]
                - learning_rate: step size for gradient descent
                - epochs: number of training iterations
                - batch_size: samples per mini-batch
            
            OUTPUT:
                - trained_network: neural network with learned weights
            
            BEGIN
                1. INITIALIZE network:
                   weights = []
                   biases = []
                   FOR each layer in architecture:
                       W = random_normal(prev_size, current_size) * sqrt(2/prev_size)  # He initialization
                       b = zeros(current_size)
                       weights.append(W)
                       biases.append(b)
                
                2. FOR epoch = 1 to epochs:
                   
                   # Shuffle data for each epoch
                   X, y = shuffle(X, y)
                   
                   # Process mini-batches
                   FOR each batch in create_batches(X, y, batch_size):
                       X_batch, y_batch = batch
                       
                       # FORWARD PROPAGATION
                       activations = [X_batch]  # Store all layer activations
                       z_values = []  # Store pre-activation values
                       
                       current_input = X_batch
                       FOR layer_idx in range(num_layers):
                           z = current_input @ weights[layer_idx] + biases[layer_idx]
                           z_values.append(z)
                           
                           IF layer_idx < num_layers - 1:  # Hidden layers
                               a = activation_function(z)  # ReLU, tanh, etc.
                           ELSE:  # Output layer
                               IF classification:
                                   a = softmax(z)
                               ELSE:  # regression
                                   a = z
                           
                           activations.append(a)
                           current_input = a
                       
                       predictions = activations[-1]
                       
                       # CALCULATE LOSS
                       IF classification:
                           loss = cross_entropy(y_batch, predictions)
                       ELSE:
                           loss = mean_squared_error(y_batch, predictions)
                       
                       # BACKPROPAGATION
                       # Calculate output layer error
                       IF classification:
                           delta = predictions - y_batch  # Softmax + cross-entropy derivative
                       ELSE:
                           delta = (predictions - y_batch) * 2  # MSE derivative
                       
                       deltas = [delta]
                       
                       # Calculate hidden layer errors (backward)
                       FOR layer_idx in range(num_layers-2, -1, -1):  # Go backwards
                           z = z_values[layer_idx]
                           delta_next = deltas[0]
                           W_next = weights[layer_idx + 1]
                           
                           # Backpropagate error
                           delta = (delta_next @ W_next.T) * activation_derivative(z)
                           deltas.insert(0, delta)
                       
                       # UPDATE WEIGHTS AND BIASES
                       FOR layer_idx in range(num_layers):
                           # Calculate gradients
                           dW = activations[layer_idx].T @ deltas[layer_idx] / batch_size
                           db = mean(deltas[layer_idx], axis=0)
                           
                           # Update parameters
                           weights[layer_idx] -= learning_rate * dW
                           biases[layer_idx] -= learning_rate * db
                   
                   # VALIDATION (optional)
                   IF validation_data provided:
                       val_predictions = forward_pass(X_val, weights, biases)
                       val_loss = calculate_loss(y_val, val_predictions)
                       
                       IF early_stopping AND val_loss stopped improving:
                           BREAK
                
                3. RETURN (weights, biases)
            END
            
            PREDICTION:
            BEGIN
                FUNCTION predict(X_new, weights, biases):
                    current_input = X_new
                    
                    FOR layer_idx in range(num_layers):
                        z = current_input @ weights[layer_idx] + biases[layer_idx]
                        
                        IF layer_idx < num_layers - 1:  # Hidden layers
                            current_input = activation_function(z)
                        ELSE:  # Output layer
                            IF classification:
                                current_input = softmax(z)
                            ELSE:
                                current_input = z
                    
                    RETURN current_input
            END
            
            HELPER FUNCTIONS:
            BEGIN
                FUNCTION activation_function(x):
                    RETURN max(0, x)  # ReLU
                
                FUNCTION activation_derivative(x):
                    RETURN (x > 0).astype(float)  # ReLU derivative
                
                FUNCTION softmax(x):
                    exp_x = exp(x - max(x, axis=1))  # For numerical stability
                    RETURN exp_x / sum(exp_x, axis=1)
                
                FUNCTION cross_entropy(y_true, y_pred):
                    RETURN -sum(y_true * log(y_pred + 1e-15))  # Add small epsilon
            END
            ```
            """,
            
            # 7. Python implementation
            'python_implementation': """
            🔹 **Python Implementation**
            
            **From Scratch (Simplified Neural Network):**
            ```python
            import numpy as np
            from sklearn.preprocessing import LabelEncoder, StandardScaler
            from sklearn.model_selection import train_test_split
            
            class SimpleNeuralNetwork:
                def __init__(self, hidden_layers=[100], learning_rate=0.01, 
                           activation='relu', epochs=100, batch_size=32):
                    self.hidden_layers = hidden_layers
                    self.learning_rate = learning_rate
                    self.activation = activation
                    self.epochs = epochs
                    self.batch_size = batch_size
                    self.weights = []
                    self.biases = []
                    self.training_history = {'loss': [], 'accuracy': []}
                
                def _activation_function(self, x, derivative=False):
                    \"\"\"Apply activation function.\"\"\"
                    if self.activation == 'relu':
                        if derivative:
                            return (x > 0).astype(float)
                        return np.maximum(0, x)
                    elif self.activation == 'sigmoid':
                        sigmoid = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
                        if derivative:
                            return sigmoid * (1 - sigmoid)
                        return sigmoid
                    elif self.activation == 'tanh':
                        if derivative:
                            return 1 - np.tanh(x) ** 2
                        return np.tanh(x)
                
                def _softmax(self, x):
                    \"\"\"Softmax activation for output layer.\"\"\"
                    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
                    return exp_x / np.sum(exp_x, axis=1, keepdims=True)
                
                def _initialize_weights(self, input_size, output_size):
                    \"\"\"Initialize weights using He initialization.\"\"\"
                    layer_sizes = [input_size] + self.hidden_layers + [output_size]
                    
                    for i in range(len(layer_sizes) - 1):
                        # He initialization for ReLU
                        w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
                        b = np.zeros((1, layer_sizes[i+1]))
                        self.weights.append(w)
                        self.biases.append(b)
                
                def _forward_propagation(self, X):
                    \"\"\"Forward pass through the network.\"\"\"
                    self.activations = [X]
                    self.z_values = []
                    
                    current_input = X
                    
                    # Hidden layers
                    for i in range(len(self.weights) - 1):
                        z = np.dot(current_input, self.weights[i]) + self.biases[i]
                        self.z_values.append(z)
                        a = self._activation_function(z)
                        self.activations.append(a)
                        current_input = a
                    
                    # Output layer
                    z = np.dot(current_input, self.weights[-1]) + self.biases[-1]
                    self.z_values.append(z)
                    
                    if self.is_classification:
                        a = self._softmax(z)
                    else:
                        a = z  # Linear output for regression
                    
                    self.activations.append(a)
                    return a
                
                def _backward_propagation(self, X, y, predictions):
                    \"\"\"Backward pass to calculate gradients.\"\"\"
                    m = X.shape[0]
                    
                    # Calculate output layer error
                    if self.is_classification:
                        delta = predictions - y  # Softmax + cross-entropy derivative
                    else:
                        delta = (predictions - y) * 2 / m  # MSE derivative
                    
                    deltas = [delta]
                    
                    # Calculate hidden layer errors (backward)
                    for i in range(len(self.weights) - 2, -1, -1):
                        delta = np.dot(deltas[0], self.weights[i + 1].T) * \
                                self._activation_function(self.z_values[i], derivative=True)
                        deltas.insert(0, delta)
                    
                    # Calculate gradients
                    for i in range(len(self.weights)):
                        dW = np.dot(self.activations[i].T, deltas[i]) / m
                        db = np.mean(deltas[i], axis=0, keepdims=True)
                        
                        # Update weights and biases
                        self.weights[i] -= self.learning_rate * dW
                        self.biases[i] -= self.learning_rate * db
                
                def fit(self, X, y):
                    \"\"\"Train the neural network.\"\"\"
                    # Determine if classification or regression
                    self.is_classification = len(np.unique(y)) < 20  # Simple heuristic
                    
                    if self.is_classification:
                        # One-hot encode labels
                        self.label_encoder = LabelEncoder()
                        y_encoded = self.label_encoder.fit_transform(y)
                        self.n_classes = len(self.label_encoder.classes_)
                        y_onehot = np.eye(self.n_classes)[y_encoded]
                        output_size = self.n_classes
                    else:
                        y_onehot = y.reshape(-1, 1)
                        output_size = 1
                    
                    # Initialize weights
                    input_size = X.shape[1]
                    self._initialize_weights(input_size, output_size)
                    
                    # Training loop
                    for epoch in range(self.epochs):
                        # Shuffle data
                        indices = np.random.permutation(X.shape[0])
                        X_shuffled = X[indices]
                        y_shuffled = y_onehot[indices]
                        
                        # Mini-batch training
                        for i in range(0, X.shape[0], self.batch_size):
                            X_batch = X_shuffled[i:i+self.batch_size]
                            y_batch = y_shuffled[i:i+self.batch_size]
                            
                            # Forward and backward pass
                            predictions = self._forward_propagation(X_batch)
                            self._backward_propagation(X_batch, y_batch, predictions)
                        
                        # Calculate epoch metrics
                        train_predictions = self._forward_propagation(X)
                        if self.is_classification:
                            loss = -np.mean(y_onehot * np.log(train_predictions + 1e-15))
                            accuracy = np.mean(np.argmax(train_predictions, axis=1) == y_encoded)
                            self.training_history['accuracy'].append(accuracy)
                        else:
                            loss = np.mean((y_onehot - train_predictions) ** 2)
                        
                        self.training_history['loss'].append(loss)
                        
                        if epoch % 10 == 0:
                            print(f'Epoch {epoch}, Loss: {loss:.4f}')
                
                def predict(self, X):
                    \"\"\"Make predictions.\"\"\"
                    predictions = self._forward_propagation(X)
                    
                    if self.is_classification:
                        return self.label_encoder.inverse_transform(np.argmax(predictions, axis=1))
                    else:
                        return predictions.flatten()
                
                def predict_proba(self, X):
                    \"\"\"Predict probabilities (classification only).\"\"\"
                    if not self.is_classification:
                        raise ValueError("predict_proba only available for classification")
                    return self._forward_propagation(X)
            
            # Example usage
            from sklearn.datasets import make_classification, make_regression
            import matplotlib.pyplot as plt
            
            # Classification example
            X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, random_state=42)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features (important for neural networks)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            nn = SimpleNeuralNetwork(hidden_layers=[64, 32], learning_rate=0.01, 
                                   activation='relu', epochs=100, batch_size=32)
            nn.fit(X_train_scaled, y_train)
            
            # Make predictions
            predictions = nn.predict(X_test_scaled)
            probabilities = nn.predict_proba(X_test_scaled)
            
            # Plot training history
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(nn.training_history['loss'])
            plt.title('Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            
            plt.subplot(1, 2, 2)
            if nn.is_classification:
                plt.plot(nn.training_history['accuracy'])
                plt.title('Training Accuracy')
                plt.ylabel('Accuracy')
            plt.show()
            ```
            
            **Using Scikit-learn:**
            ```python
            from sklearn.neural_network import MLPClassifier, MLPRegressor
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import GridSearchCV, validation_curve
            from sklearn.metrics import classification_report, accuracy_score
            
            # Feature scaling (CRUCIAL for neural networks!)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Classification
            mlp_clf = MLPClassifier(
                hidden_layer_sizes=(100, 50),    # Two hidden layers with 100 and 50 neurons
                activation='relu',               # ReLU activation function
                solver='adam',                   # Adam optimizer
                learning_rate_init=0.001,       # Initial learning rate
                max_iter=500,                   # Maximum number of iterations
                random_state=42,
                early_stopping=True,            # Stop early if validation score stops improving
                validation_fraction=0.1,        # Fraction of data for validation
                n_iter_no_change=10,            # Number of iterations with no improvement to wait
                alpha=0.0001                    # L2 regularization parameter
            )
            
            mlp_clf.fit(X_train_scaled, y_train)
            y_pred = mlp_clf.predict(X_test_scaled)
            y_proba = mlp_clf.predict_proba(X_test_scaled)
            
            # Training progress
            plt.plot(mlp_clf.loss_curve_)
            plt.title('Training Loss Curve')
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            plt.show()
            
            # Regression
            mlp_reg = MLPRegressor(
                hidden_layer_sizes=(100,),
                activation='relu',
                solver='lbfgs',                 # L-BFGS for small datasets
                alpha=0.001,
                random_state=42
            )
            
            mlp_reg.fit(X_train_scaled, y_train)
            y_pred_reg = mlp_reg.predict(X_test_scaled)
            
            # Hyperparameter tuning
            param_grid = {
                'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 100)],
                'activation': ['relu', 'tanh'],
                'learning_rate_init': [0.001, 0.01, 0.1],
                'alpha': [0.0001, 0.001, 0.01]
            }
            
            grid_search = GridSearchCV(
                MLPClassifier(max_iter=500, early_stopping=True, random_state=42),
                param_grid,
                cv=5,
                scoring='accuracy',
                n_jobs=-1
            )
            
            grid_search.fit(X_train_scaled, y_train)
            best_mlp = grid_search.best_estimator_
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
            
            # Validation curves
            param_range = [0.0001, 0.001, 0.01, 0.1]
            train_scores, validation_scores = validation_curve(
                MLPClassifier(hidden_layer_sizes=(100,), max_iter=500),
                X_train_scaled, y_train,
                param_name='alpha',
                param_range=param_range,
                cv=5, scoring='accuracy'
            )
            
            plt.plot(param_range, np.mean(train_scores, axis=1), label='Training')
            plt.plot(param_range, np.mean(validation_scores, axis=1), label='Validation')
            plt.xscale('log')
            plt.xlabel('Alpha (Regularization)')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.title('Validation Curve')
            plt.show()
            ```
            """,
            
            # 8. Example with sample input/output
            'example': """
            🔹 **Complete Example: Image Classification (Handwritten Digits)**
            
            **Input Data (Simplified 8x8 Pixel Images):**
            ```
            Digit 0: [[0,1,1,1,1,1,1,0],    Digit 1: [[0,0,1,1,0,0,0,0],
                     [1,0,0,0,0,0,0,1],             [0,1,1,1,0,0,0,0],
                     [1,0,0,0,0,0,0,1],             [0,0,1,1,0,0,0,0],
                     [1,0,0,0,0,0,0,1],             [0,0,1,1,0,0,0,0],
                     [1,0,0,0,0,0,0,1],             [0,0,1,1,0,0,0,0],
                     [1,0,0,0,0,0,0,1],             [0,0,1,1,0,0,0,0],
                     [1,0,0,0,0,0,0,1],             [0,0,1,1,0,0,0,0],
                     [0,1,1,1,1,1,1,0]]             [1,1,1,1,1,1,1,1]]
            ```
            
            **Network Architecture:**
            ```
            Input Layer:    64 neurons (8x8 flattened pixels)
                │
                │ W1: 64 × 128 weights
                ↓
            Hidden Layer 1: 128 neurons (ReLU activation)
                │
                │ W2: 128 × 64 weights  
                ↓
            Hidden Layer 2: 64 neurons (ReLU activation)
                │
                │ W3: 64 × 10 weights
                ↓
            Output Layer:   10 neurons (Softmax, one per digit 0-9)
            ```
            
            **Step-by-Step Forward Pass for Digit "1":**
            ```
            Input: x = [0,0,1,1,0,0,0,0, 0,1,1,1,0,0,0,0, ...] (64 values)
            
            Layer 1 (Hidden):
            z1 = W1 × x + b1 = [2.1, -0.8, 1.5, 0.3, ...] (128 values)
            a1 = ReLU(z1) = [2.1, 0, 1.5, 0.3, ...] (negative values → 0)
            
            Layer 2 (Hidden):
            z2 = W2 × a1 + b2 = [1.2, 0.8, -0.5, 2.1, ...] (64 values)
            a2 = ReLU(z2) = [1.2, 0.8, 0, 2.1, ...]
            
            Output Layer:
            z3 = W3 × a2 + b3 = [-2.1, 4.8, -1.2, 0.5, -0.8, 1.1, 0.2, -1.5, 0.9, -0.3]
            a3 = Softmax(z3) = [0.01, 0.89, 0.02, 0.03, 0.01, 0.02, 0.01, 0.00, 0.01, 0.00]
            
            Prediction: Digit 1 (89% confidence) ✅
            ```
            
            **Training Process (One Epoch):**
            ```
            1. Forward pass for all training samples
            
            2. Loss calculation:
               True label for "1": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
               Predicted:          [0.01, 0.89, 0.02, 0.03, 0.01, 0.02, 0.01, 0.00, 0.01, 0.00]
               Cross-entropy loss = -log(0.89) = 0.117
            
            3. Backpropagation:
               Output error: δ3 = [0.01, -0.11, 0.02, 0.03, 0.01, 0.02, 0.01, 0.00, 0.01, 0.00]
               Hidden2 error: δf = W3.T × δ2 × ReLU'(z2)
               Hidden1 error: δ1 = W2.T × δ2 × ReLU'(z1)
            
            4. Weight updates:
               W3 = W3 - 0.001 × (δ3 × a2.T)
               W2 = W2 - 0.001 × (δ2 × a1.T)
               W1 = W1 - 0.001 × (δ1 × x.T)
            ```
            
            **After Training (1000 epochs):**
            ```
            Network learns to detect patterns:
            - Hidden Layer 1: Detects edges, corners, basic shapes
            - Hidden Layer 2: Combines edges into digit parts (loops, lines, curves)
            - Output Layer: Combines parts into full digit recognition
            
            Test Results:
            - Digit 0: 96% accuracy (sometimes confused with 8)
            - Digit 1: 98% accuracy (very distinctive)
            - Digit 2: 92% accuracy (sometimes confused with 7)
            - Overall accuracy: 94% on test set
            ```
            
            **What the Network Learned:**
            ```
            Neuron 15 in Layer 1: Activates on vertical lines (detects "1", "7", "4")
            Neuron 23 in Layer 1: Activates on circles (detects "0", "8", "9")
            Neuron 67 in Layer 2: Combines vertical + horizontal lines (detects "4", "7")
            Output Neuron 1: High when seeing single vertical line pattern
            ```
            """,
            
            # 9. Visualization explanation
            'visualization': """
            🔹 **Understanding Through Visualizations**
            
            **Network Architecture Diagram:**
            🎨 Visual representation of layers and connections
            • Circles represent neurons, lines represent weights
            • Color intensity shows weight magnitude
            • Red = negative weights, blue = positive weights
            • Shows information flow from input to output
            
            **Training Loss Curves:**
            📉 Training and validation loss over epochs
            • Decreasing loss indicates learning
            • Gap between training and validation shows overfitting
            • Early stopping point where validation loss minimizes
            • Learning rate effects on convergence speed
            
            **Weight Heatmaps:**
            🌡️ Matrix visualization of learned weights
            • Each row/column represents neuron connections
            • Color intensity shows weight strength
            • Patterns reveal what features the network learned
            • Evolution of weights during training
            
            **Activation Maps:**
            📈 Neuron activations for specific inputs
            • Shows which neurons fire for different inputs
            • Reveals how information is processed layer by layer
            • Different inputs create different activation patterns
            • Helps understand internal representations
            
            **Decision Boundary Plots (2D data):**
            🎯 Classification boundaries in feature space
            • Shows complex non-linear boundaries
            • Compares single layer vs multi-layer networks
            • Effect of different activation functions
            • How depth increases boundary complexity
            
            **Feature Learning Visualization:**
            🔍 What each layer learns (for image data)
            • Layer 1: Edges, corners, simple patterns
            • Layer 2: Shapes, textures, object parts
            • Layer 3: Complete objects, complex patterns
            • Progressive complexity from simple to abstract
            """,
            
            # 10. Time and space complexity
            'complexity': """
            🔹 **Time & Space Complexity**
            
            **Time Complexity:**
            • **Forward Pass**: O(L × n × h²) where:
              - L = number of layers
              - n = number of samples
              - h = average hidden layer size
            • **Backward Pass**: O(L × n × h²) (same as forward)
            • **Single Prediction**: O(L × h²)
            • **Training Epoch**: O(epochs × L × n × h²)
            
            **Space Complexity:**
            • **Model Storage**: O(L × h²) for weights and biases
            • **Training Memory**: O(n × h) for activations and gradients
            • **Batch Processing**: Memory scales with batch size
            • **GPU Memory**: Can store large models and batches
            
            **Computational Characteristics:**
            • ✅ **Highly Parallelizable**: Matrix operations use GPU efficiently
            • ❌ **Memory Intensive**: Deep networks require significant RAM
            • ✅ **Batch Processing**: Efficient when processing many samples together
            • ⚠️ **Scalability**: Complexity grows quadratically with layer size
            
            **Optimization Strategies:**
            • **Mini-batch Processing**: Balance memory usage and convergence
            • **GPU Acceleration**: Use CUDA for parallel computation
            • **Mixed Precision**: Use 16-bit floats to reduce memory
            • **Gradient Accumulation**: Simulate larger batches with limited memory
            • **Model Compression**: Pruning, quantization to reduce size
            
            **Scaling Considerations:**
            • Training time increases dramatically with network size
            • Memory requirements can become prohibitive
            • Inference can be made very fast with optimized implementations
            • Distributed training needed for very large models
            """,
            
            # 11. Advantages and disadvantages
            'pros_cons': """
            🔹 **Advantages** ✅
            • **Universal Approximator**: Can learn any continuous function given enough neurons
            • **Automatic Feature Learning**: Discovers relevant features from raw data
            • **High Accuracy**: State-of-the-art performance on many complex tasks
            • **Versatile**: Works for classification, regression, and more
            • **Scalable**: Performance improves with more data and compute
            • **Non-linear Patterns**: Captures complex relationships in data
            • **Transfer Learning**: Pre-trained models can be adapted to new tasks
            • **Parallel Processing**: Efficient on modern GPU hardware
            • **Continuous Learning**: Can be updated with new data
            
            🔹 **Disadvantages** ❌
            • **Data Hungry**: Requires large amounts of training data
            • **Computationally Expensive**: Training takes significant time and resources
            • **Black Box**: Difficult to interpret decisions and understand reasoning
            • **Hyperparameter Sensitive**: Many parameters to tune (architecture, learning rate, etc.)
            • **Overfitting Prone**: Easily memorizes training data without generalization
            • **Vanishing Gradients**: Training can be unstable for very deep networks
            • **Local Minima**: May get stuck in suboptimal solutions
            • **Hardware Dependent**: Requires powerful GPUs for efficient training
            • **Feature Scaling Sensitive**: Requires careful data preprocessing
            • **Reproducibility Issues**: Random initialization can lead to different results
            """,
            
            # 12. When to use and when NOT to use
            'usage_guide': """
            🔹 **When TO Use Neural Networks** ✅
            
            **Perfect for:**
            • 🖼️ **Computer Vision**: Image classification, object detection, medical imaging
            • 📝 **Natural Language**: Text classification, machine translation, chatbots
            • 🔊 **Audio Processing**: Speech recognition, music generation, sound classification
            • 🎮 **Complex Patterns**: Non-linear relationships, high-dimensional data
            • 📊 **Large Datasets**: When you have millions of training samples
            • 🚀 **State-of-art Performance**: When accuracy is the top priority
            
            **Good when:**
            • Have abundant training data (>10K samples)
            • Features have complex interactions
            • Raw data needs feature extraction (pixels, text, audio)
            • Can afford computational resources for training
            • Accuracy is more important than interpretability
            • Have access to GPUs for acceleration
            
            🔹 **When NOT to Use Neural Networks** ❌
            
            **Avoid when:**
            • 📋 **Small Datasets**: < 1K samples (use simpler models)
            • 🔍 **Interpretability Critical**: Need to explain every decision
            • 📈 **Simple Relationships**: Linear patterns work fine
            • ⚡ **Resource Constrained**: Limited computational power
            • 📊 **Tabular Data**: Structured data with clear features
            • 🚀 **Quick Prototyping**: Need fast results for initial analysis
            • 💰 **Cost Sensitive**: Training and inference costs matter
            
            **Use instead:**
            • **Small Data**: Linear models, KNN, Naive Bayes
            • **Interpretability**: Decision Trees, Linear Regression
            • **Tabular Data**: Random Forest, Gradient Boosting
            • **Quick Results**: Simple models for rapid prototyping
            • **Limited Resources**: Traditional ML algorithms
            """,
            
            # 13. Common interview questions
            'interview_questions': """
            🔹 **Common Interview Questions & Answers**
            
            **Q1: What is the vanishing gradient problem and how do you solve it?**
            A: Gradients become very small in early layers during backpropagation:
            • **Problem**: Early layers learn very slowly or not at all
            • **Causes**: Sigmoid/tanh activation, very deep networks
            • **Solutions**: ReLU activation, residual connections, batch normalization, proper weight initialization (He/Xavier)
            
            **Q2: Why do we need activation functions?**
            A: Without activation functions, neural networks are just linear transformations:
            • **Linear problem**: Multiple linear layers = single linear layer
            • **Non-linearity**: Activation functions enable learning complex patterns
            • **Universal approximation**: Non-linear activations make networks capable of approximating any function
            
            **Q3: How do you prevent overfitting in neural networks?**
            A: Multiple regularization techniques:
            • **Dropout**: Randomly disable neurons during training
            • **Early stopping**: Stop training when validation loss increases
            • **L1/L2 regularization**: Add penalty terms to loss function
            • **Batch normalization**: Normalize inputs to each layer
            • **Data augmentation**: Create variations of training data
            • **Reduce complexity**: Fewer layers/neurons
            
            **Q4: What's the difference between batch, stochastic, and mini-batch gradient descent?**
            A:
            • **Batch**: Use entire dataset for each update (slow, stable)
            • **Stochastic**: Use single sample for each update (fast, noisy)
            • **Mini-batch**: Use small subset (best of both worlds, most common)
            
            **Q5: How do you choose the number of hidden layers and neurons?**
            A: Start simple and increase complexity:
            • **Begin**: Single hidden layer with neurons = (input + output) / 2
            • **Add layers**: If single layer underfits
            • **Add neurons**: If network still underfits
            • **Use validation**: Cross-validation to find optimal architecture
            • **Rule of thumb**: Start with 1-2 hidden layers, 10-100 neurons each
            
            **Q6: What is backpropagation and how does it work?**
            A: Algorithm for training neural networks:
            • **Forward pass**: Calculate predictions and loss
            • **Backward pass**: Calculate gradients using chain rule
            • **Weight update**: Adjust weights based on gradients
            • **Key insight**: Gradients flow backward through the network
            • **Chain rule**: Allows calculating gradients for any layer
            """,
            
            # 14. Common mistakes
            'common_mistakes': """
            🔹 **Common Beginner Mistakes & How to Avoid**
            
            **Mistake 1: Not scaling input features** 🚫
            ❌ Using raw features with different scales (age vs income)
            ✅ **Fix**: Always use StandardScaler() or MinMaxScaler() for neural networks
            
            **Mistake 2: Using too complex architecture initially** 🚫
            ❌ Starting with 10 hidden layers and 1000 neurons each
            ✅ **Fix**: Start simple (1-2 layers, 10-100 neurons), add complexity gradually
            
            **Mistake 3: Not using validation set** 🚫
            ❌ Training until fixed number of epochs without monitoring overfitting
            ✅ **Fix**: Use validation split and early stopping
            
            **Mistake 4: Wrong learning rate** 🚫
            ❌ Using learning rate too high (exploding gradients) or too low (no learning)
            ✅ **Fix**: Start with 0.001, try 0.0001-0.01 range, use learning rate scheduling
            
            **Mistake 5: No regularization** 🚫
            ❌ Training complex network without dropout or other regularization
            ✅ **Fix**: Use dropout (0.2-0.5), L2 regularization, batch normalization
            
            **Mistake 6: Poor weight initialization** 🚫
            ❌ Using random weights from wrong distribution
            ✅ **Fix**: Use He initialization for ReLU, Xavier for sigmoid/tanh
            
            **Mistake 7: Not checking for vanishing/exploding gradients** 🚫
            ❌ Ignoring gradient norms during training
            ✅ **Fix**: Monitor gradient norms, use gradient clipping, batch normalization
            
            **Mistake 8: Using neural networks for small datasets** 🚫
            ❌ Applying deep learning to datasets with < 1000 samples
            ✅ **Fix**: Use traditional ML algorithms for small datasets
            """,
            
            # 15. Comparison with similar algorithms
            'comparisons': """
            🔹 **Neural Networks vs Similar Algorithms**
            
            **Neural Networks vs Random Forest:**
            • **Neural Networks**: Better for unstructured data, requires more data
            • **Random Forest**: Better for structured data, more interpretable
            • **Use RF**: For tabular data with clear features
            • **Use NN**: For images, text, audio, complex patterns
            
            **Neural Networks vs SVM:**
            • **Neural Networks**: Better for large datasets, automatic feature extraction
            • **SVM**: Better for small datasets, works well in high dimensions
            • **Use SVM**: For text classification with limited data
            • **Use NN**: When you have lots of data and complex patterns
            
            **Neural Networks vs Gradient Boosting:**
            • **Neural Networks**: Better for unstructured data, requires more tuning
            • **Gradient Boosting**: Better for structured data, easier to tune
            • **Use GB**: For Kaggle competitions with tabular data
            • **Use NN**: For computer vision and NLP tasks
            
            **Neural Networks vs Linear Models:**
            • **Neural Networks**: Can learn non-linear patterns, requires more data
            • **Linear Models**: Simple, interpretable, work with small data
            • **Use Linear**: When relationships are roughly linear
            • **Use NN**: When relationships are complex and non-linear
            
            **Shallow vs Deep Networks:**
            • **Shallow**: Faster training, easier to tune, good for simple tasks
            • **Deep**: Better feature learning, handles complex tasks
            • **Use Shallow**: For simple classification with few features
            • **Use Deep**: For computer vision, NLP, complex pattern recognition
            """,
            
            # 16. Real-world applications
            'real_world_applications': """
            🔹 **Real-World Applications & Industry Use Cases**
            
            **🖼️ Computer Vision & Image Recognition:**
            • Autonomous vehicles: Object detection, lane recognition
            • Medical imaging: Cancer detection, radiology analysis
            • Security systems: Face recognition, surveillance
            • Quality control: Defect detection in manufacturing
            • Retail: Visual search, product recognition
            
            **📝 Natural Language Processing:**
            • Machine translation: Google Translate, DeepL
            • Chatbots and virtual assistants: GPT, ChatGPT, Siri
            • Sentiment analysis: Social media monitoring
            • Text generation: Content creation, summarization
            • Search engines: Query understanding, document ranking
            
            **🔊 Speech & Audio Processing:**
            • Speech recognition: Voice assistants, transcription services
            • Music generation: AI composers, style transfer
            • Audio classification: Genre recognition, sound detection
            • Voice synthesis: Text-to-speech systems
            • Noise reduction: Audio enhancement, denoising
            
            **🏥 Healthcare & Medicine:**
            • Drug discovery: Molecular property prediction
            • Diagnostic imaging: X-ray, MRI, CT scan analysis
            • Personalized medicine: Treatment recommendation
            • Epidemic modeling: Disease spread prediction
            • Clinical decision support: Risk assessment
            
            **💰 Finance & Trading:**
            • Algorithmic trading: Market prediction, strategy optimization
            • Fraud detection: Transaction monitoring, pattern recognition
            • Risk assessment: Credit scoring, portfolio management
            • Robo-advisors: Automated investment advice
            • Cryptocurrency: Price prediction, trading bots
            
            **🎮 Gaming & Entertainment:**
            • Game AI: AlphaGo, chess engines, strategy games
            • Content recommendation: Netflix, YouTube, Spotify
            • Procedural generation: Game worlds, level design
            • Character animation: Motion capture, realistic NPCs
            • Sports analytics: Player performance, strategy optimization
            
            **🚗 Transportation & Logistics:**
            • Route optimization: GPS navigation, delivery planning
            • Autonomous vehicles: Self-driving cars, drones
            • Traffic management: Smart city systems, flow optimization
            • Predictive maintenance: Vehicle health monitoring
            • Supply chain: Demand forecasting, inventory optimization
            
            **💡 Key Success Factors:**
            • Large, high-quality training datasets
            • Appropriate network architecture for the problem
            • Proper preprocessing and feature engineering
            • Effective regularization and hyperparameter tuning
            • Sufficient computational resources (GPUs)
            • Domain expertise for problem formulation
            • Continuous monitoring and model updating
            """
        }
    
    def generate_sample_data(self, task_type, n_samples=800, n_features=10):
        """Generate sample data for demonstration."""
        if task_type == 'classification':
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=max(5, n_features // 2),
                n_redundant=2,
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
        """Fit the Neural Network model."""
        if self.task_type == 'classification':
            self.model = MLPClassifier(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation=self.activation,
                learning_rate_init=self.learning_rate,
                max_iter=500,
                random_state=42
            )
        else:
            self.model = MLPRegressor(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation=self.activation,
                learning_rate_init=self.learning_rate,
                max_iter=500,
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
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y, y_pred, average='weighted', zero_division=0)
            }
        else:  # regression
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
            import numpy as np
            
            metrics = {
                'mse': mean_squared_error(y, y_pred),
                'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                'mae': mean_absolute_error(y, y_pred),
                'r2': r2_score(y, y_pred)
            }
        
        return metrics
    
    def plot_loss_curve(self):
        """Plot training loss curve."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting loss curve")
            
        if not hasattr(self.model, 'loss_curve_'):
            return None
            
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.model.loss_curve_, 'b-', linewidth=2)
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Loss')
        ax.set_title('Neural Network Training Loss')
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def streamlit_interface(self):
        """Create comprehensive Streamlit interface for Neural Network."""
        st.subheader("🧠 Neural Network (MLP)")
        
        theory = self.get_theory()
        
        # Main tabs for comprehensive coverage
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🎯 Overview", "📚 Deep Dive", "💻 Implementation", 
            "🧪 Interactive Demo", "❓ Q&A", "🏢 Applications"
        ])
        
        with tab1:
            # Overview Tab - Essential Information
            st.markdown("### 🎯 What is Neural Network?")
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
            st.markdown("### 🧪 Try Neural Network Yourself!")
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
        st.markdown("### 🔧 Network Architecture")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            task_type = st.selectbox("Task Type:", ['classification', 'regression'])
        with col2:
            n_hidden_layers = st.selectbox("Hidden layers:", [1, 2, 3])
        with col3:
            layer_size = st.slider("Neurons per layer:", 10, 200, 100, 10)
        with col4:
            activation = st.selectbox("Activation:", ['relu', 'tanh', 'logistic'])
        
        # Create hidden layer architecture
        hidden_layers = tuple([layer_size] * n_hidden_layers)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            learning_rate = st.slider("Learning rate:", 0.0001, 0.01, 0.001, 0.0001)
        with col2:
            n_samples = st.slider("Samples:", 500, 1500, 800)
        with col3:
            n_features = st.slider("Features:", 5, 20, 10)
        
        # Update parameters
        self.task_type = task_type
        self.hidden_layer_sizes = hidden_layers
        self.activation = activation
        self.learning_rate = learning_rate
        
        # Generate data and train model
        X, y = self.generate_sample_data(task_type, n_samples, n_features)
        
        # Feature scaling (important for neural networks)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42)
        
        # Train model
        with st.spinner('Training neural network...'):
            self.fit(X_train, y_train)
        
        # Results section
        st.markdown("### 📊 Results")
        
        # Architecture summary
        st.markdown("**Network Architecture:**")
        architecture = f"Input({n_features})"
        for i, size in enumerate(hidden_layers):
            architecture += f" → Hidden{i+1}({size})"
        if task_type == 'classification':
            n_classes = len(np.unique(y))
            architecture += f" → Output({n_classes})"
        else:
            architecture += f" → Output(1)"
        
        st.code(architecture)
        
        # Training information
        col1, col2, col3 = st.columns(3)
        col1.metric("Training Iterations", self.model.n_iter_)
        col2.metric("Hidden Layers", len(hidden_layers))
        col3.metric("Total Parameters", sum(layer.size for layer in self.model.coefs_))
        
        # Loss curve
        fig_loss = self.plot_loss_curve()
        if fig_loss:
            st.pyplot(fig_loss)
            plt.close()
        
        # Performance metrics
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
        
        # Interpretation
        st.markdown("### 🔍 Interpretation")
        
        if task_type == 'classification':
            if 'Accuracy' in test_metrics:
                accuracy = test_metrics['Accuracy']
                if accuracy > 0.9:
                    st.success(f"**Excellent performance!** Accuracy: {accuracy:.3f}")
                    st.write("The Neural Network is performing very well on this dataset.")
                elif accuracy > 0.8:
                    st.info(f"**Good performance.** Accuracy: {accuracy:.3f}")
                    st.write("The model is performing well.")
                elif accuracy > 0.7:
                    st.warning(f"**Moderate performance.** Accuracy: {accuracy:.3f}")
                    st.write("Consider tuning architecture or hyperparameters.")
                else:
                    st.error(f"**Poor performance.** Accuracy: {accuracy:.3f}")
                    st.write("The model may need more training time, data, or different architecture.")
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
        col1.metric("Architecture", f"{n_hidden_layers} Hidden")
        col2.metric("Activation", activation.capitalize())
        col3.metric("Learning Rate", f"{learning_rate}")
        col4.metric("Converged", "Yes" if self.model.n_iter_ < self.model.max_iter else "No")
        
        # Recommendations
        st.markdown("**Recommendations:**")
        if self.model.n_iter_ >= self.model.max_iter:
            st.write("• Model didn't converge. Try increasing max_iter or adjusting learning rate.")
        if learning_rate > 0.005:
            st.write("• High learning rate might cause instability. Consider reducing.")
        if len(hidden_layers) > 2 and n_samples < 1000:
            st.write("• Deep network with small dataset might overfit. Consider simpler architecture.")
        if task_type == 'classification' and 'Accuracy' in test_metrics:
            train_acc = train_metrics.get('Accuracy', 0)
            test_acc = test_metrics.get('Accuracy', 0)
            if train_acc - test_acc > 0.15:
                st.write("• Large gap between training and test suggests overfitting. Try regularization.")


def main():
    """Main function for testing Neural Network."""
    nn = NeuralNetwork()
    nn.streamlit_interface()


if __name__ == "__main__":
    main()