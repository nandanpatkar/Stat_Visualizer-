# Statistics & Machine Learning Learning App

An interactive Streamlit application for learning statistics and machine learning through visualization and hands-on exploration. Now featuring **10+ machine learning algorithms** with comprehensive explanations and interactive interfaces!

## 🆕 Version 2.0 Features

### 📊 Statistics (Enhanced)
- **Descriptive Statistics**: Calculate and visualize measures of central tendency, dispersion, quartiles, and percentiles
- **Inferential Statistics**: Perform Z-tests, T-tests, and confidence interval calculations  
- **Central Limit Theorem**: Interactive demonstration with various population distributions
- **Modular Architecture**: Clean, PEP8-compliant code structure

### 🤖 Machine Learning Algorithms
- **Supervised Learning**:
  - Linear Regression - Predict continuous values with linear relationships
  - Logistic Regression - Binary and multi-class classification
  - Decision Trees - Interpretable tree-based models
  - Random Forest - Ensemble of decision trees
  - Support Vector Machine (SVM) - Maximum margin classification
  - Naive Bayes - Probabilistic classification
  - K-Nearest Neighbors (KNN) - Instance-based learning
  - Neural Networks - Multi-layer perceptrons
  - Gradient Boosting - Sequential ensemble learning

- **Unsupervised Learning**:
  - K-Means Clustering - Partition data into clusters

### 🎯 Interactive Features
- **Real-time Parameter Tuning**: Adjust algorithm parameters and see results instantly
- **Comprehensive Visualizations**: Decision boundaries, learning curves, feature importance
- **Educational Explanations**: Theory, advantages, disadvantages, and use cases for each algorithm
- **Performance Metrics**: Accuracy, precision, recall, F1-score, R², and more
- **Data Input Options**: Upload CSV files, use sample data, or enter custom numbers

## 🏗️ Project Structure

```
Stat_Visualizer-/
├── algorithms/              # Machine learning algorithms
│   ├── __init__.py
│   ├── linear_regression.py
│   ├── logistic_regression.py
│   ├── decision_tree.py
│   ├── random_forest.py
│   ├── k_means.py
│   ├── k_nearest_neighbors.py
│   ├── support_vector_machine.py
│   ├── naive_bayes.py
│   ├── gradient_boosting.py
│   └── neural_network.py
├── stat_analysis/           # Statistical analysis modules  
│   ├── __init__.py
│   └── descriptive_stats.py
├── utils/                   # Utility functions
│   ├── __init__.py
│   └── data_utils.py
├── main_app.py             # Main application entry point
├── app.py                  # Original monolithic app (preserved)
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## 📋 Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd Stat_Visualizer-
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### New Modular App (Recommended)
Run the enhanced modular application:
```bash
streamlit run main_app.py
```

### Original App (Legacy)
Run the original monolithic app:
```bash
streamlit run app.py
```

## 🎓 Learning Path

1. **Start with Statistics**: Understand your data through descriptive statistics
2. **Explore Algorithms**: Try different ML algorithms on the same dataset
3. **Compare Performance**: Use metrics to evaluate which algorithms work best
4. **Understand Theory**: Read the explanations to understand when and why to use each algorithm
5. **Experiment**: Adjust parameters to see how they affect model performance

## 🛠️ Technical Details

- **Framework**: Streamlit for web interface
- **ML Library**: Scikit-learn for algorithms
- **Visualization**: Matplotlib, Seaborn, Plotly for interactive plots
- **Data Processing**: NumPy, Pandas for data manipulation
- **Code Style**: PEP8 compliant with modular architecture
- **Documentation**: Comprehensive docstrings and comments

## 📊 Supported Data Types

- **CSV Files**: Upload your own datasets
- **Sample Data**: Pre-generated datasets for quick exploration
- **Manual Entry**: Enter custom numbers for analysis

## 📈 Algorithm Capabilities

Each algorithm includes:
- ✅ Theoretical explanation with mathematical formulas
- ✅ Interactive parameter tuning
- ✅ Real-time visualizations
- ✅ Performance metrics and evaluation
- ✅ Advantages and disadvantages
- ✅ Real-world use cases
- ✅ Interpretation guidelines

## 🔧 Requirements

See `requirements.txt` for the complete list of dependencies:
- streamlit>=1.31.0
- scikit-learn>=1.4.0  
- numpy>=1.26.3
- pandas>=2.2.0
- matplotlib>=3.8.2
- seaborn>=0.13.2
- plotly>=5.18.0
- scipy>=1.12.0

## 🤝 Contributing

This project follows PEP8 standards and uses a modular architecture. To add new algorithms:

1. Create a new file in the `algorithms/` directory
2. Follow the existing class structure and interface
3. Include comprehensive documentation and visualizations
4. Add the algorithm to `__init__.py` and `main_app.py`

## 📊 Statistics Covered (Legacy App)

### Descriptive Statistics

| Statistic | Formula | Interpretation |
|-----------|---------|----------------|
| Mean (μ) | Σx / n | Average value |
| Median | Middle value | 50th percentile |
| Mode | Most frequent | Peak of distribution |
| Variance (σ²) | Σ(x - μ)² / (n-1) | Average squared deviation |
| Std Dev (σ) | √Variance | Typical deviation from mean |
| CV (%) | (σ / μ) × 100 | Relative variability |
| IQR | Q3 - Q1 | Middle 50% spread |

### Inferential Statistics

**Z-Score Formula:**
```
Z = (X - μ) / σ
```

**Z-Test Statistic:**
```
Z = (x̄ - μ₀) / (σ / √n)
```

**T-Test Statistic:**
```
t = (x̄ - μ₀) / (s / √n)
```

**Confidence Interval:**
```
CI = x̄ ± t(α/2, df) × (s / √n)
```

## 🎓 Educational Use Cases

### For Students
- Learn statistical concepts through visualization
- Understand machine learning algorithms step by step
- Experiment with parameters and see real-time results
- Build intuition for both statistics and ML

### For Teachers
- Demonstrate concepts in class with interactive examples
- Show real-time parameter effects
- Engage students with hands-on learning
- Compare different algorithms side by side

### For Data Scientists
- Quick exploratory data analysis
- Algorithm comparison and selection
- Feature importance analysis
- Model performance evaluation

## 💡 Machine Learning Tips

### Algorithm Selection Guide

**Use Linear Regression when:**
- Target is continuous
- Relationship appears linear
- Need interpretable coefficients

**Use Logistic Regression when:**
- Binary or multi-class classification
- Need probability estimates
- Linear decision boundary is sufficient

**Use Decision Trees when:**
- Need highly interpretable model
- Data has both numerical and categorical features
- Non-linear relationships expected

**Use Random Forest when:**
- Want better performance than single tree
- Can sacrifice some interpretability
- Have mixed data types

**Use SVM when:**
- High-dimensional data
- Clear margin of separation
- Robust to outliers needed

**Use K-Means when:**
- Need to find natural groupings
- Know approximate number of clusters
- Features are on similar scales

## 📝 Version History

- **v2.0.0**: Complete rewrite with machine learning integration and modular architecture
- **v1.0.0**: Original statistics learning app

## 🔍 Troubleshooting

### App doesn't start
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version: `python --version` (should be 3.8+)
- Try: `python -m streamlit run main_app.py`

### Import errors
- Make sure scikit-learn is installed: `pip install scikit-learn`
- Try upgrading dependencies: `pip install --upgrade -r requirements.txt`

### Performance issues
- Use smaller datasets for real-time interaction
- Some algorithms (like Neural Networks) may take longer to train
- Consider reducing the number of iterations for complex algorithms

---

**Made with ❤️ for data science learners everywhere**

For questions or feedback, please open an issue in the repository.