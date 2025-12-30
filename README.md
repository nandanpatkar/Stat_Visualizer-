# 📊 Statistics Learning App

An interactive web application built with Streamlit to make statistics **visual, interactive, and easy to understand** for learners at all levels. This comprehensive app covers both descriptive and inferential statistics with rich visualizations and hands-on tools.

## 🎯 Features

### Descriptive Statistics
- **Measures of Central Tendency**: Mean, Median, Mode
- **Measures of Dispersion**: Variance, Standard Deviation, Coefficient of Variation (CV), IQR, Range
- **Positional Measures**: Quartiles, Deciles, Percentiles
- **Visualizations**:
  - Histograms with mean/median lines
  - Box plots for quartile analysis
  - Q-Q plots for normality testing
  - Violin plots for distribution shape

### Inferential Statistics
- **Z-Scores**: Calculate and interpret standardized scores
- **Z-Test**: One-sample hypothesis testing with visualization
- **T-Tests**:
  - One-sample t-test
  - Two-sample independent t-test
  - Paired t-test
- **Confidence Intervals**: Calculate and visualize confidence intervals at different levels (90%, 95%, 99%)

### Central Limit Theorem (CLT)
- **Interactive Demonstration**: See CLT in action with different population distributions
- **Multiple Distributions**: Uniform, Exponential, Bimodal, Right-Skewed
- **Adjustable Parameters**: Control sample size and number of samples
- **Real-time Comparison**: Compare population vs sampling distribution
- **Normality Testing**: Automatic statistical tests for normality

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Stat_Visualizer-
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**
   ```bash
   streamlit run app.py
   ```

4. **Open in browser**
   The app will automatically open in your default browser at `http://localhost:8501`

## 📖 How to Use

### Data Input Options

The app provides three ways to input data:

1. **Sample Data (Default)**
   - Pre-loaded normal distribution data (n=1000, μ=100, σ=15)
   - Perfect for learning and experimentation

2. **Upload CSV**
   - Upload your own dataset
   - Select which column to analyze
   - Supports standard CSV format

3. **Manual Entry**
   - Enter custom numbers (comma or space separated)
   - Great for small datasets or homework problems

### Navigation

Use the sidebar to navigate between different topics:

- **Home**: Overview and getting started guide
- **Descriptive Statistics**: Explore your data's characteristics
- **Inferential Statistics**: Perform hypothesis tests and create confidence intervals
- **Central Limit Theorem**: Interactive CLT demonstration
- **About**: App information and documentation

## 📊 Statistics Covered

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
- Verify homework calculations
- Understand when to use different tests
- Build intuition for statistical distributions

### For Teachers
- Demonstrate statistical concepts in class
- Create interactive examples
- Show real-time parameter effects
- Engage students with hands-on learning

### For Data Analysts
- Quick exploratory data analysis
- Test assumptions before analysis
- Visualize distributions
- Verify statistical test results

## 💡 Tips and Best Practices

### Understanding Your Data
1. Always start with **Descriptive Statistics** to understand your data
2. Check the **histogram** and **Q-Q plot** for normality
3. Look at **box plots** to identify outliers
4. Compare **mean vs median** to detect skewness

### Choosing Statistical Tests

**Use Z-Test when:**
- Large sample size (n > 30)
- Population standard deviation is known
- Data is approximately normal

**Use T-Test when:**
- Small sample size (n < 30)
- Population standard deviation is unknown
- Data is approximately normal

**Use Paired T-Test when:**
- Comparing before/after measurements
- Matched pairs or repeated measures

### Central Limit Theorem Exploration
1. Start with a **non-normal distribution** (e.g., Exponential)
2. Use a **small sample size** (n=5) and observe the sampling distribution
3. Gradually **increase sample size** to n=30, n=50, n=100
4. Watch the sampling distribution become **more normal**
5. Understand that this works for **any population distribution**

## 🛠️ Technical Details

### Dependencies
- **streamlit**: Web framework for the app
- **numpy**: Numerical computations
- **pandas**: Data manipulation
- **scipy**: Statistical functions
- **matplotlib**: Static visualizations
- **seaborn**: Statistical visualizations
- **plotly**: Interactive visualizations

### Project Structure
```
Stat_Visualizer-/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## 🎨 Features Highlights

### Interactive Visualizations
- Real-time updates as you change parameters
- Multiple visualization types for each concept
- Color-coded statistical markers
- Detailed annotations and labels

### Educational Content
- Clear explanations for each concept
- Formula displays
- Interpretation guidelines
- Best practices and tips

### User-Friendly Interface
- Clean, intuitive navigation
- Responsive layout
- Mobile-friendly design
- Professional styling

## 📝 Example Workflows

### Workflow 1: Analyzing Survey Data
1. Upload your survey CSV file
2. Select the column to analyze (e.g., "age", "satisfaction_score")
3. View descriptive statistics to understand the distribution
4. Use box plots to identify outliers
5. Calculate confidence intervals for the mean
6. Perform hypothesis test if needed

### Workflow 2: Comparing Two Groups
1. Upload data with two groups
2. Navigate to Inferential Statistics → T-Test
3. Select "Two-Sample T-Test"
4. Examine group distributions
5. Interpret the p-value and conclusion

### Workflow 3: Understanding CLT
1. Navigate to Central Limit Theorem
2. Choose "Exponential" distribution (clearly non-normal)
3. Start with sample size n=5
4. Generate samples and observe the distribution
5. Increase sample size to n=30, then n=50
6. Watch the sampling distribution become normal

## 🔍 Troubleshooting

### App doesn't start
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version: `python --version` (should be 3.8+)
- Try: `python -m streamlit run app.py`

### Data upload issues
- Ensure CSV file is properly formatted
- Check for missing values in the selected column
- Verify column names don't have special characters

### Visualization not showing
- Refresh the page
- Check browser console for errors
- Try a different browser (Chrome recommended)

## 🤝 Contributing

This is an educational project. Suggestions for improvements are welcome!

## 📄 License

This project is created for educational purposes.

## 👨‍💻 Author

Built with ❤️ for statistics learners everywhere.

## 🙏 Acknowledgments

- Statistical formulas and concepts from standard statistics textbooks
- Visualization inspiration from educational statistics resources
- Built with the amazing Streamlit framework

---

**Happy Learning! 📊📈📉**

For questions or feedback, please open an issue in the repository.