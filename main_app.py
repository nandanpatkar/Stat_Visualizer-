"""
Main Application - Statistics Learning App with Machine Learning

This is the main entry point for the Statistics Learning App with integrated
Machine Learning algorithms. The application follows PEP8 standards and
uses a modular architecture.

Author: Statistics Learning App
Version: 2.0.0
"""

import streamlit as st
import sys
import os

# Configure Streamlit page settings FIRST (before any other Streamlit commands)
st.set_page_config(
    page_title="Statistics & ML Learning App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import algorithm modules with graceful error handling
ALGORITHMS_AVAILABLE = False
STATISTICS_AVAILABLE = False

try:
    # Test if dependencies are available
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import sklearn
    
    # If dependencies are available, import algorithm modules
    from algorithms.linear_regression import LinearRegression
    from algorithms.logistic_regression import LogisticRegression
    from algorithms.decision_tree import DecisionTree
    from algorithms.random_forest import RandomForest
    from algorithms.k_means import KMeans
    from algorithms.k_nearest_neighbors import KNearestNeighbors
    from algorithms.support_vector_machine import SupportVectorMachine
    from algorithms.naive_bayes import NaiveBayes
    from algorithms.gradient_boosting import GradientBoosting
    from algorithms.neural_network import NeuralNetwork
    from stat_analysis.descriptive_stats import DescriptiveStatistics
    
    ALGORITHMS_AVAILABLE = True
    STATISTICS_AVAILABLE = True
    
except ImportError as e:
    # If dependencies are missing, show helpful error message
    st.error("📦 **Dependencies Missing!**")
    st.markdown(f"""
    **Error:** {e}
    
    To run this application, please install the required dependencies:
    
    ```bash
    pip install -r requirements.txt
    ```
    
    **Required packages:**
    - streamlit
    - numpy  
    - pandas
    - matplotlib
    - seaborn
    - plotly
    - scipy
    - scikit-learn
    
    After installing dependencies, restart the application with:
    ```bash
    streamlit run main_app.py
    ```
    """)
    st.info("💡 **Preview Mode:** You can still view the application structure and documentation below!")
    
    # Define placeholder classes for demonstration
    class PlaceholderAlgorithm:
        def streamlit_interface(self):
            st.warning("⚠️ Dependencies not installed. Please install requirements.txt to use this algorithm.")
    
    LinearRegression = PlaceholderAlgorithm
    LogisticRegression = PlaceholderAlgorithm
    DecisionTree = PlaceholderAlgorithm
    RandomForest = PlaceholderAlgorithm
    KMeans = PlaceholderAlgorithm
    KNearestNeighbors = PlaceholderAlgorithm
    SupportVectorMachine = PlaceholderAlgorithm
    NaiveBayes = PlaceholderAlgorithm
    GradientBoosting = PlaceholderAlgorithm
    NeuralNetwork = PlaceholderAlgorithm
    
    class PlaceholderStats:
        def streamlit_interface(self, data):
            st.warning("⚠️ Dependencies not installed. Please install requirements.txt to use statistics features.")
    
    DescriptiveStatistics = PlaceholderStats


class StatisticsMLApp:
    """
    Main application class for the Statistics and Machine Learning App.
    
    This class orchestrates the entire application, managing navigation,
    data input, and algorithm selection following PEP8 standards.
    """
    
    def __init__(self):
        """Initialize the main application."""
        self.apply_custom_css()
    
    def apply_custom_css(self):
        """Apply custom CSS styling."""
        st.markdown("""
            <style>
            .main {
                padding: 0rem 1rem;
            }
            .stAlert {
                margin-top: 1rem;
            }
            h1 {
                color: #1f77b4;
            }
            h2 {
                color: #ff7f0e;
            }
            h3 {
                color: #2ca02c;
            }
            .stat-card {
                background-color: #f0f2f6;
                padding: 20px;
                border-radius: 10px;
                margin: 10px 0;
            }
            .algorithm-card {
                border: 2px solid #e0e0e0;
                border-radius: 10px;
                padding: 15px;
                margin: 10px 0;
                background-color: #fafafa;
            }
            </style>
            """, unsafe_allow_html=True)
    
    def get_sample_data(self):
        """Generate sample data for demonstration."""
        if ALGORITHMS_AVAILABLE:
            import numpy as np
            np.random.seed(42)
            return np.random.normal(100, 15, 1000)
        else:
            # Return simple Python list when numpy not available
            import random
            random.seed(42)
            return [random.gauss(100, 15) for _ in range(1000)]
    
    def handle_data_input(self):
        """
        Handle data input from various sources.
        
        Returns:
            np.array or None: Processed data array
        """
        st.sidebar.markdown("---")
        st.sidebar.subheader("📥 Data Input")
        
        data_source = st.sidebar.selectbox(
            "Select Data Source:",
            ["Sample Data (Normal Distribution)", "Upload CSV", "Manual Entry"]
        )

        data = None

        if data_source == "Sample Data (Normal Distribution)":
            st.sidebar.info("Using randomly generated normal distribution data (n=1000, μ=100, σ=15)")
            data = self.get_sample_data()

        elif data_source == "Upload CSV":
            uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=['csv'])
            if uploaded_file is not None:
                if ALGORITHMS_AVAILABLE:
                    try:
                        import pandas as pd
                        df = pd.read_csv(uploaded_file)
                        st.sidebar.success("File uploaded successfully!")
                        column = st.sidebar.selectbox("Select column for analysis:", df.columns)
                        data = df[column].dropna().values
                    except Exception as e:
                        st.sidebar.error(f"Error reading file: {e}")
                else:
                    st.sidebar.error("CSV upload requires pandas. Please install dependencies.")

        elif data_source == "Manual Entry":
            data_input = st.sidebar.text_area(
                "Enter numbers (comma or space separated):",
                "12, 15, 18, 20, 22, 25, 28, 30, 32, 35"
            )
            try:
                data_list = [float(x.strip()) for x in data_input.replace(',', ' ').split()]
                if ALGORITHMS_AVAILABLE:
                    import numpy as np
                    data = np.array(data_list)
                else:
                    data = data_list
            except:
                st.sidebar.error("Invalid input. Please enter valid numbers.")

        return data
    
    def show_home_page(self):
        """Display the home page."""
        st.header("Welcome to the Statistics & Machine Learning App!")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📈 What You'll Learn")
            st.markdown("""
            This interactive app helps you understand both statistics and machine learning 
            through visualization and hands-on exploration:

            **📊 Statistics:**
            - Descriptive Statistics (Mean, Median, Mode, Variance, etc.)
            - Inferential Statistics (Z-tests, T-tests, Confidence Intervals)
            - Central Limit Theorem visualization

            **🤖 Machine Learning Algorithms:**
            - **Supervised Learning:** Linear/Logistic Regression, Decision Trees, 
              Random Forest, SVM, Naive Bayes, KNN, Neural Networks
            - **Unsupervised Learning:** K-Means Clustering
            - **Ensemble Methods:** Random Forest, Gradient Boosting
            
            **🎯 Key Features:**
            - Interactive parameter tuning
            - Real-time visualizations
            - Theoretical explanations
            - Performance metrics
            """)

        with col2:
            st.subheader("🚀 Getting Started")
            st.markdown("""
            1. **Choose your data source** from the sidebar:
               - Use sample data for quick exploration
               - Upload your own CSV file
               - Enter custom numbers manually

            2. **Select a topic** from the navigation menu:
               - Start with **Statistics** to understand your data
               - Explore **Machine Learning** algorithms
               - Experiment with different parameters

            3. **Learn by doing:**
               - Adjust parameters and see results change
               - Read algorithm explanations
               - Interpret performance metrics

            4. **Compare algorithms:**
               - Try different ML algorithms on the same data
               - Understand when to use each algorithm
            """)

            st.info("💡 **Tip:** Start with 'Statistics' to understand your data, then explore machine learning algorithms!")
    
    def show_statistics_section(self, data):
        """
        Display statistics section with original functionality.
        
        Args:
            data (np.array): Input data for analysis
        """
        if data is None or len(data) == 0:
            st.warning("Please provide data using the sidebar options.")
            return
            
        # Import and use descriptive statistics
        desc_stats = DescriptiveStatistics()
        desc_stats.streamlit_interface(data)
        
        # Add note about the original inferential statistics
        st.markdown("---")
        st.info("🔄 **Note:** Advanced inferential statistics and Central Limit Theorem "
                "from the original app will be integrated in the next update!")
    
    def show_machine_learning_section(self):
        """Display machine learning algorithms section."""
        st.header("🤖 Machine Learning Algorithms")
        st.markdown("### Explore and learn different machine learning algorithms interactively")
        
        # Algorithm categories
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🎯 Supervised Learning**
            - Linear Regression
            - Logistic Regression  
            - Decision Trees
            - Random Forest
            - Support Vector Machine
            - Naive Bayes
            - K-Nearest Neighbors
            - Neural Networks
            - Gradient Boosting
            """)
            
        with col2:
            st.markdown("""
            **🔍 Unsupervised Learning**
            - K-Means Clustering
            - (More algorithms coming soon!)
            """)
            
        with col3:
            st.markdown("""
            **📊 Algorithm Comparison**
            - Performance metrics
            - Visualization tools
            - Parameter sensitivity
            - Use case recommendations
            """)
        
        # Algorithm selection
        st.markdown("### Select an Algorithm to Explore")
        
        algorithm_options = {
            "Linear Regression": LinearRegression,
            "Logistic Regression": LogisticRegression,
            "Decision Tree": DecisionTree,
            "Random Forest": RandomForest,
            "K-Means Clustering": KMeans,
            "K-Nearest Neighbors": KNearestNeighbors,
            "Support Vector Machine": SupportVectorMachine,
            "Naive Bayes": NaiveBayes,
            "Gradient Boosting": GradientBoosting,
            "Neural Network": NeuralNetwork
        }
        
        selected_algorithm = st.selectbox(
            "Choose an algorithm:",
            list(algorithm_options.keys())
        )
        
        if selected_algorithm:
            st.markdown("---")
            
            # Instantiate and run the selected algorithm
            try:
                algorithm_class = algorithm_options[selected_algorithm]
                algorithm_instance = algorithm_class()
                algorithm_instance.streamlit_interface()
                
            except Exception as e:
                st.error(f"Error running {selected_algorithm}: {str(e)}")
                st.info("This algorithm is still being developed. Please try another one!")
    
    def show_about_page(self):
        """Display the about page."""
        st.header("ℹ️ About This App")

        st.markdown("""
        ## 📊 Statistics & Machine Learning Learning App

        This interactive application combines **statistical analysis** with **machine learning**
        to provide a comprehensive learning platform for data science concepts.

        ### 🎯 New Features (Version 2.0)

        **🤖 Machine Learning Algorithms:**
        - 10+ interactive ML algorithms with full explanations
        - Real-time parameter tuning and visualization
        - Performance metrics and model comparison
        - Educational theory sections for each algorithm

        **📊 Enhanced Statistics:**
        - Modular, PEP8-compliant code structure
        - Improved visualizations and interactivity
        - Better error handling and user experience

        **🏗️ Technical Improvements:**
        - Modular architecture with separate algorithm files
        - Following PEP8 coding standards
        - Better separation of concerns
        - Extensible design for adding new algorithms

        ### 🛠️ Built With
        - **Streamlit** - Web framework
        - **Scikit-learn** - Machine learning library
        - **NumPy** - Numerical computing
        - **Pandas** - Data manipulation
        - **SciPy** - Statistical functions
        - **Matplotlib & Seaborn** - Static visualizations
        - **Plotly** - Interactive visualizations

        ### 📖 How to Use

        1. **Select Data Source**: Upload CSV, use sample data, or enter custom numbers
        2. **Choose Section**: Statistics for data analysis, ML for algorithm exploration
        3. **Interact**: Adjust parameters and see real-time updates
        4. **Learn**: Read theoretical explanations and interpret results

        ### 📝 Version Information
        - **Version 2.0.0** - Complete Machine Learning Integration
        - **Architecture**: Modular, PEP8-compliant
        - **Algorithms**: 10+ ML algorithms with interactive interfaces

        ---

        **Made with ❤️ for data science learners everywhere**
        """)

        st.balloons()
    
    def run(self):
        """Run the main application."""
        st.title("📊 Interactive Statistics & ML Learning App")
        st.markdown("### Learn Statistics and Machine Learning Visually")
        
        # Show dependency status
        if ALGORITHMS_AVAILABLE:
            st.success("✅ All dependencies loaded - Full functionality available!")
        else:
            st.warning("⚠️ Preview mode - Please install dependencies for full functionality")

        # Sidebar navigation
        st.sidebar.title("Navigation")
        page = st.sidebar.radio(
            "Choose a Section:",
            ["🏠 Home", "📊 Statistics", "🤖 Machine Learning", "ℹ️ About"]
        )

        # Data input (shared across pages that need it)
        data = None
        if page in ["📊 Statistics"]:
            data = self.handle_data_input()

        # Page routing
        if page == "🏠 Home":
            self.show_home_page()
        elif page == "📊 Statistics":
            self.show_statistics_section(data)
        elif page == "🤖 Machine Learning":
            self.show_machine_learning_section()
        elif page == "ℹ️ About":
            self.show_about_page()


def main():
    """Main application entry point."""
    app = StatisticsMLApp()
    app.run()


if __name__ == "__main__":
    main()