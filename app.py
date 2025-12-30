import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from scipy.stats import norm, t as t_dist
import io

# Page configuration
st.set_page_config(
    page_title="Statistics Learning App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
    </style>
    """, unsafe_allow_html=True)

# Helper functions
def get_sample_data():
    """Generate sample data for demonstration"""
    np.random.seed(42)
    return np.random.normal(100, 15, 1000)

def calculate_descriptive_stats(data):
    """Calculate all descriptive statistics"""
    stats_dict = {
        'Mean': np.mean(data),
        'Median': np.median(data),
        'Mode': stats.mode(data, keepdims=True)[0][0] if len(data) > 0 else None,
        'Variance': np.var(data, ddof=1),
        'Standard Deviation': np.std(data, ddof=1),
        'Coefficient of Variation (CV)': (np.std(data, ddof=1) / np.mean(data)) * 100 if np.mean(data) != 0 else None,
        'Range': np.ptp(data),
        'IQR': np.percentile(data, 75) - np.percentile(data, 25),
        'Q1 (25th percentile)': np.percentile(data, 25),
        'Q2 (50th percentile)': np.percentile(data, 50),
        'Q3 (75th percentile)': np.percentile(data, 75),
        'Min': np.min(data),
        'Max': np.max(data),
        'Count': len(data)
    }
    return stats_dict

def plot_distribution(data, title="Data Distribution"):
    """Create distribution plot with multiple visualizations"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Histogram
    axes[0, 0].hist(data, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0, 0].axvline(np.mean(data), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(data):.2f}')
    axes[0, 0].axvline(np.median(data), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(data):.2f}')
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Histogram')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Box plot
    bp = axes[0, 1].boxplot(data, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    axes[0, 1].set_ylabel('Value')
    axes[0, 1].set_title('Box Plot')
    axes[0, 1].grid(True, alpha=0.3)

    # Q-Q plot
    stats.probplot(data, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot (Normality Check)')
    axes[1, 0].grid(True, alpha=0.3)

    # Violin plot
    parts = axes[1, 1].violinplot([data], vert=True, showmeans=True, showmedians=True)
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].set_title('Violin Plot')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

# Main app
def main():
    st.title("📊 Interactive Statistics Learning App")
    st.markdown("### Learn Statistics Visually - From Descriptive to Inferential")

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose a Topic:",
        ["Home", "Descriptive Statistics", "Inferential Statistics", "Central Limit Theorem", "About"]
    )

    # Data input section (common across pages)
    st.sidebar.markdown("---")
    st.sidebar.subheader("📥 Data Input")
    data_source = st.sidebar.selectbox(
        "Select Data Source:",
        ["Sample Data (Normal Distribution)", "Upload CSV", "Manual Entry"]
    )

    data = None

    if data_source == "Sample Data (Normal Distribution)":
        st.sidebar.info("Using randomly generated normal distribution data (n=1000, μ=100, σ=15)")
        data = get_sample_data()

    elif data_source == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=['csv'])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.sidebar.success("File uploaded successfully!")
                column = st.sidebar.selectbox("Select column for analysis:", df.columns)
                data = df[column].dropna().values
            except Exception as e:
                st.sidebar.error(f"Error reading file: {e}")

    elif data_source == "Manual Entry":
        data_input = st.sidebar.text_area(
            "Enter numbers (comma or space separated):",
            "12, 15, 18, 20, 22, 25, 28, 30, 32, 35"
        )
        try:
            data = np.array([float(x.strip()) for x in data_input.replace(',', ' ').split()])
        except:
            st.sidebar.error("Invalid input. Please enter valid numbers.")

    # Page routing
    if page == "Home":
        show_home()
    elif page == "Descriptive Statistics":
        if data is not None and len(data) > 0:
            show_descriptive_statistics(data)
        else:
            st.warning("Please provide data using the sidebar options.")
    elif page == "Inferential Statistics":
        if data is not None and len(data) > 0:
            show_inferential_statistics(data)
        else:
            st.warning("Please provide data using the sidebar options.")
    elif page == "Central Limit Theorem":
        show_central_limit_theorem()
    elif page == "About":
        show_about()

def show_home():
    st.header("Welcome to the Statistics Learning App!")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 What You'll Learn")
        st.markdown("""
        This interactive app helps you understand statistics through visualization and hands-on exploration:

        **Descriptive Statistics:**
        - Measures of Central Tendency (Mean, Median, Mode)
        - Measures of Dispersion (Variance, Standard Deviation, CV, IQR)
        - Quartiles, Deciles, and Percentiles
        - Visual representations (Histograms, Box plots, Q-Q plots)

        **Inferential Statistics:**
        - Z-scores and standardization
        - Z-tests for hypothesis testing
        - T-tests (One-sample, Two-sample, Paired)
        - Confidence Intervals

        **Central Limit Theorem:**
        - Interactive visualization of CLT
        - Understanding sampling distributions
        - Effect of sample size on distribution shape
        """)

    with col2:
        st.subheader("🚀 Getting Started")
        st.markdown("""
        1. **Choose your data source** from the sidebar:
           - Use sample data for quick exploration
           - Upload your own CSV file
           - Enter custom numbers manually

        2. **Navigate** through different topics using the sidebar menu

        3. **Interact** with visualizations and controls to see statistics in action

        4. **Learn** by experimenting with different parameters and datasets
        """)

        st.info("💡 **Tip:** Start with 'Descriptive Statistics' to understand your data, then move to 'Inferential Statistics' for deeper analysis!")

def show_descriptive_statistics(data):
    st.header("📊 Descriptive Statistics")
    st.markdown("Understand your data through summary statistics and visualizations")

    # Calculate statistics
    stats_dict = calculate_descriptive_stats(data)

    # Display data preview
    with st.expander("📋 View Data", expanded=False):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.write(f"**First 20 values:** {data[:20]}")
            st.write(f"**Total observations:** {len(data)}")
        with col2:
            df_preview = pd.DataFrame(data, columns=['Value'])
            st.download_button(
                label="Download Data as CSV",
                data=df_preview.to_csv(index=False),
                file_name="data.csv",
                mime="text/csv"
            )

    # Measures of Central Tendency
    st.subheader("📍 Measures of Central Tendency")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Mean (μ)", f"{stats_dict['Mean']:.2f}")
        st.caption("Average of all values")

    with col2:
        st.metric("Median", f"{stats_dict['Median']:.2f}")
        st.caption("Middle value when sorted")

    with col3:
        st.metric("Mode", f"{stats_dict['Mode']:.2f}" if stats_dict['Mode'] is not None else "N/A")
        st.caption("Most frequent value")

    # Measures of Dispersion
    st.subheader("📏 Measures of Dispersion")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Variance (σ²)", f"{stats_dict['Variance']:.2f}")
        st.caption("Average squared deviation")

    with col2:
        st.metric("Std Deviation (σ)", f"{stats_dict['Standard Deviation']:.2f}")
        st.caption("Square root of variance")

    with col3:
        st.metric("CV (%)", f"{stats_dict['Coefficient of Variation (CV)']:.2f}%" if stats_dict['Coefficient of Variation (CV)'] is not None else "N/A")
        st.caption("Relative variability")

    with col4:
        st.metric("IQR", f"{stats_dict['IQR']:.2f}")
        st.caption("Q3 - Q1")

    # Distribution Plots
    st.subheader("📈 Data Distribution Visualizations")
    fig = plot_distribution(data)
    st.pyplot(fig)
    plt.close()

    # Quartiles, Deciles, Percentiles
    st.subheader("📊 Quartiles, Deciles, and Percentiles")

    tab1, tab2, tab3 = st.tabs(["Quartiles", "Deciles", "Percentiles"])

    with tab1:
        st.markdown("**Quartiles divide the data into 4 equal parts**")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Min", f"{stats_dict['Min']:.2f}")
        col2.metric("Q1 (25%)", f"{stats_dict['Q1 (25th percentile)']:.2f}")
        col3.metric("Q2 (50%)", f"{stats_dict['Q2 (50th percentile)']:.2f}")
        col4.metric("Q3 (75%)", f"{stats_dict['Q3 (75th percentile)']:.2f}")
        col5.metric("Max", f"{stats_dict['Max']:.2f}")

        # Quartile visualization
        fig_q = go.Figure()
        quartiles = [stats_dict['Min'], stats_dict['Q1 (25th percentile)'],
                     stats_dict['Q2 (50th percentile)'], stats_dict['Q3 (75th percentile)'],
                     stats_dict['Max']]
        fig_q.add_trace(go.Bar(
            x=['Min', 'Q1', 'Q2 (Median)', 'Q3', 'Max'],
            y=quartiles,
            marker_color=['blue', 'lightblue', 'green', 'lightblue', 'blue']
        ))
        fig_q.update_layout(title="Quartile Values", xaxis_title="Quartile", yaxis_title="Value", height=400)
        st.plotly_chart(fig_q, use_container_width=True)

    with tab2:
        st.markdown("**Deciles divide the data into 10 equal parts**")
        deciles = [np.percentile(data, i*10) for i in range(11)]
        decile_df = pd.DataFrame({
            'Decile': [f'D{i}' for i in range(11)],
            'Percentile': [f'{i*10}%' for i in range(11)],
            'Value': [f'{d:.2f}' for d in deciles]
        })
        st.dataframe(decile_df, use_container_width=True)

        # Decile visualization
        fig_d = go.Figure()
        fig_d.add_trace(go.Scatter(
            x=[f'D{i}' for i in range(11)],
            y=deciles,
            mode='lines+markers',
            marker=dict(size=10, color='orange'),
            line=dict(color='orange', width=2)
        ))
        fig_d.update_layout(title="Decile Values", xaxis_title="Decile", yaxis_title="Value", height=400)
        st.plotly_chart(fig_d, use_container_width=True)

    with tab3:
        st.markdown("**Custom Percentile Calculator**")
        percentile_value = st.slider("Select Percentile:", 0, 100, 50)
        result = np.percentile(data, percentile_value)
        st.success(f"The {percentile_value}th percentile is: **{result:.2f}**")
        st.caption(f"This means {percentile_value}% of the data falls below {result:.2f}")

        # Show common percentiles
        st.markdown("**Common Percentiles:**")
        common_percentiles = [10, 25, 50, 75, 90, 95, 99]
        percentile_df = pd.DataFrame({
            'Percentile': [f'{p}%' for p in common_percentiles],
            'Value': [f'{np.percentile(data, p):.2f}' for p in common_percentiles]
        })
        st.dataframe(percentile_df, use_container_width=True)

def show_inferential_statistics(data):
    st.header("🔬 Inferential Statistics")
    st.markdown("Make inferences and test hypotheses about your data")

    tab1, tab2, tab3, tab4 = st.tabs(["Z-Scores", "Z-Test", "T-Test", "Confidence Intervals"])

    with tab1:
        st.subheader("📐 Z-Score Analysis")
        st.markdown("""
        **Z-Score** measures how many standard deviations a value is from the mean.

        Formula: `Z = (X - μ) / σ`

        - Z = 0: Value is equal to the mean
        - Z > 0: Value is above the mean
        - Z < 0: Value is below the mean
        - |Z| > 2: Value is considered unusual (outside 95% of data)
        """)

        col1, col2 = st.columns(2)

        with col1:
            value_for_z = st.number_input("Enter a value to calculate Z-score:",
                                          value=float(np.mean(data)))

            mean = np.mean(data)
            std = np.std(data, ddof=1)
            z_score = (value_for_z - mean) / std

            st.metric("Z-Score", f"{z_score:.4f}")

            if abs(z_score) > 2:
                st.warning(f"⚠️ This value is unusual (|Z| > 2)")
            else:
                st.success(f"✅ This value is within normal range")

            st.info(f"The value {value_for_z:.2f} is {abs(z_score):.2f} standard deviations {'above' if z_score > 0 else 'below'} the mean.")

        with col2:
            # Calculate Z-scores for all data
            z_scores = (data - mean) / std

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(z_scores, bins=30, edgecolor='black', alpha=0.7, color='lightcoral')
            ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Mean (Z=0)')
            ax.axvline(-2, color='orange', linestyle='--', linewidth=1, label='Z = -2')
            ax.axvline(2, color='orange', linestyle='--', linewidth=1, label='Z = 2')
            ax.axvline(z_score, color='green', linestyle='-', linewidth=2, label=f'Your value (Z={z_score:.2f})')
            ax.set_xlabel('Z-Score')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Z-Scores')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()

    with tab2:
        st.subheader("🧪 Z-Test (One-Sample)")
        st.markdown("""
        **Z-Test** is used to determine if a sample mean is significantly different from a known population mean.

        **Assumptions:**
        - Large sample size (n > 30) OR known population standard deviation
        - Data is normally distributed (or n > 30 by CLT)
        """)

        col1, col2 = st.columns(2)

        with col1:
            pop_mean = st.number_input("Population mean (μ₀) to test against:",
                                       value=100.0)
            alpha = st.select_slider("Significance level (α):",
                                    options=[0.01, 0.05, 0.10],
                                    value=0.05)

            sample_mean = np.mean(data)
            sample_std = np.std(data, ddof=1)
            n = len(data)

            # Calculate Z-statistic
            z_stat = (sample_mean - pop_mean) / (sample_std / np.sqrt(n))

            # Critical value (two-tailed)
            z_critical = norm.ppf(1 - alpha/2)

            # P-value (two-tailed)
            p_value = 2 * (1 - norm.cdf(abs(z_stat)))

            st.markdown("### Results:")
            st.metric("Sample Mean", f"{sample_mean:.4f}")
            st.metric("Z-Statistic", f"{z_stat:.4f}")
            st.metric("P-Value", f"{p_value:.4f}")
            st.metric("Critical Value (±)", f"{z_critical:.4f}")

            if p_value < alpha:
                st.error(f"❌ **Reject H₀**: The sample mean is significantly different from {pop_mean} (p < {alpha})")
            else:
                st.success(f"✅ **Fail to Reject H₀**: No significant difference from {pop_mean} (p ≥ {alpha})")

        with col2:
            # Visualization
            x = np.linspace(-4, 4, 1000)
            y = norm.pdf(x)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(x, y, 'b-', linewidth=2, label='Standard Normal Distribution')

            # Critical regions
            ax.fill_between(x[x <= -z_critical], y[x <= -z_critical], alpha=0.3, color='red', label='Rejection Region')
            ax.fill_between(x[x >= z_critical], y[x >= z_critical], alpha=0.3, color='red')

            # Z-statistic line
            ax.axvline(z_stat, color='green', linestyle='--', linewidth=2, label=f'Z-stat = {z_stat:.2f}')
            ax.axvline(-z_critical, color='orange', linestyle='--', linewidth=1)
            ax.axvline(z_critical, color='orange', linestyle='--', linewidth=1)

            ax.set_xlabel('Z-Score')
            ax.set_ylabel('Probability Density')
            ax.set_title('Z-Test Visualization')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()

    with tab3:
        st.subheader("📊 T-Test")

        test_type = st.selectbox("Select T-Test Type:",
                                 ["One-Sample T-Test", "Two-Sample T-Test (Independent)", "Paired T-Test"])

        if test_type == "One-Sample T-Test":
            st.markdown("**One-Sample T-Test**: Compare sample mean to a known value")

            col1, col2 = st.columns(2)

            with col1:
                pop_mean_t = st.number_input("Population mean to test against:", value=100.0, key='ttest_mean')
                alpha_t = st.select_slider("Significance level:",
                                          options=[0.01, 0.05, 0.10],
                                          value=0.05, key='ttest_alpha')

                t_stat, p_value_t = stats.ttest_1samp(data, pop_mean_t)
                df = len(data) - 1
                t_critical = t_dist.ppf(1 - alpha_t/2, df)

                st.markdown("### Results:")
                st.metric("Sample Mean", f"{np.mean(data):.4f}")
                st.metric("T-Statistic", f"{t_stat:.4f}")
                st.metric("P-Value", f"{p_value_t:.4f}")
                st.metric("Degrees of Freedom", df)
                st.metric("Critical Value (±)", f"{t_critical:.4f}")

                if p_value_t < alpha_t:
                    st.error(f"❌ **Reject H₀**: Significant difference (p < {alpha_t})")
                else:
                    st.success(f"✅ **Fail to Reject H₀**: No significant difference (p ≥ {alpha_t})")

            with col2:
                x_t = np.linspace(-4, 4, 1000)
                y_t = t_dist.pdf(x_t, df)

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(x_t, y_t, 'b-', linewidth=2, label=f't-distribution (df={df})')
                ax.fill_between(x_t[x_t <= -t_critical], y_t[x_t <= -t_critical], alpha=0.3, color='red')
                ax.fill_between(x_t[x_t >= t_critical], y_t[x_t >= t_critical], alpha=0.3, color='red')
                ax.axvline(t_stat, color='green', linestyle='--', linewidth=2, label=f't-stat = {t_stat:.2f}')
                ax.axvline(-t_critical, color='orange', linestyle='--', linewidth=1)
                ax.axvline(t_critical, color='orange', linestyle='--', linewidth=1)
                ax.set_xlabel('T-Score')
                ax.set_ylabel('Probability Density')
                ax.set_title('One-Sample T-Test Visualization')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()

        elif test_type == "Two-Sample T-Test (Independent)":
            st.markdown("**Two-Sample T-Test**: Compare means of two independent groups")
            st.info("For demonstration, we'll split your data into two random groups")

            # Split data randomly
            np.random.shuffle(data)
            mid = len(data) // 2
            group1 = data[:mid]
            group2 = data[mid:]

            col1, col2 = st.columns(2)

            with col1:
                alpha_t2 = st.select_slider("Significance level:",
                                           options=[0.01, 0.05, 0.10],
                                           value=0.05, key='ttest2_alpha')

                t_stat2, p_value_t2 = stats.ttest_ind(group1, group2)

                st.markdown("### Results:")
                st.metric("Group 1 Mean", f"{np.mean(group1):.4f}")
                st.metric("Group 2 Mean", f"{np.mean(group2):.4f}")
                st.metric("Mean Difference", f"{np.mean(group1) - np.mean(group2):.4f}")
                st.metric("T-Statistic", f"{t_stat2:.4f}")
                st.metric("P-Value", f"{p_value_t2:.4f}")

                if p_value_t2 < alpha_t2:
                    st.error(f"❌ **Reject H₀**: Groups have significantly different means")
                else:
                    st.success(f"✅ **Fail to Reject H₀**: No significant difference between groups")

            with col2:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(group1, bins=20, alpha=0.5, label='Group 1', color='blue', edgecolor='black')
                ax.hist(group2, bins=20, alpha=0.5, label='Group 2', color='red', edgecolor='black')
                ax.axvline(np.mean(group1), color='blue', linestyle='--', linewidth=2)
                ax.axvline(np.mean(group2), color='red', linestyle='--', linewidth=2)
                ax.set_xlabel('Value')
                ax.set_ylabel('Frequency')
                ax.set_title('Distribution of Two Groups')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()

        else:  # Paired T-Test
            st.markdown("**Paired T-Test**: Compare paired observations (before/after, matched pairs)")
            st.info("For demonstration, we'll create 'before' and 'after' measurements by adding random noise")

            before = data[:len(data)//2]
            after = before + np.random.normal(2, 5, len(before))  # Add effect with noise

            col1, col2 = st.columns(2)

            with col1:
                alpha_tp = st.select_slider("Significance level:",
                                           options=[0.01, 0.05, 0.10],
                                           value=0.05, key='ttestp_alpha')

                t_statp, p_value_tp = stats.ttest_rel(before, after)
                differences = after - before

                st.markdown("### Results:")
                st.metric("Mean Before", f"{np.mean(before):.4f}")
                st.metric("Mean After", f"{np.mean(after):.4f}")
                st.metric("Mean Difference", f"{np.mean(differences):.4f}")
                st.metric("T-Statistic", f"{t_statp:.4f}")
                st.metric("P-Value", f"{p_value_tp:.4f}")

                if p_value_tp < alpha_tp:
                    st.error(f"❌ **Reject H₀**: Significant difference between paired observations")
                else:
                    st.success(f"✅ **Fail to Reject H₀**: No significant difference")

            with col2:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(before, after, alpha=0.6, s=50)
                ax.plot([before.min(), before.max()], [before.min(), before.max()],
                       'r--', linewidth=2, label='No Change Line')
                ax.set_xlabel('Before')
                ax.set_ylabel('After')
                ax.set_title('Paired Observations: Before vs After')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()

    with tab4:
        st.subheader("📏 Confidence Intervals")
        st.markdown("""
        **Confidence Interval** provides a range of values that likely contains the true population parameter.

        A 95% CI means: If we repeated this study 100 times, about 95 of the intervals would contain the true population mean.
        """)

        col1, col2 = st.columns(2)

        with col1:
            confidence_level = st.select_slider(
                "Confidence Level:",
                options=[0.90, 0.95, 0.99],
                value=0.95,
                format_func=lambda x: f"{int(x*100)}%"
            )

            mean = np.mean(data)
            std_err = stats.sem(data)
            df_ci = len(data) - 1

            # Calculate CI using t-distribution
            ci = t_dist.interval(confidence_level, df_ci, loc=mean, scale=std_err)
            margin_of_error = ci[1] - mean

            st.markdown("### Results:")
            st.metric("Sample Mean", f"{mean:.4f}")
            st.metric("Standard Error", f"{std_err:.4f}")
            st.metric("Margin of Error", f"± {margin_of_error:.4f}")

            st.success(f"""
            **{int(confidence_level*100)}% Confidence Interval:**

            [{ci[0]:.4f}, {ci[1]:.4f}]

            We are {int(confidence_level*100)}% confident that the true population mean lies within this range.
            """)

            # Show different confidence levels
            st.markdown("### Compare Confidence Levels:")
            for cl in [0.90, 0.95, 0.99]:
                ci_temp = t_dist.interval(cl, df_ci, loc=mean, scale=std_err)
                width = ci_temp[1] - ci_temp[0]
                st.write(f"**{int(cl*100)}% CI:** [{ci_temp[0]:.2f}, {ci_temp[1]:.2f}] (width: {width:.2f})")

        with col2:
            # Visualization
            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot different confidence levels
            levels = [0.90, 0.95, 0.99]
            colors = ['lightblue', 'blue', 'darkblue']

            for i, (cl, color) in enumerate(zip(levels, colors)):
                ci_temp = t_dist.interval(cl, df_ci, loc=mean, scale=std_err)
                ax.barh(i, ci_temp[1] - ci_temp[0], left=ci_temp[0], height=0.3,
                       color=color, alpha=0.6, label=f'{int(cl*100)}% CI')
                ax.plot([ci_temp[0], ci_temp[1]], [i, i], 'k-', linewidth=2)
                ax.plot(ci_temp[0], i, 'k|', markersize=15, markeredgewidth=2)
                ax.plot(ci_temp[1], i, 'k|', markersize=15, markeredgewidth=2)

            ax.axvline(mean, color='red', linestyle='--', linewidth=2, label='Sample Mean')
            ax.set_yticks(range(len(levels)))
            ax.set_yticklabels([f'{int(cl*100)}%' for cl in levels])
            ax.set_xlabel('Value')
            ax.set_ylabel('Confidence Level')
            ax.set_title('Confidence Intervals Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='x')
            st.pyplot(fig)
            plt.close()

            st.caption("Notice: Higher confidence level → Wider interval")

def show_central_limit_theorem():
    st.header("🎯 Central Limit Theorem (CLT)")
    st.markdown("""
    The **Central Limit Theorem** states that the distribution of sample means approaches a normal distribution
    as the sample size increases, regardless of the population's distribution.

    This is one of the most important concepts in statistics!
    """)

    st.subheader("Interactive CLT Demonstration")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Settings")

        population_dist = st.selectbox(
            "Population Distribution:",
            ["Uniform", "Exponential", "Bimodal", "Right-Skewed"]
        )

        sample_size = st.slider("Sample Size (n):", 1, 100, 30)
        num_samples = st.slider("Number of Samples:", 100, 10000, 1000, step=100)

        if st.button("🎲 Generate Samples", type="primary"):
            st.session_state.clt_generated = True

            # Generate population based on selection
            np.random.seed(42)
            if population_dist == "Uniform":
                population = np.random.uniform(0, 10, 100000)
            elif population_dist == "Exponential":
                population = np.random.exponential(2, 100000)
            elif population_dist == "Bimodal":
                pop1 = np.random.normal(3, 1, 50000)
                pop2 = np.random.normal(7, 1, 50000)
                population = np.concatenate([pop1, pop2])
            else:  # Right-Skewed
                population = np.random.chisquare(3, 100000)

            # Generate sample means
            sample_means = []
            for _ in range(num_samples):
                sample = np.random.choice(population, size=sample_size, replace=True)
                sample_means.append(np.mean(sample))

            st.session_state.population = population
            st.session_state.sample_means = np.array(sample_means)

    with col2:
        if 'clt_generated' in st.session_state and st.session_state.clt_generated:
            population = st.session_state.population
            sample_means = st.session_state.sample_means

            # Create visualization
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))

            # Population distribution
            axes[0].hist(population, bins=50, edgecolor='black', alpha=0.7, color='lightcoral', density=True)
            axes[0].axvline(np.mean(population), color='red', linestyle='--', linewidth=2,
                          label=f'Population Mean = {np.mean(population):.2f}')
            axes[0].set_xlabel('Value')
            axes[0].set_ylabel('Density')
            axes[0].set_title(f'Original Population Distribution ({population_dist})')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Sampling distribution of means
            axes[1].hist(sample_means, bins=50, edgecolor='black', alpha=0.7, color='lightblue', density=True)
            axes[1].axvline(np.mean(sample_means), color='blue', linestyle='--', linewidth=2,
                          label=f'Mean of Sample Means = {np.mean(sample_means):.2f}')

            # Overlay normal distribution
            mu = np.mean(sample_means)
            sigma = np.std(sample_means)
            x = np.linspace(sample_means.min(), sample_means.max(), 100)
            axes[1].plot(x, norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal Distribution Fit')

            axes[1].set_xlabel('Sample Mean')
            axes[1].set_ylabel('Density')
            axes[1].set_title(f'Sampling Distribution of Means (n={sample_size}, samples={num_samples})')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Statistics comparison
            st.markdown("### 📊 Statistics Comparison")
            col_a, col_b, col_c = st.columns(3)

            with col_a:
                st.metric("Population Mean", f"{np.mean(population):.4f}")
                st.metric("Sample Means Mean", f"{np.mean(sample_means):.4f}")

            with col_b:
                st.metric("Population Std Dev", f"{np.std(population):.4f}")
                st.metric("Sample Means Std Dev", f"{np.std(sample_means):.4f}")
                theoretical_se = np.std(population) / np.sqrt(sample_size)
                st.caption(f"Theoretical SE: {theoretical_se:.4f}")

            with col_c:
                # Test for normality using Shapiro-Wilk test
                if len(sample_means) <= 5000:
                    _, p_value_norm = stats.shapiro(sample_means)
                    st.metric("Normality Test (p-value)", f"{p_value_norm:.4f}")
                    if p_value_norm > 0.05:
                        st.success("✅ Approximately Normal")
                    else:
                        st.warning("⚠️ May not be normal yet")

            st.info(f"""
            **Key Observations:**
            - The mean of sample means ≈ population mean
            - Standard error = population std / √n = {theoretical_se:.4f}
            - As sample size (n) increases, the distribution becomes more normal
            - As number of samples increases, the approximation improves
            """)
        else:
            st.info("👆 Click 'Generate Samples' to see the CLT in action!")

            # Show explanation
            st.markdown("""
            ### What to expect:

            1. **Top plot**: Shows the original population distribution (can be any shape)
            2. **Bottom plot**: Shows the distribution of sample means (approaches normal)

            ### Try this:
            - Start with a small sample size (n=5) and gradually increase it
            - Notice how the bottom plot becomes more bell-shaped
            - Compare different population distributions
            - The magic of CLT: Even weird distributions → Normal sampling distribution!
            """)

    # Additional educational content
    st.markdown("---")
    st.subheader("📚 Understanding CLT")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **Why is CLT important?**

        - Enables hypothesis testing
        - Justifies using normal distribution
        - Works for ANY population distribution
        - Foundation for confidence intervals
        """)

    with col2:
        st.markdown("""
        **Key Requirements:**

        - Independent random samples
        - Sample size typically n ≥ 30
        - Larger n = better approximation
        - Smaller n works if population is normal
        """)

    with col3:
        st.markdown("""
        **Real-world Applications:**

        - Quality control in manufacturing
        - Medical research studies
        - Political polling
        - A/B testing in tech
        """)

def show_about():
    st.header("ℹ️ About This App")

    st.markdown("""
    ## 📊 Statistics Learning App

    This interactive application is designed to make statistics **visual, interactive, and easy to understand**
    for learners at all levels.

    ### 🎯 Features

    **Descriptive Statistics:**
    - Measures of central tendency (mean, median, mode)
    - Measures of dispersion (variance, standard deviation, CV, IQR)
    - Quartiles, deciles, and percentiles
    - Interactive visualizations (histograms, box plots, Q-Q plots, violin plots)

    **Inferential Statistics:**
    - Z-score calculation and interpretation
    - Z-tests for hypothesis testing
    - T-tests (one-sample, two-sample, paired)
    - Confidence interval estimation

    **Central Limit Theorem:**
    - Interactive visualization of CLT
    - Multiple population distributions
    - Adjustable sample sizes and number of samples
    - Real-time statistical comparisons

    ### 🛠️ Built With
    - **Streamlit** - Web framework
    - **NumPy** - Numerical computing
    - **Pandas** - Data manipulation
    - **SciPy** - Statistical functions
    - **Matplotlib & Seaborn** - Static visualizations
    - **Plotly** - Interactive visualizations

    ### 📖 How to Use

    1. **Select Data Source**: Choose from sample data, upload your CSV, or enter custom numbers
    2. **Navigate Topics**: Use the sidebar to explore different statistical concepts
    3. **Interact**: Adjust parameters and see real-time updates
    4. **Learn**: Read explanations and interpret results

    ### 💡 Tips for Learning

    - Start with sample data to understand each concept
    - Try different datasets to see how statistics change
    - Experiment with parameters in hypothesis tests
    - Use the CLT demonstration to build intuition
    - Compare results across different statistical tests

    ### 🎓 Educational Goals

    This app aims to:
    - Demystify statistical concepts through visualization
    - Provide hands-on experience with real calculations
    - Build intuition for when to use different tests
    - Make statistics accessible and engaging

    ### 📝 Version
    Version 1.0.0 - Complete Statistics Learning Platform

    ---

    **Made with ❤️ for statistics learners everywhere**
    """)

    st.balloons()

if __name__ == "__main__":
    main()
