"""
Descriptive Statistics Module

Contains all functions and visualizations for descriptive statistical analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy import stats
import streamlit as st


class DescriptiveStatistics:
    """
    Class containing methods for descriptive statistical analysis.
    
    This class provides comprehensive descriptive statistics calculations
    and visualizations following PEP8 standards.
    """
    
    @staticmethod
    def calculate_descriptive_stats(data):
        """
        Calculate all descriptive statistics.
        
        Args:
            data (array-like): Input data for statistical analysis
            
        Returns:
            dict: Dictionary containing all descriptive statistics
        """
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
    
    @staticmethod
    def plot_distribution(data, title="Data Distribution"):
        """
        Create distribution plot with multiple visualizations.
        
        Args:
            data (array-like): Input data to plot
            title (str): Title for the plot
            
        Returns:
            matplotlib.figure.Figure: Figure object with distribution plots
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Histogram
        axes[0, 0].hist(data, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        axes[0, 0].axvline(np.mean(data), color='red', linestyle='--', linewidth=2, 
                          label=f'Mean: {np.mean(data):.2f}')
        axes[0, 0].axvline(np.median(data), color='green', linestyle='--', linewidth=2, 
                          label=f'Median: {np.median(data):.2f}')
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
    
    @staticmethod
    def create_quartile_visualization(data):
        """
        Create interactive quartile visualization.
        
        Args:
            data (array-like): Input data
            
        Returns:
            plotly.graph_objects.Figure: Interactive quartile plot
        """
        stats_dict = DescriptiveStatistics.calculate_descriptive_stats(data)
        
        fig = go.Figure()
        quartiles = [
            stats_dict['Min'], 
            stats_dict['Q1 (25th percentile)'],
            stats_dict['Q2 (50th percentile)'], 
            stats_dict['Q3 (75th percentile)'],
            stats_dict['Max']
        ]
        
        fig.add_trace(go.Bar(
            x=['Min', 'Q1', 'Q2 (Median)', 'Q3', 'Max'],
            y=quartiles,
            marker_color=['blue', 'lightblue', 'green', 'lightblue', 'blue']
        ))
        
        fig.update_layout(
            title="Quartile Values", 
            xaxis_title="Quartile", 
            yaxis_title="Value", 
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_decile_visualization(data):
        """
        Create decile visualization.
        
        Args:
            data (array-like): Input data
            
        Returns:
            tuple: (DataFrame with decile data, plotly Figure)
        """
        deciles = [np.percentile(data, i*10) for i in range(11)]
        
        decile_df = pd.DataFrame({
            'Decile': [f'D{i}' for i in range(11)],
            'Percentile': [f'{i*10}%' for i in range(11)],
            'Value': [f'{d:.2f}' for d in deciles]
        })
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[f'D{i}' for i in range(11)],
            y=deciles,
            mode='lines+markers',
            marker=dict(size=10, color='orange'),
            line=dict(color='orange', width=2)
        ))
        
        fig.update_layout(
            title="Decile Values", 
            xaxis_title="Decile", 
            yaxis_title="Value", 
            height=400
        )
        
        return decile_df, fig
    
    @staticmethod
    def create_percentile_table(data, percentiles=None):
        """
        Create percentile lookup table.
        
        Args:
            data (array-like): Input data
            percentiles (list): List of percentiles to calculate
            
        Returns:
            pd.DataFrame: DataFrame with percentile values
        """
        if percentiles is None:
            percentiles = [10, 25, 50, 75, 90, 95, 99]
            
        percentile_df = pd.DataFrame({
            'Percentile': [f'{p}%' for p in percentiles],
            'Value': [f'{np.percentile(data, p):.2f}' for p in percentiles]
        })
        
        return percentile_df
    
    def streamlit_interface(self, data):
        """
        Create Streamlit interface for descriptive statistics.
        
        Args:
            data (array-like): Input data for analysis
        """
        st.header("📊 Descriptive Statistics")
        st.markdown("Understand your data through summary statistics and visualizations")

        # Calculate statistics
        stats_dict = self.calculate_descriptive_stats(data)

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
            cv_value = stats_dict['Coefficient of Variation (CV)']
            st.metric("CV (%)", f"{cv_value:.2f}%" if cv_value is not None else "N/A")
            st.caption("Relative variability")

        with col4:
            st.metric("IQR", f"{stats_dict['IQR']:.2f}")
            st.caption("Q3 - Q1")

        # Distribution Plots
        st.subheader("📈 Data Distribution Visualizations")
        fig = self.plot_distribution(data)
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
            fig_q = self.create_quartile_visualization(data)
            st.plotly_chart(fig_q, use_container_width=True)

        with tab2:
            st.markdown("**Deciles divide the data into 10 equal parts**")
            decile_df, fig_d = self.create_decile_visualization(data)
            st.dataframe(decile_df, use_container_width=True)
            st.plotly_chart(fig_d, use_container_width=True)

        with tab3:
            st.markdown("**Custom Percentile Calculator**")
            percentile_value = st.slider("Select Percentile:", 0, 100, 50)
            result = np.percentile(data, percentile_value)
            st.success(f"The {percentile_value}th percentile is: **{result:.2f}**")
            st.caption(f"This means {percentile_value}% of the data falls below {result:.2f}")

            # Show common percentiles
            st.markdown("**Common Percentiles:**")
            percentile_df = self.create_percentile_table(data)
            st.dataframe(percentile_df, use_container_width=True)