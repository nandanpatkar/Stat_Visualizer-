"""
K-Means Clustering Algorithm Implementation

K-Means is an unsupervised learning algorithm that partitions data into k clusters
by minimizing the within-cluster sum of squares (WCSS). It's one of the most
popular clustering algorithms.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import streamlit as st
import seaborn as sns


class KMeans:
    """
    K-Means Clustering implementation with educational explanations.
    
    K-Means algorithm works by:
    1. Initialize k centroids randomly
    2. Assign each point to nearest centroid
    3. Update centroids to mean of assigned points
    4. Repeat steps 2-3 until convergence
    
    Mathematical Formula: 
    Minimize: Σ(i=1 to k) Σ(x in Ci) ||x - μi||²
    
    Where:
    - k: number of clusters
    - Ci: i-th cluster
    - μi: centroid of i-th cluster
    - ||x - μi||²: squared Euclidean distance
    """
    
    def __init__(self, n_clusters=3, max_iter=300, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = None
        self.is_fitted = False
        
    @staticmethod
    def get_theory():
        """Return comprehensive theoretical explanation of K-Means Clustering."""
        return {
            'name': 'K-Means Clustering',
            'type': 'Unsupervised Learning - Clustering',
            
            # 1. What the algorithm is
            'definition': """
            🔹 **What is K-Means Clustering?**
            K-Means is like organizing a messy closet by grouping similar items together.
            It automatically finds groups (clusters) in your data by putting similar 
            data points close to each other and separating different groups.
            """,
            
            # 2. Why the algorithm is used
            'motivation': """
            🔹 **Why Use K-Means Clustering?**
            • 🎯 **Customer Segmentation**: Group customers by behavior, demographics
            • 📊 **Market Research**: Find distinct market segments
            • 🖼️ **Image Processing**: Compress images, segment objects
            • 📈 **Exploratory Analysis**: Discover hidden patterns in data
            • 🔍 **Preprocessing**: Prepare data for other algorithms
            • 💰 **Business Intelligence**: Identify distinct business opportunities
            """,
            
            # 3. Intuition with real-life analogy
            'intuition': """
            🔹 **Real-Life Analogy: The Party Planning Problem**
            
            Imagine you're planning seating arrangements for a party with 30 guests:
            
            🎉 **The Problem**: Create 3 tables where people at each table have similar interests
            🎯 **Goal**: Minimize awkward conversations, maximize fun discussions
            
            **K-Means is like a smart party planner who:**
            
            **Step 1**: 🎲 Randomly places 3 table centers in the room
            **Step 2**: 👥 Assigns each guest to the closest table
            **Step 3**: 📍 Moves each table to the center of its assigned guests  
            **Step 4**: 🔄 Repeats steps 2-3 until tables stop moving
            
            **Final Result**: 3 well-organized tables with similar people together!
            
            🎯 **In data terms**: Tables = Clusters, Guests = Data Points, Interests = Features
            """,
            
            # 4. Mathematical foundation
            'math_foundation': """
            🔹 **Mathematical Foundation (Step-by-Step)**
            
            **Objective Function (What we're minimizing):**
            ```
            WCSS = Σ(i=1 to k) Σ(x in Ci) ||x - μi||²
            ```
            
            **Where:**
            • `WCSS` = Within-Cluster Sum of Squares (total "tightness")
            • `k` = Number of clusters
            • `Ci` = i-th cluster
            • `μi` = Centroid (center) of i-th cluster
            • `||x - μi||²` = Squared distance from point x to centroid μi
            
            **Distance Calculation (Euclidean):**
            ```
            distance = √[(x₁-c₁)² + (x₂-c₂)² + ... + (xₙ-cₙ)²]
            ```
            
            **Centroid Update:**
            ```
            μi = (1/|Ci|) × Σ(x in Ci) x
            ```
            *Translation: New centroid = average of all points in cluster*
            
            **Convergence Condition:**
            ```
            |old_centroid - new_centroid| < tolerance
            ```
            """,
            
            # 5. Step-by-step working
            'algorithm_steps': """
            🔹 **How K-Means Works (Step-by-Step)**
            
            **Step 1: Choose K** 🎯
            • Decide how many clusters you want (k=3, k=5, etc.)
            • Use domain knowledge or elbow method to find optimal k
            
            **Step 2: Initialize Centroids** 🎲
            • Randomly place k points as initial cluster centers
            • Alternative: Use K-means++ for smarter initialization
            
            **Step 3: Assign Points to Clusters** 📍
            • For each data point: Calculate distance to all centroids
            • Assign point to the cluster with nearest centroid
            • Create k groups of data points
            
            **Step 4: Update Centroids** 🔄
            • For each cluster: Calculate average position of all points
            • Move centroid to this average position
            • New centroid = (sum of all points in cluster) / (number of points)
            
            **Step 5: Check Convergence** ✅
            • Compare old and new centroid positions
            • If centroids barely moved: STOP (converged)
            • If centroids moved significantly: Go back to Step 3
            
            **Step 6: Final Result** 🎉
            • Each point belongs to one cluster
            • Each cluster has a final centroid position
            • Total algorithm typically converges in 10-100 iterations
            """,
            
            # 6. Pseudocode
            'pseudocode': """
            🔹 **Pseudocode (Easy to Understand)**
            
            ```
            ALGORITHM: K-Means Clustering
            
            INPUT: 
                - data: matrix of points (n_points × n_features)
                - k: number of clusters
                - max_iterations: stopping condition
            
            OUTPUT:
                - cluster_labels: which cluster each point belongs to
                - centroids: final cluster centers
            
            BEGIN
                1. INITIALIZE k centroids randomly
                   centroids = random_points_from_data(k)
                
                2. FOR iteration = 1 to max_iterations:
                   
                   a. ASSIGN each point to nearest centroid:
                      FOR each point in data:
                          distances = calculate_distances(point, all_centroids)
                          cluster_label[point] = argmin(distances)
                   
                   b. UPDATE centroids:
                      FOR each cluster i:
                          points_in_cluster = get_points_with_label(i)
                          centroids[i] = mean(points_in_cluster)
                   
                   c. CHECK convergence:
                      IF centroids barely changed:
                          BREAK  // Algorithm converged
                
                3. RETURN cluster_labels, centroids
            END
            ```
            
            **K-Means++ Initialization (Smarter):**
            ```
            BEGIN
                1. CHOOSE first centroid randomly
                2. FOR i = 2 to k:
                   a. CALCULATE distance from each point to nearest centroid
                   b. CHOOSE next centroid with probability proportional to squared distance
                3. PROCEED with normal K-means
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
            
            class KMeansFromScratch:
                def __init__(self, k=3, max_iterations=100, tolerance=1e-4):
                    self.k = k
                    self.max_iterations = max_iterations
                    self.tolerance = tolerance
                
                def fit(self, X):
                    # Step 1: Initialize centroids randomly
                    n_samples, n_features = X.shape
                    self.centroids = X[np.random.randint(0, n_samples, self.k)]
                    
                    for iteration in range(self.max_iterations):
                        # Step 2: Assign points to clusters
                        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
                        self.labels = np.argmin(distances, axis=0)
                        
                        # Step 3: Update centroids
                        new_centroids = np.array([X[self.labels == i].mean(axis=0) 
                                                for i in range(self.k)])
                        
                        # Step 4: Check convergence
                        if np.allclose(self.centroids, new_centroids, atol=self.tolerance):
                            break
                        
                        self.centroids = new_centroids
                    
                    return self
                
                def predict(self, X):
                    distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
                    return np.argmin(distances, axis=0)
            
            # Example usage
            X = np.random.rand(100, 2)  # 100 points in 2D
            kmeans = KMeansFromScratch(k=3)
            kmeans.fit(X)
            labels = kmeans.predict(X)
            ```
            
            **Using Scikit-learn:**
            ```python
            from sklearn.cluster import KMeans
            import numpy as np
            
            # Generate sample data
            X = np.random.rand(100, 2)
            
            # Create and fit K-means
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            
            # Get results
            centroids = kmeans.cluster_centers_
            inertia = kmeans.inertia_  # WCSS
            
            print(f"Centroids: {centroids}")
            print(f"Inertia: {inertia:.2f}")
            ```
            """,
            
            # 8. Example with sample input/output
            'example': """
            🔹 **Complete Example: Customer Segmentation**
            
            **Input Data (Customer Features):**
            ```
            Customer | Annual Income ($k) | Spending Score (1-100)
            1        | 15                 | 39
            2        | 15                 | 81
            3        | 16                 | 6
            4        | 16                 | 77
            5        | 17                 | 40
            6        | 18                 | 76
            7        | 18                 | 6
            8        | 19                 | 94
            ```
            
            **K-Means Process (k=3):**
            
            **Initial State:**
            ```
            Random centroids: [(20, 50), (25, 30), (30, 70)]
            ```
            
            **Iteration 1:**
            ```
            Assign customers to nearest centroid:
            - Cluster 1: [Customer 1, 5] → Low income, medium spending
            - Cluster 2: [Customer 3, 7] → Low income, low spending  
            - Cluster 3: [Customer 2, 4, 6, 8] → Low income, high spending
            
            Update centroids:
            - New Centroid 1: (16, 39.5)
            - New Centroid 2: (17, 6)
            - New Centroid 3: (16.25, 82)
            ```
            
            **Final Result:**
            ```
            Cluster 1: "Budget Conscious" - Low income, moderate spending
            Cluster 2: "Conservative" - Low income, very low spending
            Cluster 3: "Trendy" - Low income, high spending
            ```
            
            **Business Insights:**
            • Target Cluster 1 with value products
            • Offer basic services to Cluster 2
            • Show premium trendy items to Cluster 3
            """,
            
            # 9. Visualization explanation
            'visualization': """
            🔹 **Understanding Through Visualizations**
            
            **Scatter Plot with Clusters:**
            📊 Different colors for each cluster
            • Points: Individual data samples
            • Colors: Show cluster membership
            • Stars/X marks: Cluster centroids
            
            **Elbow Method Plot:**
            📈 WCSS vs Number of Clusters
            • X-axis: Number of clusters (k)
            • Y-axis: Within-cluster sum of squares
            • "Elbow point": Optimal k value
            
            **Silhouette Analysis:**
            📊 Bar chart showing cluster quality
            • Each bar: One cluster's silhouette score
            • Height: How well-separated the cluster is
            • Good clustering: All bars above 0.5
            
            **Animation Visualization:**
            🎬 Step-by-step algorithm progress
            • Shows centroids moving during iterations
            • Points changing colors (cluster assignments)
            • Convergence when movement stops
            """,
            
            # 10. Time and space complexity
            'complexity': """
            🔹 **Time & Space Complexity**
            
            **Time Complexity:**
            • **Per Iteration**: O(n × k × d) 
              - n = number of data points
              - k = number of clusters  
              - d = number of dimensions/features
            • **Total Algorithm**: O(n × k × d × i)
              - i = number of iterations (usually 10-100)
            • **Typical Performance**: Very fast for most datasets
            
            **Space Complexity:**
            • **Centroids Storage**: O(k × d) - store k centroids
            • **Labels Storage**: O(n) - store cluster label for each point
            • **Distance Calculations**: O(n × k) - temporary storage
            • **Total**: O(n × k + k × d) ≈ O(n × k) for practical cases
            
            **Scalability:**
            • ✅ **Large n (samples)**: Handles millions of points efficiently
            • ✅ **Large k (clusters)**: Works well up to hundreds of clusters
            • ⚠️ **Large d (dimensions)**: Performance degrades with many features
            • ✅ **Parallelizable**: Can distribute across multiple cores
            """,
            
            # 11. Advantages and disadvantages
            'pros_cons': """
            🔹 **Advantages** ✅
            • **Simple**: Easy to understand and implement
            • **Fast**: Efficient for large datasets
            • **Guaranteed convergence**: Always finds a solution
            • **Scalable**: Handles big data well
            • **Memory efficient**: Low memory requirements
            • **Parallelizable**: Can run on multiple cores
            • **Deterministic**: Same result with same initialization
            
            🔹 **Disadvantages** ❌
            • **Choose k manually**: Need to specify number of clusters
            • **Sensitive to initialization**: Different starts → different results
            • **Assumes spherical clusters**: Struggles with elongated clusters
            • **Sensitive to outliers**: One extreme point affects entire cluster
            • **Equal cluster sizes**: Assumes clusters have similar sizes
            • **Feature scaling required**: Performance depends on feature scales
            • **Local optima**: May not find global best solution
            """,
            
            # 12. When to use and when NOT to use
            'usage_guide': """
            🔹 **When TO Use K-Means** ✅
            
            **Perfect for:**
            • 🎯 **Customer segmentation**: Group customers by behavior
            • 📊 **Market research**: Find distinct market segments  
            • 🔍 **Exploratory analysis**: Discover data patterns
            • 🖼️ **Image compression**: Reduce color palette
            • 📈 **Data preprocessing**: Prepare for other algorithms
            • ⚡ **Quick clustering**: Fast results needed
            
            **Good when:**
            • Clusters are roughly spherical (circular)
            • Clusters are similar in size
            • You have an idea of how many clusters to expect
            • Features are on similar scales
            • Dataset is not too high-dimensional (< 50 features)
            
            🔹 **When NOT to Use K-Means** ❌
            
            **Avoid when:**
            • 🌙 **Non-spherical clusters**: Crescent, elongated, or complex shapes
            • 📏 **Very different cluster sizes**: One big cluster, several tiny ones
            • 🎲 **Unknown k**: No idea how many clusters should exist
            • 🔢 **High dimensions**: Many features (curse of dimensionality)
            • 🎯 **Overlapping clusters**: Clusters blend into each other
            • 📊 **Categorical data**: Features are categories, not numbers
            
            **Use instead:**
            • DBSCAN (for arbitrary shapes, unknown k)
            • Hierarchical clustering (for unknown k, cluster relationships)
            • Gaussian Mixture Models (for overlapping clusters)
            • Spectral clustering (for complex shapes)
            """,
            
            # 13. Common interview questions
            'interview_questions': """
            🔹 **Common Interview Questions & Answers**
            
            **Q1: How do you choose the optimal number of clusters (k)?**
            A: Several methods:
            • **Elbow Method**: Plot WCSS vs k, look for "elbow" bend
            • **Silhouette Analysis**: Choose k with highest average silhouette score
            • **Domain Knowledge**: Business logic (e.g., 3 customer types)
            • **Gap Statistic**: Compare with random data clustering
            
            **Q2: What's the difference between K-means and K-means++?**
            A: **Initialization method:**
            • **K-means**: Random centroid initialization
            • **K-means++**: Smart initialization - spreads initial centroids far apart
            • **Result**: K-means++ typically converges faster and finds better solutions
            
            **Q3: How does K-means handle outliers?**
            A: **Poorly!** Outliers can:
            • Pull centroids toward extreme values
            • Create their own clusters
            • **Solution**: Remove outliers first, or use robust algorithms like DBSCAN
            
            **Q4: What's inertia/WCSS and why does it always decrease?**
            A: **Inertia = Within-Cluster Sum of Squares**
            • Measures total "tightness" of clusters
            • Always decreases because algorithm minimizes this value
            • Lower inertia = tighter clusters
            
            **Q5: Can K-means be used for non-numerical data?**
            A: **Not directly.** K-means needs:
            • Numerical features for distance calculation
            • **Solutions**: One-hot encoding, embedding, or use K-modes algorithm
            """,
            
            # 14. Common mistakes
            'common_mistakes': """
            🔹 **Common Beginner Mistakes & How to Avoid**
            
            **Mistake 1: Not scaling features** 🚫
            ❌ Using features with different units (age vs income)
            ✅ **Fix**: Standardize all features to same scale (StandardScaler)
            
            **Mistake 2: Wrong choice of k** 🚫
            ❌ Randomly guessing number of clusters
            ✅ **Fix**: Use elbow method, silhouette analysis, domain knowledge
            
            **Mistake 3: Ignoring cluster shapes** 🚫
            ❌ Assuming all clusters are circular
            ✅ **Fix**: Visualize data first, consider other algorithms for complex shapes
            
            **Mistake 4: Not handling outliers** 🚫
            ❌ Including extreme values that skew results
            ✅ **Fix**: Remove outliers using IQR method or Z-score filtering
            
            **Mistake 5: Using only one initialization** 🚫
            ❌ Running K-means once and trusting the result
            ✅ **Fix**: Run multiple times with different initializations, pick best result
            
            **Mistake 6: Wrong interpretation** 🚫
            ❌ "These clusters represent real customer types"
            ✅ **Fix**: "These are mathematical groupings that might represent customer types"
            """,
            
            # 15. Comparison with similar algorithms
            'comparisons': """
            🔹 **K-Means vs Similar Clustering Algorithms**
            
            **K-Means vs Hierarchical Clustering:**
            • **K-Means**: Fast, needs k, spherical clusters
            • **Hierarchical**: Slower, finds k automatically, any cluster shape
            • **Use hierarchical**: When you don't know k or want cluster relationships
            
            **K-Means vs DBSCAN:**
            • **K-Means**: Needs k, sensitive to outliers, spherical clusters
            • **DBSCAN**: Finds k automatically, handles outliers, arbitrary shapes
            • **Use DBSCAN**: For noise-robust clustering with unknown k
            
            **K-Means vs Gaussian Mixture Models:**
            • **K-Means**: Hard assignment (point belongs to one cluster)
            • **GMM**: Soft assignment (probability of belonging to each cluster)
            • **Use GMM**: When clusters overlap or you need probabilities
            
            **K-Means vs K-Medoids:**
            • **K-Means**: Uses mean as centroid (sensitive to outliers)
            • **K-Medoids**: Uses actual data point as centroid (robust to outliers)
            • **Use K-Medoids**: When you have outliers or need interpretable centroids
            """,
            
            # 16. Real-world applications
            'real_world_applications': """
            🔹 **Real-World Applications & Industry Use Cases**
            
            **🛒 E-commerce & Retail:**
            • **Customer Segmentation**: Group customers by purchase behavior
            • **Product Recommendation**: Cluster similar products
            • **Inventory Management**: Group stores with similar demand patterns
            • **Price Optimization**: Segment markets for dynamic pricing
            
            **📊 Marketing & Advertising:**
            • **Market Segmentation**: Identify target demographics
            • **Campaign Optimization**: Group audiences for personalized ads
            • **A/B Testing**: Cluster user responses
            • **Brand Positioning**: Map competitive landscape
            
            **🏥 Healthcare:**
            • **Disease Subtyping**: Find disease variants in patient data
            • **Drug Discovery**: Group molecules with similar properties
            • **Treatment Optimization**: Cluster patients for personalized medicine
            • **Healthcare Resource Planning**: Group hospitals by patient load
            
            **🖼️ Image & Computer Vision:**
            • **Image Segmentation**: Separate objects in medical images
            • **Color Quantization**: Reduce colors for compression
            • **Object Recognition**: Group similar visual features
            • **Video Analytics**: Cluster scenes or activities
            
            **📈 Finance:**
            • **Risk Segmentation**: Group loans/investments by risk level
            • **Fraud Detection**: Cluster transaction patterns
            • **Algorithmic Trading**: Group stocks with similar behavior
            • **Credit Scoring**: Segment customers for loan approval
            
            **🏭 Manufacturing:**
            • **Quality Control**: Group defects by type and cause
            • **Supply Chain**: Cluster suppliers by performance
            • **Predictive Maintenance**: Group machines by failure patterns
            • **Process Optimization**: Cluster production runs
            
            **📱 Technology:**
            • **User Behavior Analysis**: Group users by app usage
            • **Content Recommendation**: Cluster similar content
            • **Network Analysis**: Group servers or devices
            • **Feature Engineering**: Create new features from clusters
            
            **💡 Key Success Factors:**
            • Choose appropriate k using multiple methods
            • Scale features properly
            • Handle outliers before clustering
            • Validate results with domain experts
            • Iterate and refine based on business feedback
            """
        }
    
    def generate_sample_data(self, n_samples=300, n_features=2, n_centers=3, cluster_std=1.0):
        """Generate sample clustering data."""
        X, y_true = make_blobs(
            n_samples=n_samples,
            centers=n_centers,
            n_features=n_features,
            cluster_std=cluster_std,
            random_state=self.random_state
        )
        return X, y_true
    
    def fit(self, X):
        """Fit the K-Means model."""
        self.model = SklearnKMeans(
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            random_state=self.random_state,
            n_init=10
        )
        self.model.fit(X)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Predict cluster labels for new data."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)
    
    def fit_predict(self, X):
        """Fit the model and predict cluster labels."""
        self.fit(X)
        return self.predict(X)
    
    def get_centroids(self):
        """Get cluster centroids."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting centroids")
        return self.model.cluster_centers_
    
    def get_metrics(self, X, y_pred, y_true=None):
        """Calculate clustering evaluation metrics."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calculating metrics")
            
        metrics = {
            'Inertia (WCSS)': self.model.inertia_,
            'Silhouette Score': silhouette_score(X, y_pred),
            'Number of Iterations': self.model.n_iter_
        }
        
        # Add external validation metrics if true labels are available
        if y_true is not None:
            metrics.update({
                'Adjusted Rand Index': adjusted_rand_score(y_true, y_pred),
                'Normalized Mutual Information': normalized_mutual_info_score(y_true, y_pred)
            })
        
        return metrics
    
    def elbow_method(self, X, max_k=10):
        """Perform elbow method to find optimal number of clusters."""
        k_range = range(1, max_k + 1)
        inertias = []
        silhouette_scores = []
        
        for k in k_range:
            kmeans = SklearnKMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
            
            if k > 1:  # Silhouette score requires at least 2 clusters
                silhouette_scores.append(silhouette_score(X, kmeans.labels_))
            else:
                silhouette_scores.append(0)
        
        return k_range, inertias, silhouette_scores
    
    def plot_elbow_method(self, X, max_k=10):
        """Plot elbow method results."""
        k_range, inertias, silhouette_scores = self.elbow_method(X, max_k)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Elbow plot (Inertia)
        axes[0].plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
        axes[0].set_xlabel('Number of Clusters (k)')
        axes[0].set_ylabel('Inertia (WCSS)')
        axes[0].set_title('Elbow Method: Inertia vs Number of Clusters')
        axes[0].grid(True, alpha=0.3)
        
        # Highlight the elbow point (simple heuristic)
        if len(inertias) > 2:
            # Calculate second derivative to find elbow
            second_derivative = np.diff(inertias, 2)
            elbow_idx = np.argmax(second_derivative) + 2  # +2 because of diff operations
            if elbow_idx < len(k_range):
                axes[0].axvline(x=k_range[elbow_idx], color='red', linestyle='--', 
                               label=f'Suggested k={k_range[elbow_idx]}')
                axes[0].legend()
        
        # Silhouette score plot
        axes[1].plot(k_range[1:], silhouette_scores[1:], 'ro-', linewidth=2, markersize=8)
        axes[1].set_xlabel('Number of Clusters (k)')
        axes[1].set_ylabel('Silhouette Score')
        axes[1].set_title('Silhouette Score vs Number of Clusters')
        axes[1].grid(True, alpha=0.3)
        
        # Highlight best silhouette score
        if len(silhouette_scores[1:]) > 0:
            best_silhouette_idx = np.argmax(silhouette_scores[1:]) + 1
            best_k = k_range[best_silhouette_idx]
            axes[1].axvline(x=best_k, color='red', linestyle='--', 
                           label=f'Best k={best_k}')
            axes[1].legend()
        
        plt.tight_layout()
        return fig
    
    def plot_clusters(self, X, y_pred, title="K-Means Clustering Results"):
        """Plot clustering results."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting")
            
        centroids = self.get_centroids()
        
        if X.shape[1] == 2:
            # 2D visualization
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot data points colored by cluster
            scatter = ax.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', 
                               alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
            
            # Plot centroids
            ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', 
                      s=200, linewidths=3, label='Centroids')
            
            # Add cluster boundaries (Voronoi diagram approximation)
            if len(centroids) > 1:
                # Create a mesh for plotting decision boundary
                h = 0.1
                x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
                y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                   np.arange(y_min, y_max, h))
                
                mesh_points = np.c_[xx.ravel(), yy.ravel()]
                mesh_labels = self.predict(mesh_points)
                mesh_labels = mesh_labels.reshape(xx.shape)
                
                ax.contour(xx, yy, mesh_labels, colors='black', alpha=0.3, linewidths=0.5)
            
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add colorbar
            plt.colorbar(scatter, label='Cluster')
            
        else:
            # Multi-dimensional data: plot pairwise features
            n_features = min(X.shape[1], 4)  # Limit to 4 features for visualization
            fig, axes = plt.subplots(n_features, n_features, figsize=(15, 15))
            
            if n_features == 1:
                axes = [axes]
            
            for i in range(n_features):
                for j in range(n_features):
                    if i == j:
                        # Diagonal: histogram of feature
                        if n_features == 1:
                            ax = axes[0] if isinstance(axes, list) else axes
                        else:
                            ax = axes[i, j]
                        for cluster in range(self.n_clusters):
                            mask = y_pred == cluster
                            ax.hist(X[mask, i], alpha=0.5, label=f'Cluster {cluster}', bins=20)
                        ax.set_xlabel(f'Feature {i+1}')
                        ax.set_ylabel('Frequency')
                        if i == 0 and j == 0:
                            ax.legend()
                    else:
                        # Off-diagonal: scatter plot
                        if n_features == 1:
                            continue
                        ax = axes[i, j]
                        scatter = ax.scatter(X[:, j], X[:, i], c=y_pred, cmap='viridis', 
                                           alpha=0.6, s=30)
                        ax.scatter(centroids[:, j], centroids[:, i], c='red', marker='x', 
                                  s=100, linewidths=2)
                        ax.set_xlabel(f'Feature {j+1}')
                        ax.set_ylabel(f'Feature {i+1}')
                    
                    if n_features > 1:
                        axes[i, j].grid(True, alpha=0.3)
            
            if n_features > 1:
                plt.suptitle(title, fontsize=16)
        
        plt.tight_layout()
        return fig
    
    def plot_cluster_analysis(self, X, y_pred):
        """Create comprehensive cluster analysis plots."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting")
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Cluster size distribution
        unique, counts = np.unique(y_pred, return_counts=True)
        axes[0, 0].bar(unique, counts, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_xlabel('Cluster')
        axes[0, 0].set_ylabel('Number of Points')
        axes[0, 0].set_title('Cluster Size Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add count labels on bars
        for i, count in enumerate(counts):
            axes[0, 0].text(unique[i], count + max(counts) * 0.01, str(count), 
                           ha='center', va='bottom')
        
        # Within-cluster distances
        distances_to_centroid = []
        centroids = self.get_centroids()
        
        for i in range(self.n_clusters):
            cluster_mask = y_pred == i
            cluster_points = X[cluster_mask]
            if len(cluster_points) > 0:
                centroid = centroids[i]
                distances = np.linalg.norm(cluster_points - centroid, axis=1)
                distances_to_centroid.extend(distances)
        
        axes[0, 1].hist(distances_to_centroid, bins=30, alpha=0.7, color='lightgreen', 
                       edgecolor='black')
        axes[0, 1].set_xlabel('Distance to Centroid')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Distances to Centroids')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Silhouette analysis
        silhouette_scores = []
        sample_silhouette_values = silhouette_score(X, y_pred, metric='euclidean')
        
        from sklearn.metrics import silhouette_samples
        sample_silhouette_values = silhouette_samples(X, y_pred)
        
        y_lower = 10
        for i in range(self.n_clusters):
            cluster_silhouette_values = sample_silhouette_values[y_pred == i]
            cluster_silhouette_values.sort()
            
            size_cluster_i = cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            
            color = plt.cm.nipy_spectral(float(i) / self.n_clusters)
            axes[1, 0].fill_betweenx(np.arange(y_lower, y_upper),
                                    0, cluster_silhouette_values,
                                    facecolor=color, edgecolor=color, alpha=0.7)
            
            # Label the silhouette plots with their cluster numbers at the middle
            axes[1, 0].text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10
        
        axes[1, 0].set_xlabel('Silhouette Coefficient Values')
        axes[1, 0].set_ylabel('Cluster Label')
        axes[1, 0].set_title('Silhouette Plot for Individual Clusters')
        
        # Add average silhouette score line
        avg_silhouette = silhouette_score(X, y_pred)
        axes[1, 0].axvline(x=avg_silhouette, color="red", linestyle="--", 
                          label=f'Average Score: {avg_silhouette:.3f}')
        axes[1, 0].legend()
        
        # Cluster centers comparison
        if X.shape[1] <= 10:  # Only show if reasonable number of features
            feature_names = [f'Feature {i+1}' for i in range(X.shape[1])]
            centroids = self.get_centroids()
            
            x_pos = np.arange(len(feature_names))
            width = 0.8 / self.n_clusters
            
            for i in range(self.n_clusters):
                offset = (i - self.n_clusters/2 + 0.5) * width
                axes[1, 1].bar(x_pos + offset, centroids[i], width, 
                              label=f'Cluster {i}', alpha=0.7)
            
            axes[1, 1].set_xlabel('Features')
            axes[1, 1].set_ylabel('Centroid Value')
            axes[1, 1].set_title('Cluster Centroids Comparison')
            axes[1, 1].set_xticks(x_pos)
            axes[1, 1].set_xticklabels(feature_names, rotation=45, ha='right')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Too many features\nto display centroids', 
                           ha='center', va='center', transform=axes[1, 1].transAxes,
                           fontsize=12, bbox=dict(boxstyle="round", facecolor="lightgray"))
            axes[1, 1].set_title('Cluster Centroids')
        
        plt.tight_layout()
        return fig
    
    def streamlit_interface(self):
        """Create comprehensive Streamlit interface for K-Means Clustering."""
        st.subheader("🎯 K-Means Clustering")
        
        theory = self.get_theory()
        
        # Main tabs for comprehensive coverage
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🎯 Overview", "📚 Deep Dive", "💻 Implementation", 
            "🧪 Interactive Demo", "❓ Q&A", "🏢 Applications"
        ])
        
        with tab1:
            # Overview Tab - Essential Information
            st.markdown("### 🎯 What is K-Means Clustering?")
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
            # Interactive Demo Tab - This will contain the actual working demo
            st.markdown("### 🧪 Try K-Means Yourself!")
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
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            n_samples = st.slider("Number of samples:", 100, 1000, 300)
            
        with col2:
            n_features = st.slider("Number of features:", 2, 6, 2)
            
        with col3:
            n_true_clusters = st.slider("True number of clusters:", 2, 8, 3)
            
        with col4:
            cluster_std = st.slider("Cluster standard deviation:", 0.5, 3.0, 1.0, 0.1)
        
        # Generate sample data
        X, y_true = self.generate_sample_data(n_samples, n_features, 
                                             n_true_clusters, cluster_std)
        
        # Elbow method to find optimal k
        st.markdown("### 📊 Finding Optimal Number of Clusters")
        
        max_k = min(10, n_samples // 10)  # Reasonable upper limit for k
        fig_elbow = self.plot_elbow_method(X, max_k)
        st.pyplot(fig_elbow)
        plt.close()
        
        # K-Means parameters
        st.markdown("### ⚙️ K-Means Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            n_clusters = st.slider("Number of clusters (k):", 2, max_k, 
                                 min(n_true_clusters, max_k))
            
        with col2:
            standardize = st.checkbox("Standardize features", value=True)
        
        # Preprocess data
        if standardize:
            scaler = StandardScaler()
            X_processed = scaler.fit_transform(X)
        else:
            X_processed = X
        
        # Update model parameters and fit
        self.n_clusters = n_clusters
        y_pred = self.fit_predict(X_processed)
        
        # Results section
        st.markdown("### 📊 Results")
        
        # Metrics
        metrics = self.get_metrics(X_processed, y_pred, y_true)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Inertia (WCSS)", f"{metrics['Inertia (WCSS)']:.2f}")
            
        with col2:
            st.metric("Silhouette Score", f"{metrics['Silhouette Score']:.3f}")
            
        with col3:
            st.metric("Iterations to Converge", metrics['Number of Iterations'])
        
        # External validation metrics (if true labels available)
        if 'Adjusted Rand Index' in metrics:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Adjusted Rand Index", f"{metrics['Adjusted Rand Index']:.3f}")
            with col2:
                st.metric("Normalized Mutual Info", f"{metrics['Normalized Mutual Information']:.3f}")
        
        # Visualizations
        st.markdown("### 📈 Cluster Visualization")
        
        # Main clustering plot
        fig_clusters = self.plot_clusters(X_processed, y_pred, "K-Means Clustering Results")
        st.pyplot(fig_clusters)
        plt.close()
        
        # Detailed analysis
        st.markdown("### 📊 Detailed Cluster Analysis")
        fig_analysis = self.plot_cluster_analysis(X_processed, y_pred)
        st.pyplot(fig_analysis)
        plt.close()
        
        # Cluster summary
        st.markdown("### 📋 Cluster Summary")
        
        cluster_summary = []
        for i in range(n_clusters):
            cluster_mask = y_pred == i
            cluster_size = np.sum(cluster_mask)
            
            if cluster_size > 0:
                cluster_data = X_processed[cluster_mask]
                cluster_mean = np.mean(cluster_data, axis=0)
                cluster_std = np.std(cluster_data, axis=0)
                
                summary = {
                    'Cluster': i,
                    'Size': cluster_size,
                    'Percentage': f"{cluster_size/len(X_processed)*100:.1f}%"
                }
                
                # Add feature statistics
                for j in range(n_features):
                    summary[f'Feature {j+1} Mean'] = f"{cluster_mean[j]:.3f}"
                    summary[f'Feature {j+1} Std'] = f"{cluster_std[j]:.3f}"
                
                cluster_summary.append(summary)
        
        summary_df = pd.DataFrame(cluster_summary)
        st.dataframe(summary_df)
        
        # Interpretation
        st.markdown("### 🔍 Interpretation")
        
        silhouette = metrics['Silhouette Score']
        
        if silhouette > 0.7:
            st.success(f"**Excellent clustering!** Silhouette Score: {silhouette:.3f}")
            st.write("Clusters are well-separated and internally cohesive.")
        elif silhouette > 0.5:
            st.info(f"**Good clustering.** Silhouette Score: {silhouette:.3f}")
            st.write("Clusters are reasonably well-defined.")
        elif silhouette > 0.3:
            st.warning(f"**Moderate clustering.** Silhouette Score: {silhouette:.3f}")
            st.write("Some clusters may be overlapping or poorly defined.")
        else:
            st.error(f"**Poor clustering.** Silhouette Score: {silhouette:.3f}")
            st.write("Clusters are not well-separated. Consider different k or algorithm.")
        
        # Recommendations
        st.markdown("**Recommendations:**")
        if silhouette < 0.5:
            st.write("• Try different values of k using the elbow method")
            st.write("• Consider standardizing features if not done already")
            st.write("• Check if data has natural cluster structure")
            st.write("• Consider other clustering algorithms (hierarchical, DBSCAN)")


def main():
    """Main function for testing K-Means Clustering."""
    kmeans = KMeans()
    kmeans.streamlit_interface()


if __name__ == "__main__":
    main()