"""
Shopper Spectrum - E-Commerce Customer Analytics Engine
======================================================

A comprehensive machine learning solution for e-commerce customer segmentation and product recommendations.

This script performs:
1. Data preprocessing and exploratory data analysis
2. RFM (Recency, Frequency, Monetary) analysis for customer segmentation
3. K-means clustering for customer classification
4. Item-based collaborative filtering for product recommendations
5. Model persistence for deployment

Key Features:
- Advanced customer segmentation using RFM metrics
- Product recommendation engine using collaborative filtering
- Comprehensive data visualization and insights
- Model serialization for production deployment

Author: Arunov Chakraborty
Created: 2nd August, 2025
Dataset: Online Retail Dataset (UCI ML Repository)
Dependencies: pandas, numpy, matplotlib, seaborn, scikit-learn, scipy

Usage:
    python ShopSpec.py

Output:
    - Trained models saved in ./models/ directory
    - Comprehensive visualizations and analysis
    - Ready-to-deploy models for Streamlit dashboard
"""

# ================================================================================
# IMPORTING LIBRARIES
# ================================================================================

# Core data processing libraries
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')  # Suppressing warnings for cleaner output

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning libraries
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

# Model persistence libraries
import pickle
import joblib

# Setting visualization style for professional plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ================================================================================
# MAIN ANALYZER CLASS
# ================================================================================

class EcommerceAnalyzer:
    """
    A comprehensive e-commerce analytics engine for customer segmentation and product recommendations.
    
    This class encapsulates the entire machine learning pipeline from data preprocessing
    to model deployment, providing methods for:
    - Data loading and preprocessing
    - Exploratory data analysis
    - RFM analysis and customer segmentation
    - Product recommendation system development
    - Model persistence and deployment preparation
    
    Attributes:
        data_path (str): Path to the input dataset
        df (pd.DataFrame): Main dataset after preprocessing
        rfm_df (pd.DataFrame): RFM metrics dataset
        scaler (StandardScaler): Scaler for RFM normalization
        kmeans_model (KMeans): Trained clustering model
        product_similarity_matrix (pd.DataFrame): Product similarity matrix
        product_mapping (dict): Product name to code mapping
    """
    
    def __init__(self, data_path):
        """
        Initializing the analyzer with the dataset path.
        
        Args:
            data_path (str): Path to the e-commerce dataset CSV file
        """
        self.data_path = data_path
        
        # Initializing all attributes to None - will be populated during analysis
        self.df = None                          # Main preprocessed dataset
        self.rfm_df = None                      # RFM analysis results
        self.scaler = None                      # StandardScaler for normalization
        self.kmeans_model = None                # Trained K-means model
        self.product_similarity_matrix = None   # Product similarity matrix
        self.product_mapping = None             # Product name to code mapping
        
    # ============================================================================
    # DATA LOADING AND PREPROCESSING
    # ============================================================================
    
    def load_and_preprocess_data(self):
        """
        Loading and preprocessing the e-commerce dataset.
        
        This method performs comprehensive data cleaning including:
        - Loading the dataset with appropriate encoding
        - Handling missing values
        - Filtering out cancelled transactions
        - Removing invalid quantity and price values
        - Creating derived features
        
        Returns:
            pd.DataFrame: Preprocessed dataset ready for analysis
        """
        
        print("📊 Loading dataset...")
        
        # Load dataset with ISO-8859-1 encoding (common for older datasets)
        # low_memory=False prevents DtypeWarning for mixed types
        self.df = pd.read_csv(self.data_path, encoding='ISO-8859-1', low_memory=False)
        
        print(f"Original dataset shape: {self.df.shape}")
        print("\n📋 Dataset Info:")
        print(self.df.info())
        
        # ===== DATA CLEANING PIPELINE =====
        
        print("\n🧹 Starting data preprocessing...")
        
        # Step 1: Remove rows with missing CustomerID
        # CustomerID is essential for customer analysis
        initial_rows = len(self.df)
        self.df = self.df.dropna(subset=['CustomerID'])
        print(f"Removed {initial_rows - len(self.df)} rows with missing CustomerID")
        
        # Step 2: Clean CustomerID format
        # Convert to string and remove decimal points (e.g., '12345.0' -> '12345')
        self.df['CustomerID'] = self.df['CustomerID'].astype(str).str.replace('.0', '')
        
        # Step 3: Remove cancelled transactions
        # Cancelled invoices start with 'C' and represent returns/cancellations
        cancelled_invoices = self.df['InvoiceNo'].astype(str).str.startswith('C').sum()
        self.df = self.df[~self.df['InvoiceNo'].astype(str).str.startswith('C')]
        print(f"Removed {cancelled_invoices} cancelled invoices")
        
        # Step 4: Remove invalid quantity and price values
        # Business logic: Quantity and price must be positive
        invalid_qty = len(self.df[(self.df['Quantity'] <= 0)])
        invalid_price = len(self.df[(self.df['UnitPrice'] <= 0)])
        self.df = self.df[(self.df['Quantity'] > 0) & (self.df['UnitPrice'] > 0)]
        print(f"Removed {invalid_qty} rows with invalid quantity")
        print(f"Removed {invalid_price} rows with invalid price")
        
        # Step 5: Convert data types
        # Convert InvoiceDate to datetime for time-based analysis
        self.df['InvoiceDate'] = pd.to_datetime(self.df['InvoiceDate'])
        
        # Step 6: Create derived features
        # Calculate total amount per transaction line
        self.df['TotalAmount'] = self.df['Quantity'] * self.df['UnitPrice']
        
        print(f"\n✅ Preprocessed dataset shape: {self.df.shape}")
        return self.df
    
    # ============================================================================
    # EXPLORATORY DATA ANALYSIS
    # ============================================================================
    
    def perform_eda(self):
        """
        Perform comprehensive Exploratory Data Analysis (EDA).
        
        This method generates multiple visualizations to understand:
        - Geographic distribution of transactions
        - Product sales patterns
        - Temporal trends
        - Customer behavior patterns
        - Revenue distribution
        
        The analysis helps inform business decisions and model development.
        """
        
        print("\n📈 Starting Exploratory Data Analysis...")
        
        # Create a comprehensive figure with multiple subplots
        fig = plt.figure(figsize=(20, 15))
        
        # === VISUALIZATION 1: GEOGRAPHIC ANALYSIS ===
        ax1 = plt.subplot(3, 2, 1)
        country_transactions = self.df['Country'].value_counts().head(10)
        country_transactions.plot(kind='bar', ax=ax1, color='skyblue', edgecolor='black')
        ax1.set_title('Top 10 Countries by Transaction Volume', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Country')
        ax1.set_ylabel('Number of Transactions')
        plt.xticks(rotation=45)
        
        # === VISUALIZATION 2: PRODUCT ANALYSIS ===
        ax2 = plt.subplot(3, 2, 2)
        top_products = self.df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
        top_products.plot(kind='barh', ax=ax2, color='lightcoral')
        ax2.set_title('Top 10 Best-Selling Products', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Total Quantity Sold')
        
        # === VISUALIZATION 3: TEMPORAL ANALYSIS ===
        ax3 = plt.subplot(3, 2, 3)
        # Resample by month and calculate total revenue
        monthly_revenue = self.df.set_index('InvoiceDate').resample('M')['TotalAmount'].sum()
        monthly_revenue.plot(ax=ax3, marker='o', linewidth=2, markersize=6)
        ax3.set_title('Monthly Revenue Trend', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Month')
        ax3.set_ylabel('Total Revenue ($)')
        ax3.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # === VISUALIZATION 4: TRANSACTION VALUE DISTRIBUTION ===
        ax4 = plt.subplot(3, 2, 4)
        # Focus on transactions under $100 for better visualization
        transaction_amounts = self.df['TotalAmount'][self.df['TotalAmount'] <= 100]
        transaction_amounts.hist(bins=50, ax=ax4, edgecolor='black', alpha=0.7, color='gold')
        ax4.set_title('Distribution of Transaction Amounts (≤$100)', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Transaction Amount ($)')
        ax4.set_ylabel('Frequency')
        
        # === VISUALIZATION 5: CUSTOMER SPENDING DISTRIBUTION ===
        ax5 = plt.subplot(3, 2, 5)
        # Calculate total spending per customer
        customer_spending = self.df.groupby('CustomerID')['TotalAmount'].sum()
        # Focus on customers spending under $5000 for better visualization
        customer_spending_filtered = customer_spending[customer_spending <= 5000]
        customer_spending_filtered.hist(bins=50, ax=ax5, edgecolor='black', alpha=0.7, color='mediumseagreen')
        ax5.set_title('Distribution of Customer Total Spending (≤$5000)', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Total Spending per Customer ($)')
        ax5.set_ylabel('Number of Customers')
        
        # === VISUALIZATION 6: HOURLY TRANSACTION PATTERN ===
        ax6 = plt.subplot(3, 2, 6)
        # Extract hour from InvoiceDate
        self.df['Hour'] = self.df['InvoiceDate'].dt.hour
        hourly_transactions = self.df.groupby('Hour').size()
        hourly_transactions.plot(kind='line', marker='o', ax=ax6, linewidth=2, markersize=6, color='purple')
        ax6.set_title('Transaction Pattern by Hour of Day', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Hour of Day')
        ax6.set_ylabel('Number of Transactions')
        ax6.set_xticks(range(0, 24))
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # === SUMMARY STATISTICS ===
        print("\n📊 Summary Statistics:")
        print(f"Total Transactions: {len(self.df):,}")
        print(f"Unique Customers: {self.df['CustomerID'].nunique():,}")
        print(f"Unique Products: {self.df['StockCode'].nunique():,}")
        print(f"Date Range: {self.df['InvoiceDate'].min()} to {self.df['InvoiceDate'].max()}")
        print(f"Total Revenue: ${self.df['TotalAmount'].sum():,.2f}")
        print(f"Average Transaction Value: ${self.df['TotalAmount'].mean():.2f}")
        print(f"Average Customer Lifetime Value: ${self.df.groupby('CustomerID')['TotalAmount'].sum().mean():.2f}")
    
    # ============================================================================
    # RFM ANALYSIS
    # ============================================================================
    
    def calculate_rfm(self):
        """
        Calculate RFM (Recency, Frequency, Monetary) metrics for customer segmentation.
        
        RFM Analysis segments customers based on:
        - Recency: How recently did the customer purchase?
        - Frequency: How often does the customer purchase?
        - Monetary: How much does the customer spend?
        
        Returns:
            pd.DataFrame: RFM metrics for each customer with outliers removed
        """
        
        print("\n🔢 Calculating RFM metrics...")
        
        # Define snapshot date (day after the last transaction for recency calculation)
        snapshot_date = self.df['InvoiceDate'].max() + timedelta(days=1)
        print(f"Snapshot date for recency calculation: {snapshot_date}")
        
        # === RFM CALCULATION ===
        rfm_df = self.df.groupby('CustomerID').agg({
            # Recency: Days between snapshot date and last purchase
            'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
            # Frequency: Number of unique transactions (invoices)
            'InvoiceNo': 'nunique',
            # Monetary: Total amount spent
            'TotalAmount': 'sum'
        })
        
        # Rename columns for clarity
        rfm_df.columns = ['Recency', 'Frequency', 'Monetary']
        
        # === OUTLIER REMOVAL ===
        # Use IQR (Interquartile Range) method to remove outliers
        print("\n🔍 Removing outliers using IQR method...")
        
        for col in ['Recency', 'Frequency', 'Monetary']:
            Q1 = rfm_df[col].quantile(0.25)      # First quartile
            Q3 = rfm_df[col].quantile(0.75)      # Third quartile
            IQR = Q3 - Q1                        # Interquartile range
            
            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Remove outliers
            initial_count = len(rfm_df)
            rfm_df = rfm_df[(rfm_df[col] >= lower_bound) & (rfm_df[col] <= upper_bound)]
            removed = initial_count - len(rfm_df)
            
            if removed > 0:
                print(f"Removed {removed} outliers based on {col}")
        
        self.rfm_df = rfm_df
        
        # === RFM STATISTICS ===
        print("\n📊 RFM Statistics:")
        print(self.rfm_df.describe())
        
        # === RFM VISUALIZATIONS ===
        self._visualize_rfm_distributions()
        
        return self.rfm_df
    
    def _visualize_rfm_distributions(self):
        """Create visualizations for RFM metric distributions."""
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Recency distribution
        axes[0].hist(self.rfm_df['Recency'], bins=50, edgecolor='black', alpha=0.7, color='lightblue')
        axes[0].set_title('Recency Distribution\n(Days since last purchase)', fontweight='bold')
        axes[0].set_xlabel('Days')
        axes[0].set_ylabel('Number of customers')
        axes[0].grid(True, alpha=0.3)
        
        # Frequency distribution
        axes[1].hist(self.rfm_df['Frequency'], bins=50, edgecolor='black', alpha=0.7, color='lightgreen')
        axes[1].set_title('Frequency Distribution\n(Number of transactions)', fontweight='bold')
        axes[1].set_xlabel('Number of transactions')
        axes[1].set_ylabel('Number of customers')
        axes[1].grid(True, alpha=0.3)
        
        # Monetary distribution
        axes[2].hist(self.rfm_df['Monetary'], bins=50, edgecolor='black', alpha=0.7, color='lightcoral')
        axes[2].set_title('Monetary Distribution\n(Total amount spent)', fontweight='bold')
        axes[2].set_xlabel('Total amount spent ($)')
        axes[2].set_ylabel('Number of customers')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    # ============================================================================
    # CUSTOMER SEGMENTATION USING K-MEANS CLUSTERING
    # ============================================================================
    
    def perform_clustering(self, max_clusters=10):
        """
        Perform customer segmentation using K-means clustering on RFM metrics.
        
        This method:
        1. Standardizes RFM metrics using StandardScaler
        2. Finds optimal number of clusters using elbow method and silhouette analysis
        3. Performs final clustering and labels segments
        4. Visualizes results using PCA
        
        Args:
            max_clusters (int): Maximum number of clusters to evaluate
            
        Returns:
            pd.DataFrame: RFM data with cluster assignments and segment labels
        """
        
        print("\n🎯 Starting clustering analysis...")
        
        # === FEATURE SCALING ===
        # Standardize RFM values to ensure equal contribution to clustering
        self.scaler = StandardScaler()
        rfm_scaled = self.scaler.fit_transform(self.rfm_df)
        
        # === OPTIMAL CLUSTER SELECTION ===
        print("Finding optimal number of clusters...")
        
        inertias = []           # Within-cluster sum of squares
        silhouette_scores = []  # Silhouette analysis scores
        K_range = range(2, max_clusters + 1)
        
        for k in K_range:
            # Train K-means model
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(rfm_scaled)
            
            # Store evaluation metrics
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(rfm_scaled, kmeans.labels_))
        
        # === VISUALIZATION OF CLUSTER EVALUATION ===
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Elbow curve for inertia
        ax1.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of clusters (k)')
        ax1.set_ylabel('Inertia (Within-cluster sum of squares)')
        ax1.set_title('Elbow Method For Optimal k', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Silhouette scores
        ax2.plot(K_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Score For Different k', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # === FINAL CLUSTERING ===
        # Choose optimal k (4 clusters typically work well for RFM analysis)
        optimal_k = 4
        print(f"\n✨ Using {optimal_k} clusters for segmentation")
        
        # Train final K-means model
        self.kmeans_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        self.rfm_df['Cluster'] = self.kmeans_model.fit_predict(rfm_scaled)
        
        # === CLUSTER ANALYSIS ===
        cluster_summary = self.rfm_df.groupby('Cluster').agg({
            'Recency': ['mean', 'count'],
            'Frequency': 'mean',
            'Monetary': 'mean'
        }).round(2)
        
        print("\n📊 Cluster Summary:")
        print(cluster_summary)
        
        # === SEGMENT LABELING ===
        # Label clusters based on business logic
        cluster_labels = self._label_clusters(cluster_summary)
        self.rfm_df['Segment'] = self.rfm_df['Cluster'].map(cluster_labels)
        
        # === VISUALIZATION ===
        self._visualize_clusters(rfm_scaled)
        
        return self.rfm_df
    
    def _label_clusters(self, cluster_summary):
        """
        Assign business-meaningful labels to clusters based on RFM characteristics.
        
        Args:
            cluster_summary (pd.DataFrame): Summary statistics for each cluster
            
        Returns:
            dict: Mapping from cluster ID to segment label
        """
        
        labels = {}
        
        # Analyze each cluster and assign appropriate label
        for cluster in cluster_summary.index:
            recency = cluster_summary.loc[cluster, ('Recency', 'mean')]
            frequency = cluster_summary.loc[cluster, ('Frequency', 'mean')]
            monetary = cluster_summary.loc[cluster, ('Monetary', 'mean')]
            
            # Business logic for segment assignment
            if recency < 30 and frequency > 20 and monetary > 1000:
                # Recent, frequent, high-value customers
                labels[cluster] = 'High-Value'
            elif recency > 150:
                # Customers who haven't purchased recently
                labels[cluster] = 'At-Risk'
            elif frequency > 10 and monetary > 500:
                # Moderately active customers
                labels[cluster] = 'Regular'
            else:
                # Infrequent customers
                labels[cluster] = 'Occasional'
        
        return labels
    
    def _visualize_clusters(self, rfm_scaled):
        """
        Visualize clusters using PCA and create segment distribution plots.
        
        Args:
            rfm_scaled (np.array): Scaled RFM data
        """
        
        # === PCA VISUALIZATION ===
        # Reduce dimensions to 2D for visualization
        pca = PCA(n_components=2)
        rfm_pca = pca.fit_transform(rfm_scaled)
        
        plt.figure(figsize=(10, 8))
        
        # Create scatter plot with cluster colors
        scatter = plt.scatter(rfm_pca[:, 0], rfm_pca[:, 1], 
                            c=self.rfm_df['Cluster'], 
                            cmap='viridis', 
                            alpha=0.6,
                            edgecolors='black',
                            linewidth=0.5,
                            s=50)
        
        # Add cluster centers
        centers_pca = pca.transform(self.kmeans_model.cluster_centers_)
        plt.scatter(centers_pca[:, 0], centers_pca[:, 1], 
                   c='red', s=200, alpha=0.8, 
                   edgecolors='black', linewidth=2,
                   marker='X', label='Centroids')
        
        plt.title('Customer Segments (PCA Visualization)', fontsize=16, fontweight='bold')
        plt.xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.colorbar(scatter, label='Cluster')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # === SEGMENT DISTRIBUTION ===
        plt.figure(figsize=(10, 6))
        segment_counts = self.rfm_df['Segment'].value_counts()
        colors = plt.cm.Set3(range(len(segment_counts)))
        
        # Create bar chart with value labels
        bars = plt.bar(segment_counts.index, segment_counts.values, 
                      color=colors, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}\n({height/len(self.rfm_df)*100:.1f}%)',
                    ha='center', va='bottom', fontweight='bold')
        
        plt.title('Customer Segment Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Segment')
        plt.ylabel('Number of Customers')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.show()
    
    # ============================================================================
    # PRODUCT RECOMMENDATION SYSTEM
    # ============================================================================
    
    def build_recommendation_system(self):
        """
        Build an item-based collaborative filtering recommendation system.
        
        This method:
        1. Creates a customer-product interaction matrix
        2. Calculates item-item similarity using cosine similarity
        3. Creates product mappings for easy lookup
        4. Visualizes the similarity matrix
        
        The resulting system can recommend products based on what customers
        frequently buy together.
        """
        
        print("\n🎁 Building recommendation system...")
        
        # === CREATE INTERACTION MATRIX ===
        print("Creating customer-product interaction matrix...")
        
        # Pivot table: rows=customers, columns=products, values=quantity
        customer_product_matrix = self.df.pivot_table(
            index='CustomerID',
            columns='StockCode',
            values='Quantity',
            aggfunc='sum',        # Sum quantities for multiple purchases
            fill_value=0          # Fill missing values with 0
        )
        
        print(f"Interaction matrix shape: {customer_product_matrix.shape}")
        
        # Convert to sparse matrix for memory efficiency
        customer_product_sparse = csr_matrix(customer_product_matrix.values)
        
        # === CALCULATE PRODUCT SIMILARITY ===
        print("Calculating product similarities using cosine similarity...")
        
        # Calculate item-item similarity (transpose to get product-product similarity)
        product_similarity = cosine_similarity(customer_product_matrix.T)
        
        # Convert to DataFrame for easy access
        self.product_similarity_matrix = pd.DataFrame(
            product_similarity,
            index=customer_product_matrix.columns,
            columns=customer_product_matrix.columns
        )
        
        print(f"Similarity matrix shape: {self.product_similarity_matrix.shape}")
        
        # === CREATE PRODUCT MAPPINGS ===
        # Create bidirectional mappings between product names and stock codes
        product_names = self.df[['StockCode', 'Description']].drop_duplicates()
        
        # Product name to stock code mapping
        self.product_mapping = dict(zip(product_names['Description'], product_names['StockCode']))
        
        # Stock code to product name mapping (reverse mapping)
        self.reverse_product_mapping = dict(zip(product_names['StockCode'], product_names['Description']))
        
        print(f"✅ Recommendation system built with {len(self.product_similarity_matrix)} products")
        
        # === VISUALIZE SIMILARITY MATRIX SAMPLE ===
        self._visualize_similarity_matrix()
    
    def _visualize_similarity_matrix(self):
        """Visualize a sample of the product similarity matrix."""
        
        # Take a sample of 20x20 for visualization
        sample_products = list(self.product_similarity_matrix.index[:20])
        sample_similarity = self.product_similarity_matrix.loc[sample_products, sample_products]
        
        plt.figure(figsize=(12, 10))
        
        # Create heatmap
        sns.heatmap(sample_similarity, 
                   cmap='coolwarm', 
                   center=0,
                   annot=False,  # Don't show values (too cluttered)
                   cbar_kws={'label': 'Similarity Score'},
                   square=True)
        
        plt.title('Sample Product Similarity Matrix (20×20)', fontsize=16, fontweight='bold')
        plt.xlabel('Product Stock Code')
        plt.ylabel('Product Stock Code')
        plt.tight_layout()
        plt.show()
    
    def get_product_recommendations(self, product_name, n_recommendations=5):
        """
        Get product recommendations for a given product.
        
        Args:
            product_name (str): Name of the product to get recommendations for
            n_recommendations (int): Number of recommendations to return
            
        Returns:
            list: List of dictionaries containing recommended products and scores,
                  or None if product not found
        """
        
        try:
            # === FIND PRODUCT ===
            if product_name in self.product_mapping:
                stock_code = self.product_mapping[product_name]
            else:
                # Try partial matching for user convenience
                matching_products = [desc for desc in self.product_mapping.keys() 
                                   if product_name.lower() in desc.lower()]
                if matching_products:
                    product_name = matching_products[0]
                    stock_code = self.product_mapping[product_name]
                    print(f"Using closest match: {product_name}")
                else:
                    print(f"Product '{product_name}' not found in database")
                    return None
            
            # === GET RECOMMENDATIONS ===
            if stock_code in self.product_similarity_matrix.index:
                # Get similarity scores for this product
                sim_scores = self.product_similarity_matrix[stock_code].sort_values(ascending=False)
                
                # Get top N similar products (excluding the product itself)
                top_products = sim_scores.iloc[1:n_recommendations+1]
                
                # Convert back to product names with scores
                recommendations = []
                for prod_code, score in top_products.items():
                    if prod_code in self.reverse_product_mapping:
                        recommendations.append({
                            'product': self.reverse_product_mapping[prod_code],
                            'similarity_score': score,
                            'stock_code': prod_code
                        })
                
                return recommendations
            else:
                print(f"Product '{product_name}' not found in similarity matrix")
                return None
                
        except Exception as e:
            print(f"Error getting recommendations: {e}")
            return None
    
    # ============================================================================
    # MODEL PERSISTENCE
    # ============================================================================
    
    def save_models(self, output_dir='./models'):
        """
        Save all trained models and preprocessed data for deployment.
        
        This method saves:
        - K-means clustering model
        - StandardScaler for RFM normalization
        - Product similarity matrix
        - Product mappings
        - Cluster labels
        - RFM statistics for reference
        
        Args:
            output_dir (str): Directory to save models (created if doesn't exist)
        """
        
        import os
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print(f"\n💾 Saving models to {output_dir}...")
        
        # === SAVE CLUSTERING MODELS ===
        # Save K-means model using joblib (recommended for scikit-learn models)
        joblib.dump(self.kmeans_model, f'{output_dir}/kmeans_model.pkl')
        
        # Save StandardScaler
        joblib.dump(self.scaler, f'{output_dir}/scaler.pkl')
        
        # === SAVE RECOMMENDATION SYSTEM ===
        # Save product similarity matrix using pandas pickle
        self.product_similarity_matrix.to_pickle(f'{output_dir}/product_similarity.pkl')
        
        # Save product mappings
        with open(f'{output_dir}/product_mapping.pkl', 'wb') as f:
            pickle.dump({
                'product_to_code': self.product_mapping,
                'code_to_product': self.reverse_product_mapping
            }, f)
        
        # === SAVE METADATA ===
        # Save cluster to segment label mapping
        cluster_labels = self.rfm_df.groupby('Cluster')['Segment'].first().to_dict()
        with open(f'{output_dir}/cluster_labels.pkl', 'wb') as f:
            pickle.dump(cluster_labels, f)
        
        # Save RFM statistics for reference and validation
        rfm_stats = {
            'mean': self.rfm_df[['Recency', 'Frequency', 'Monetary']].mean().to_dict(),
            'std': self.rfm_df[['Recency', 'Frequency', 'Monetary']].std().to_dict(),
            'min': self.rfm_df[['Recency', 'Frequency', 'Monetary']].min().to_dict(),
            'max': self.rfm_df[['Recency', 'Frequency', 'Monetary']].max().to_dict()
        }
        with open(f'{output_dir}/rfm_stats.pkl', 'wb') as f:
            pickle.dump(rfm_stats, f)
        
        print("✅ All models saved successfully!")
        
        # === PRINT SAVE SUMMARY ===
        print("\n📋 Saved files:")
        print("- kmeans_model.pkl: K-means clustering model")
        print("- scaler.pkl: StandardScaler for RFM normalization")
        print("- product_similarity.pkl: Product similarity matrix")
        print("- product_mapping.pkl: Product name to code mappings")
        print("- cluster_labels.pkl: Cluster to segment label mapping")
        print("- rfm_stats.pkl: RFM statistics for reference")

# ================================================================================
# MAIN EXECUTION
# ================================================================================

if __name__ == "__main__":
    """
    Main execution pipeline for the Shopper Spectrum analytics engine.
    
    This script performs the complete machine learning pipeline:
    1. Data loading and preprocessing
    2. Exploratory data analysis
    3. RFM analysis and customer segmentation
    4. Product recommendation system development
    5. Model testing and validation
    6. Model persistence for deployment
    """
    
    # ===== CONFIGURATION =====
    # Update this path to point to your dataset
    DATA_PATH = r"D:\Machine Learning\Shopper Spectrum Project\online_retail.csv"
    
    print("🚀 Starting Shopper Spectrum Analytics Engine...")
    print("=" * 60)
    
    # ===== INITIALIZE ANALYZER =====
    analyzer = EcommerceAnalyzer(DATA_PATH)
    
    # ===== STEP 1 & 2: DATA LOADING AND PREPROCESSING =====
    print("\n🔄 STEP 1-2: Data Loading and Preprocessing")
    print("-" * 50)
    df = analyzer.load_and_preprocess_data()
    
    # ===== STEP 3: EXPLORATORY DATA ANALYSIS =====
    print("\n🔄 STEP 3: Exploratory Data Analysis")
    print("-" * 50)
    analyzer.perform_eda()
    
    # ===== STEP 4: RFM ANALYSIS AND CLUSTERING =====
    print("\n🔄 STEP 4: RFM Analysis and Customer Segmentation")
    print("-" * 50)
    rfm_df = analyzer.calculate_rfm()
    clustered_df = analyzer.perform_clustering()
    
    # ===== STEP 5: RECOMMENDATION SYSTEM =====
    print("\n🔄 STEP 5: Building Product Recommendation System")
    print("-" * 50)
    analyzer.build_recommendation_system()
    
    # ===== STEP 6: MODEL TESTING =====
    print("\n🔄 STEP 6: Testing Recommendation System")
    print("-" * 50)
    
    # Test with sample products
    test_products = [
        'VINTAGE UNION JACK BUNTING', 
        'WHITE HANGING HEART T-LIGHT HOLDER', 
        'FRENCH ENAMEL CANDLEHOLDER'
    ]
    
    for product in test_products:
        print(f"\n📦 Recommendations for: {product}")
        recommendations = analyzer.get_product_recommendations(product)
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec['product']} (Similarity: {rec['similarity_score']:.3f})")
        else:
            print("   No recommendations found")
    
    # ===== STEP 7: MODEL PERSISTENCE =====
    print("\n🔄 STEP 7: Saving Models for Deployment")
    print("-" * 50)
    analyzer.save_models()
    
    # ===== COMPLETION MESSAGE =====
    print("\n" + "=" * 60)
    print("🎉 ANALYSIS COMPLETE!")
    print("✅ All models have been trained and saved")
    print("✅ Ready for Streamlit dashboard deployment")
    print("✅ Check ./models/ directory for saved files")
    print("=" * 60)