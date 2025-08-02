"""
Shopper Spectrum - E-commerce Analytics Dashboard
================================================

A comprehensive Streamlit web application for e-commerce customer analytics and product recommendations.
This dashboard provides:
- AI-powered product recommendations using collaborative filtering
- Customer segmentation using RFM (Recency, Frequency, Monetary) analysis
- Business intelligence insights and visualizations
- Advanced analytics for customer lifetime value and churn prediction

Author: Arunov Chakraborty
Created: 2nd August, 2025
Dependencies: streamlit, pandas, numpy, plotly, scikit-learn, joblib
"""

# Core Libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from datetime import datetime

# Visualization Libraries
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ================================================================================
# STREAMLIT PAGE CONFIGURATION
# ================================================================================

# Configure the Streamlit page with custom settings
st.set_page_config(
    page_title="🛒 Shopper Spectrum",      # Browser tab title
    page_icon="🛍️",                        # Browser tab icon
    layout="wide",                          # Use full width of the screen
    initial_sidebar_state="expanded"        # Show sidebar by default
)

# ================================================================================
# CUSTOM CSS STYLING
# ================================================================================

# Enhanced CSS for professional UI/UX with dark theme support
st.markdown("""
<style>
    /* ===== CSS VARIABLES FOR CONSISTENT THEMING ===== */
    :root {
        --primary-color: #1e3d59;          /* Main brand color */
        --secondary-color: #3e5c76;        /* Secondary brand color */
        --accent-color: #f5f0e1;           /* Accent color */
        --success-color: #28a745;          /* Success/positive color */
        --warning-color: #ffc107;          /* Warning color */
        --danger-color: #dc3545;           /* Error/danger color */
        --info-color: #17a2b8;             /* Information color */
        --text-primary: #f9fafb;           /* Primary text color (bright) */
        --text-secondary: #e5e7eb;         /* Secondary text color */
        --text-light: #f3f4f6;             /* Light text color */
        --background: #181c24;             /* Dark background */
    }

    /* ===== GLOBAL STYLES ===== */
    html, body {
        color: #f9fafb !important;
        background-color: #181c24 !important;
    }
    
    /* Header styles with enhanced visibility */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    
    /* Sidebar text styling */
    .sidebar .sidebar-content, .css-1d391kg, .css-ffhzg2 {
        color: #f9fafb !important;
    }
    
    /* Markdown container text */
    .stMarkdown {
        color: var(--text-primary) !important;
    }
    
    /* Enhanced font rendering */
    * {
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        text-rendering: optimizeLegibility;
    }
    
    /* Override streamlit's transparent backgrounds */
    [data-testid="stMarkdownContainer"] > div {
        background-color: transparent !important;
    }
    
    /* ===== COMPONENT SPECIFIC STYLES ===== */
    
    /* Main application header */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1e3d59 0%, #3e5c76 100%);
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        animation: fadeIn 1s ease-in;
    }
    
    /* Section sub-headers */
    .sub-header {
        font-size: 2rem;
        color: #1e3d59;
        margin-bottom: 1.5rem;
        font-weight: 600;
    }
    
    /* ===== BUTTON STYLING ===== */
    .stButton>button {
        background-color: #1e3d59 !important;
        color: white !important;
        font-size: 1.1rem;
        font-weight: 600;
        padding: 0.75rem 2.5rem;
        border-radius: 10px;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(30, 61, 89, 0.3);
    }
    
    /* Button hover effects */
    .stButton>button:hover {
        background-color: #3e5c76 !important;
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(30, 61, 89, 0.4);
    }
    
    /* ===== RECOMMENDATION CARD STYLING ===== */
    .recommendation-card {
        background: #ffffff !important;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #1e3d59;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        color: #2c3e50 !important;
        position: relative;
        overflow: hidden;
    }
    
    /* Recommendation card hover effect */
    .recommendation-card:hover {
        transform: translateX(10px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    /* Recommendation card text styling */
    .recommendation-card strong {
        color: #1e3d59 !important;
        font-size: 1.2rem;
        display: block;
        margin-bottom: 0.5rem;
    }
    
    .recommendation-card small {
        color: #6c757d !important;
        font-size: 0.9rem;
        display: block;
        margin-top: 0.5rem;
    }
    
    /* Recommendation number badge */
    .recommendation-number {
        position: absolute;
        top: 10px;
        right: 15px;
        background: #1e3d59;
        color: white !important;
        width: 35px;
        height: 35px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 1.1rem;
    }
    
    /* ===== METRIC CARD STYLING ===== */
    .metric-card {
        background: #f8f9fa !important;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }
    
    /* Metric card hover effects */
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        border-color: #1e3d59;
    }
    
    /* Metric card values */
    .metric-card h3 {
        color: #1e3d59 !important;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    /* Metric card labels */
    .metric-card p {
        color: #6c757d !important;
        font-size: 1rem;
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    .metric-card small {
        color: #6c757d !important;
        font-size: 0.8rem;
        display: block;
        margin-top: 0.5rem;
    }
    
    /* ===== CUSTOMER SEGMENT BADGES ===== */
    .segment-badge {
        padding: 1rem 2rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.3rem;
        text-align: center;
        margin: 1.5rem auto;
        max-width: 400px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.2);
        text-transform: uppercase;
        letter-spacing: 1px;
        animation: pulse 2s infinite;
    }
    
    /* Pulse animation for segment badges */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    /* Segment-specific colors */
    .high-value {
        background-color: #28a745 !important;
        color: white !important;
    }
    
    .regular {
        background-color: #17a2b8 !important;
        color: white !important;
    }
    
    .occasional {
        background-color: #ffc107 !important;
        color: #212529 !important;
    }
    
    .at-risk {
        background-color: #dc3545 !important;
        color: white !important;
    }
    
    /* ===== FEATURE CARD STYLING ===== */
    .feature-card {
        background: #ffffff !important;
        background-color: #ffffff !important;
        padding: 2rem !important;
        border-radius: 15px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
        transition: all 0.3s ease;
        height: 100%;
        color: #2c3e50 !important;
        border: 1px solid #e9ecef !important;
        display: block !important;
        position: relative !important;
        overflow: visible !important;
    }
    
    /* Feature card hover effects */
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.12);
    }
    
    /* Feature card headings */
    .feature-card h3 {
        color: #1e3d59 !important;
        margin-bottom: 1rem;
        font-size: 1.5rem;
        font-weight: 600;
    }
    
    /* Feature card text */
    .feature-card p {
        color: #495057 !important;
        font-size: 1rem;
        line-height: 1.6;
        margin-bottom: 0.5rem;
        font-weight: 400;
    }
    
    /* Feature card lists */
    .feature-card ul {
        color: #495057 !important;
        padding-left: 1.5rem !important;
        margin-top: 1rem !important;
        list-style: disc !important;
    }
    
    /* Feature card emphasized text */
    .feature-card p strong {
        color: #1e3d59 !important;
        font-weight: 700;
        font-size: 1.05rem;
    }
    
    /* ===== ANIMATIONS ===== */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* ===== SIMILARITY SCORE BAR ===== */
    .similarity-bar {
        background: #e9ecef;
        border-radius: 10px;
        height: 8px;
        margin-top: 0.5rem;
        overflow: hidden;
    }
    
    .similarity-fill {
        background-color: #1e3d59 !important;
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease;
    }
    
    /* ===== INFO BOX STYLING ===== */
    .info-box {
        background: #e3f2fd !important;
        border-left: 5px solid #1e3d59;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: #1e3d59 !important;
    }
    
    .info-box strong {
        color: #0d2137 !important;
        font-weight: 700;
    }
    
    .info-box p {
        color: #1e3d59 !important;
        margin: 0;
        line-height: 1.6;
    }
    
    /* ===== DARK MODE SUPPORT ===== */
    @media (prefers-color-scheme: dark) {
        .recommendation-card {
            background: #2c3e50 !important;
            color: #ecf0f1 !important;
        }
        
        .recommendation-card strong {
            color: #3498db !important;
        }
        
        .metric-card {
            background: #2c3e50 !important;
        }
        
        .metric-card h3 {
            color: #3498db !important;
        }
        
        .metric-card p {
            color: #ecf0f1 !important;
        }
        
        .feature-card {
            background: #2c3e50 !important;
            color: #ecf0f1 !important;
        }
        
        .feature-card h3 {
            color: #3498db !important;
        }
        
        .feature-card p, .feature-card li {
            color: #ecf0f1 !important;
        }
        
        .info-box {
            background: #34495e !important;
            color: #ecf0f1 !important;
        }
        
        .info-box strong {
            color: #3498db !important;
        }
        
        .info-box p {
            color: #ecf0f1 !important;
        }
    }
    
    /* ===== STREAMLIT SPECIFIC OVERRIDES ===== */
    .stSelectbox label {
        color: var(--text-primary) !important;
    }
    
    .stNumberInput label {
        color: var(--text-primary) !important;
    }
    
    .element-container div[data-testid="stMarkdownContainer"] p {
        color: inherit !important;
    }
    
    /* ===== VISIBILITY FIXES ===== */
    /* Force text visibility in all custom divs */
    div[class*="card"] * {
        opacity: 1 !important;
        visibility: visible !important;
    }
    
    /* Comprehensive visibility fixes */
    .feature-card, .metric-card, .recommendation-card {
        background-color: #ffffff !important;
        isolation: isolate !important;
        mix-blend-mode: normal !important;
    }
    
    .info-box {
        background-color: #e3f2fd !important;
        isolation: isolate !important;
        mix-blend-mode: normal !important;
    }
    
    /* Ensure text is always readable */
    .feature-card *, .metric-card *, .recommendation-card *, .info-box * {
        color: inherit !important;
        opacity: 1 !important;
        mix-blend-mode: normal !important;
    }
    
    /* Final fallback for all text in cards */
    .feature-card h3, .feature-card p, .feature-card li,
    .metric-card h3, .metric-card p, .metric-card small,
    .recommendation-card strong, .recommendation-card small,
    .info-box p, .info-box strong {
        opacity: 1 !important;
        visibility: visible !important;
    }
    
    /* Ensure all text elements are visible */
    h1, h2, h3, h4, h5, h6, p, span, li, strong, small {
        opacity: 1 !important;
        visibility: visible !important;
        position: relative;
        z-index: 1;
    }
    
    /* Fix for Streamlit's rendering issues */
    .element-container {
        position: relative;
        z-index: 1;
    }
</style>
""", unsafe_allow_html=True)

# ================================================================================
# MODEL LOADING FUNCTIONS
# ================================================================================

@st.cache_resource
def load_models():
    """
    Load all saved ML models and preprocessed data from disk.
    
    This function loads:
    - K-means clustering model for customer segmentation
    - StandardScaler for RFM normalization
    - Product similarity matrix for recommendations
    - Product name to code mappings
    - Cluster labels for segment identification
    - RFM statistics for reference
    
    Returns:
        dict: Dictionary containing all loaded models and data
        None: If models cannot be loaded
    """
    try:
        # Load clustering model and scaler
        kmeans_model = joblib.load('./models/kmeans_model.pkl')
        scaler = joblib.load('./models/scaler.pkl')
        
        # Load product similarity matrix and mappings
        product_similarity = pd.read_pickle('./models/product_similarity.pkl')
        
        # Load product mappings (name ↔ code conversion)
        with open('./models/product_mapping.pkl', 'rb') as f:
            mappings = pickle.load(f)
        
        # Load cluster to segment label mappings
        with open('./models/cluster_labels.pkl', 'rb') as f:
            cluster_labels = pickle.load(f)
            
        # Load RFM statistics for reference and validation
        with open('./models/rfm_stats.pkl', 'rb') as f:
            rfm_stats = pickle.load(f)
        
        return {
            'kmeans': kmeans_model,
            'scaler': scaler,
            'similarity_matrix': product_similarity,
            'product_to_code': mappings['product_to_code'],
            'code_to_product': mappings['code_to_product'],
            'cluster_labels': cluster_labels,
            'rfm_stats': rfm_stats
        }
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Please ensure all model files are in the './models' directory")
        return None

# ================================================================================
# RECOMMENDATION SYSTEM FUNCTIONS
# ================================================================================

def get_recommendations(product_name, models, n_recommendations=5):
    """
    Generate product recommendations using collaborative filtering.
    
    This function uses item-based collaborative filtering to find products
    that are frequently bought together with the input product.
    
    Args:
        product_name (str): Name of the product to get recommendations for
        models (dict): Dictionary containing loaded models and data
        n_recommendations (int): Number of recommendations to return
        
    Returns:
        tuple: (recommendations_list, matched_product_name) or (None, product_name)
    """
    try:
        # Extract required mappings and similarity matrix
        product_to_code = models['product_to_code']
        code_to_product = models['code_to_product']
        similarity_matrix = models['similarity_matrix']
        
        # Find exact product match
        if product_name in product_to_code:
            stock_code = product_to_code[product_name]
        else:
            # Try partial matching for user convenience
            matching_products = [desc for desc in product_to_code.keys() 
                               if product_name.lower() in desc.lower()]
            if matching_products:
                product_name = matching_products[0]
                stock_code = product_to_code[product_name]
                st.info(f"Using closest match: {product_name}")
            else:
                return None, product_name
        
        # Generate recommendations using similarity scores
        if stock_code in similarity_matrix.index:
            # Get similarity scores for the product (sorted descending)
            sim_scores = similarity_matrix[stock_code].sort_values(ascending=False)
            
            # Get top N products (excluding the product itself)
            top_products = sim_scores.iloc[1:n_recommendations+1]
            
            # Convert product codes back to names with scores
            recommendations = []
            for prod_code, score in top_products.items():
                if prod_code in code_to_product:
                    recommendations.append({
                        'product': code_to_product[prod_code],
                        'similarity_score': score,
                        'stock_code': prod_code
                    })
            
            return recommendations, product_name
        else:
            return None, product_name
            
    except Exception as e:
        st.error(f"Error getting recommendations: {e}")
        return None, product_name

# ================================================================================
# CUSTOMER SEGMENTATION FUNCTIONS
# ================================================================================

def predict_customer_segment(recency, frequency, monetary, models):
    """
    Predict customer segment based on RFM (Recency, Frequency, Monetary) values.
    
    This function uses the trained K-means model to classify customers into segments
    such as High-Value, Regular, Occasional, or At-Risk.
    
    Args:
        recency (int): Days since last purchase
        frequency (int): Number of purchases
        monetary (float): Total amount spent
        models (dict): Dictionary containing loaded models and data
        
    Returns:
        tuple: (cluster_id, segment_name, confidence_score) or (None, None, None)
    """
    try:
        # Prepare input data as numpy array
        input_data = np.array([[recency, frequency, monetary]])
        
        # Scale the input using the same scaler used during training
        scaled_input = models['scaler'].transform(input_data)
        
        # Predict cluster using trained K-means model
        cluster = models['kmeans'].predict(scaled_input)[0]
        
        # Get human-readable segment label
        segment = models['cluster_labels'].get(cluster, 'Unknown')
        
        # Calculate prediction confidence based on distance to cluster center
        distances = models['kmeans'].transform(scaled_input)[0]
        min_distance = distances[cluster]
        
        # Convert distance to confidence percentage (closer = higher confidence)
        confidence = max(0, 1 - (min_distance / np.max(distances))) * 100
        
        return cluster, segment, confidence
    except Exception as e:
        st.error(f"Error predicting segment: {e}")
        return None, None, None

# ================================================================================
# MAIN APPLICATION
# ================================================================================

def main():
    """
    Main application function that handles the Streamlit interface and routing.
    
    This function sets up the main dashboard interface with:
    - Header and navigation
    - Page routing for different modules
    - Sidebar with metrics and navigation
    """
    
    # Main application header with enhanced styling
    st.markdown('<h1 class="main-header">🛒 Shopper Spectrum</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;">Advanced Customer Analytics & Intelligent Product Recommendations</p>', unsafe_allow_html=True)
    
    # Load all required models and data
    models = load_models()
    if models is None:
        st.stop()  # Stop execution if models cannot be loaded
    
    # ================================================================================
    # SIDEBAR NAVIGATION
    # ================================================================================
    
    st.sidebar.markdown("## 📊 Navigation")
    page = st.sidebar.radio("Select Module", 
                           ["🏠 Home", "🎁 Product Recommendations", "👥 Customer Segmentation", 
                            "📈 Insights", "🔍 Advanced Analytics"])
    
    # Add sidebar metrics for quick reference
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Quick Stats")
    total_products = len(models['product_to_code'])
    total_segments = len(models['cluster_labels'])
    st.sidebar.metric("Total Products", f"{total_products:,}")
    st.sidebar.metric("Customer Segments", total_segments)
    
    # ================================================================================
    # PAGE ROUTING
    # ================================================================================
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "🎁 Product Recommendations":
        show_recommendations_page(models)
    elif page == "👥 Customer Segmentation":
        show_segmentation_page(models)
    elif page == "📈 Insights":
        show_insights_page(models)
    elif page == "🔍 Advanced Analytics":
        show_advanced_analytics_page()

def show_home_page():
    """Display the home page with platform overview and features."""
    
    st.markdown("## Welcome to Shopper Spectrum Analytics Platform!")
    
    # Hero section with centered content
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background-color: #f5f7fa; border-radius: 20px; margin-bottom: 2rem; border: 1px solid #e9ecef;">
            <h3 style="color: #1e3d59 !important; margin-bottom: 1rem; font-weight: 600;">Transform Your E-Commerce Business</h3>
            <p style="color: #495057 !important; font-size: 1.1rem; margin: 0;">Leverage AI-powered insights to understand customers better and boost sales</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature showcase section
    st.markdown("### 🚀 Platform Features")
    
    col1, col2, col3 = st.columns(3)
    
    # Feature 1: Smart Recommendations
    with col1:
        with st.container():
            st.markdown("""
            <div class="feature-card" style="background-color: #ffffff !important;">
                <h3 style="color: #1e3d59 !important;">🎯 Smart Recommendations</h3>
                <p style="color: #495057 !important;"><strong style="color: #1e3d59 !important;">AI-powered product suggestions based on customer purchase patterns and behavior analysis</strong></p>
                <ul style="color: #495057 !important;">
                    <li style="color: #495057 !important;">Item-based collaborative filtering</li>
                    <li style="color: #495057 !important;">Real-time similarity scoring</li>
                    <li style="color: #495057 !important;">Cross-selling opportunities</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Feature 2: RFM Segmentation
    with col2:
        with st.container():
            st.markdown("""
            <div class="feature-card" style="background-color: #ffffff !important;">
                <h3 style="color: #1e3d59 !important;">👥 RFM Segmentation</h3>
                <p style="color: #495057 !important;"><strong style="color: #1e3d59 !important;">Advanced customer segmentation using Recency, Frequency, and Monetary analysis</strong></p>
                <ul style="color: #495057 !important;">
                    <li style="color: #495057 !important;">4 distinct customer segments</li>
                    <li style="color: #495057 !important;">Predictive clustering</li>
                    <li style="color: #495057 !important;">Actionable insights</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Feature 3: Business Intelligence
    with col3:
        with st.container():
            st.markdown("""
            <div class="feature-card" style="background-color: #ffffff !important;">
                <h3 style="color: #1e3d59 !important;">📊 Business Intelligence</h3>
                <p style="color: #495057 !important;"><strong style="color: #1e3d59 !important;">Comprehensive analytics dashboard with real-time metrics and visualizations</strong></p>
                <ul style="color: #495057 !important;">
                    <li style="color: #495057 !important;">Interactive dashboards</li>
                    <li style="color: #495057 !important;">Trend analysis</li>
                    <li style="color: #495057 !important;">Performance metrics</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Business impact metrics section
    st.markdown("### 📈 Expected Business Impact")
    col1, col2, col3, col4 = st.columns(4)
    
    # Define expected business metrics
    metrics = [
        ("15-25%", "Revenue Increase", "Through personalized recommendations"),
        ("30%", "Customer Retention", "By identifying at-risk customers"),
        ("40%", "Marketing ROI", "With targeted campaigns"),
        ("2.5x", "Conversion Rate", "Using customer insights")
    ]
    
    # Display metrics in cards
    for col, (value, label, desc) in zip([col1, col2, col3, col4], metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #1e3d59 !important;">{value}</h3>
                <p style="color: #495057 !important; font-weight: 600;">{label}</p>
                <small style="color: #6c757d !important;">{desc}</small>
            </div>
            """, unsafe_allow_html=True)

def show_recommendations_page(models):
    """Display the product recommendations page."""
    
    st.markdown('<h2 class="sub-header">🎁 Intelligent Product Recommendation Engine</h2>', unsafe_allow_html=True)
    
    # Information box explaining how recommendations work
    st.markdown("""
    <div class="info-box" style="background-color: #e3f2fd !important;">
        <p style="color: #1e3d59 !important; font-weight: 500; margin: 0;"><strong style="color: #0d2137 !important;">How it works:</strong> Our recommendation engine uses collaborative filtering to find products frequently bought together. 
        Enter a product name below to discover related items that your customers are likely to purchase.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Input section with enhanced UI
    col1, col2, col3 = st.columns([2, 3, 1])
    
    with col1:
        # Display total products available
        st.metric("Available Products", f"{len(models['product_to_code']):,}")
    
    with col2:
        # Product selection dropdown with search functionality
        product_list = list(models['product_to_code'].keys())
        product_name = st.selectbox(
            "🔍 Search for a product:",
            options=[""] + sorted(product_list),
            help="Start typing to search through our product catalog"
        )
    
    with col3:
        # Number of recommendations selector
        n_recs = st.number_input("# Recommendations", min_value=3, max_value=10, value=5)
    
    # Recommendation generation button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        recommend_btn = st.button("🚀 Get Recommendations", type="primary", use_container_width=True)
    
    # Process recommendations when button is clicked
    if recommend_btn and product_name:
        with st.spinner("🔮 Finding perfect matches..."):
            recommendations, matched_product = get_recommendations(product_name, models, n_recs)
            
            if recommendations:
                st.success(f"✅ Found {len(recommendations)} recommendations for: **{matched_product}**")
                
                # Display recommendations in enhanced cards
                st.markdown("### 📦 Recommended Products")
                
                # Create two columns for recommendations layout
                for i in range(0, len(recommendations), 2):
                    col1, col2 = st.columns(2)
                    
                    for col, idx in [(col1, i), (col2, i+1)]:
                        if idx < len(recommendations):
                            rec = recommendations[idx]
                            with col:
                                # Enhanced recommendation card with HTML
                                similarity_percent = rec['similarity_score'] * 100
                                st.markdown(f"""
                                <div class="recommendation-card">
                                    <div class="recommendation-number">{idx + 1}</div>
                                    <strong>{rec['product']}</strong>
                                    <small>Stock Code: {rec['stock_code']}</small>
                                    <small>Similarity Score: {rec['similarity_score']:.3f}</small>
                                    <div class="similarity-bar">
                                        <div class="similarity-fill" style="width: {similarity_percent}%;"></div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                
                # Advanced visualizations section
                st.markdown("### 📊 Recommendation Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Similarity scores horizontal bar chart
                    fig_bar = go.Figure(data=[
                        go.Bar(
                            x=[rec['similarity_score'] for rec in recommendations],
                            y=[f"Product {i+1}" for i in range(len(recommendations))],
                            orientation='h',
                            text=[f"{rec['similarity_score']:.3f}" for rec in recommendations],
                            textposition='auto',
                            marker=dict(
                                color=[rec['similarity_score'] for rec in recommendations],
                                colorscale='Blues',
                                showscale=True,
                                colorbar=dict(title="Similarity")
                            )
                        )
                    ])
                    fig_bar.update_layout(
                        title="Similarity Score Distribution",
                        xaxis_title="Similarity Score",
                        yaxis_title="Products",
                        height=400,
                        yaxis=dict(autorange="reversed"),
                        showlegend=False
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                with col2:
                    # Radar chart for top 5 recommendations
                    top_5 = recommendations[:5]
                    categories = [f"Rec {i+1}" for i in range(len(top_5))]
                    
                    fig_radar = go.Figure(data=go.Scatterpolar(
                        r=[rec['similarity_score'] for rec in top_5],
                        theta=categories,
                        fill='toself',
                        name='Similarity Scores'
                    ))
                    
                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 1]
                            )),
                        showlegend=False,
                        title="Top 5 Recommendations Radar"
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)
                
            else:
                st.error("❌ Product not found in database. Please try another product.")
    
    elif recommend_btn:
        st.warning("⚠️ Please select a product first")

def show_segmentation_page(models):
    """Display the customer segmentation page."""
    
    st.markdown('<h2 class="sub-header">👥 Advanced Customer Segmentation Predictor</h2>', unsafe_allow_html=True)
    
    # RFM analysis explanation section
    with st.expander("📚 Understanding RFM Analysis", expanded=True):
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("""
            ### What is RFM?
            
            RFM Analysis is a proven marketing technique that segments customers based on three key metrics:
            
            🕐 **Recency** - How recently did the customer purchase?  
            📊 **Frequency** - How often do they purchase?  
            💰 **Monetary** - How much do they spend?
            """)
        
        with col2:
            st.markdown("""
            ### Why RFM Matters
            
            - **Identify** your most valuable customers
            - **Predict** future purchase behavior
            - **Personalize** marketing campaigns
            - **Optimize** resource allocation
            - **Increase** customer lifetime value
            """)
    
    # RFM input form with interactive sliders
    st.markdown("### 📝 Enter Customer Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Recency input with contextual information
        recency = st.slider(
            "🕐 Recency (days since last purchase)",
            min_value=0,
            max_value=365,
            value=30,
            help="How many days ago was their last purchase?"
        )
        st.info(f"Customer last purchased **{recency}** days ago")
    
    with col2:
        # Frequency input with contextual information
        frequency = st.slider(
            "📊 Frequency (number of purchases)",
            min_value=1,
            max_value=100,
            value=10,
            help="How many times have they purchased?"
        )
        st.info(f"Customer has made **{frequency}** purchases")
    
    with col3:
        # Monetary input with contextual information
        monetary = st.slider(
            "💰 Monetary (total spend $)",
            min_value=0,
            max_value=10000,
            value=500,
            step=50,
            help="What's their total spending?"
        )
        st.info(f"Customer has spent **${monetary:,}**")
    
    # Prediction button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_btn = st.button("🎯 Predict Customer Segment", type="primary", use_container_width=True)
    
    # Process prediction when button is clicked
    if predict_btn:
        with st.spinner("🔍 Analyzing customer profile..."):
            cluster, segment, confidence = predict_customer_segment(recency, frequency, monetary, models)
            
            if segment:
                # Display prediction results
                st.markdown("### 🎯 Prediction Results")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Display segment badge with appropriate styling
                    segment_class = segment.lower().replace('-', '').replace(' ', '-')
                    st.markdown(f"""
                    <div class="segment-badge {segment_class}">
                        {segment} Customer
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Display confidence score
                    st.metric("Confidence Score", f"{confidence:.1f}%")
                
                # Detailed segment information and recommendations
                display_segment_details(segment, recency, frequency, monetary, models)
                
            else:
                st.error("❌ Error predicting segment. Please check your inputs.")

def display_segment_details(segment, recency, frequency, monetary, models):
    """Display detailed information about the predicted customer segment."""
    
    # Comprehensive segment information dictionary
    segment_info = {
        'High-Value': {
            'description': 'Premium customers who are highly engaged and valuable to your business.',
            'characteristics': ['Recent purchases', 'Buy frequently', 'High spending', 'Brand advocates'],
            'strategies': [
                'VIP treatment and exclusive perks',
                'Early access to new products',
                'Personal account managers',
                'Premium loyalty rewards'
            ],
            'icon': '⭐',
            'color': '#28a745'
        },
        'Regular': {
            'description': 'Consistent customers who form the backbone of your business.',
            'characteristics': ['Steady purchasing', 'Moderate spending', 'Good engagement', 'Reliable revenue'],
            'strategies': [
                'Loyalty program enrollment',
                'Regular engagement emails',
                'Seasonal promotions',
                'Cross-selling opportunities'
            ],
            'icon': '💎',
            'color': '#17a2b8'
        },
        'Occasional': {
            'description': 'Customers who purchase infrequently but have potential for growth.',
            'characteristics': ['Sporadic purchases', 'Lower spending', 'Price sensitive', 'Growth potential'],
            'strategies': [
                'Targeted promotions',
                'Bundle offers',
                'Re-engagement campaigns',
                'Educational content'
            ],
            'icon': '🎯',
            'color': '#ffc107'
        },
        'At-Risk': {
            'description': 'Customers showing signs of churn who need immediate attention.',
            'characteristics': ['Long absence', 'Declining activity', 'Low engagement', 'Churn risk'],
            'strategies': [
                'Win-back campaigns',
                'Special discounts',
                'Feedback surveys',
                'Personalized outreach'
            ],
            'icon': '⚠️',
            'color': '#dc3545'
        }
    }
    
    if segment in segment_info:
        info = segment_info[segment]
        
        # Display segment profile and recommended actions
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"### {info['icon']} Segment Profile")
            st.markdown(f"**Description:** {info['description']}")
            
            st.markdown("**Key Characteristics:**")
            for char in info['characteristics']:
                st.markdown(f"• {char}")
        
        with col2:
            st.markdown("### 📋 Recommended Actions")
            for i, strategy in enumerate(info['strategies'], 1):
                st.markdown(f"{i}. {strategy}")
        
        # Advanced visualizations section
        st.markdown("### 📊 Customer Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # RFM Profile Radar Chart
            display_rfm_radar(recency, frequency, monetary, models, info)
        
        with col2:
            # Comparison with segment averages
            display_segment_comparison(segment, recency, frequency, monetary, info)
        
        # Personalized insights section
        display_personalized_insights(recency, frequency, monetary)

def display_rfm_radar(recency, frequency, monetary, models, info):
    """Display RFM profile as a radar chart."""
    
    rfm_stats = models['rfm_stats']
    
    # Normalize values to 0-100 scale for better visualization
    r_norm = min(100, (1 - recency / 365) * 100)  # Higher score for recent purchases
    f_norm = min(100, (frequency / rfm_stats['max']['Frequency']) * 100)
    m_norm = min(100, (monetary / rfm_stats['max']['Monetary']) * 100)
    
    # Create radar chart
    fig_radar = go.Figure(data=go.Scatterpolar(
        r=[r_norm, f_norm, m_norm, r_norm],  # Close the polygon
        theta=['Recency Score', 'Frequency Score', 'Monetary Score', 'Recency Score'],
        fill='toself',
        fillcolor=info['color'],
        opacity=0.6,
        line=dict(color=info['color'], width=3),
        name='Customer Profile'
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                ticksuffix='%'
            )),
        showlegend=False,
        title="RFM Score Profile"
    )
    st.plotly_chart(fig_radar, use_container_width=True)

def display_segment_comparison(segment, recency, frequency, monetary, info):
    """Display comparison between customer and segment averages."""
    
    # Mock segment averages (in production, calculate from actual data)
    segment_avg = {
        'High-Value': [20, 30, 2000],
        'Regular': [45, 15, 800],
        'Occasional': [90, 5, 300],
        'At-Risk': [180, 3, 200]
    }
    
    categories = ['Recency', 'Frequency', 'Monetary']
    customer_values = [recency, frequency, monetary]
    
    # Create comparison bar chart
    fig_comparison = go.Figure()
    
    fig_comparison.add_trace(go.Bar(
        name='Customer',
        x=categories,
        y=customer_values,
        marker_color=info['color']
    ))
    
    fig_comparison.add_trace(go.Bar(
        name=f'{segment} Avg',
        x=categories,
        y=segment_avg.get(segment, [50, 10, 500]),
        marker_color='lightgray'
    ))
    
    fig_comparison.update_layout(
        title="Customer vs Segment Average",
        barmode='group',
        yaxis_title="Value"
    )
    st.plotly_chart(fig_comparison, use_container_width=True)

def display_personalized_insights(recency, frequency, monetary):
    """Display personalized insights based on customer metrics."""
    
    st.markdown("### 💡 Personalized Insights")
    
    insights = []
    
    # Generate insights based on RFM values
    if recency > 60:
        insights.append("⏰ Customer hasn't purchased recently - consider a re-engagement campaign")
    if frequency < 5:
        insights.append("📈 Low purchase frequency - introduce loyalty incentives")
    if monetary < 200:
        insights.append("💵 Below average spending - offer bundle deals")
    
    # Display insights or positive message
    if insights:
        for insight in insights:
            st.info(insight)
    else:
        st.success("✅ This customer is performing well across all metrics!")

def show_insights_page(models):
    """Display the business insights dashboard."""
    
    st.markdown('<h2 class="sub-header">📈 Business Intelligence Dashboard</h2>', unsafe_allow_html=True)
    
    # Create tabbed interface for different insight categories
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "🎯 Segments", "💼 Recommendations"])
    
    with tab1:
        display_rfm_overview(models)
    
    with tab2:
        display_segment_analysis()
    
    with tab3:
        display_strategic_recommendations()

def display_rfm_overview(models):
    """Display RFM metrics overview."""
    
    rfm_stats = models['rfm_stats']
    
    st.markdown("### 📊 RFM Metrics Overview")
    
    # Create metrics display in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🕐 Recency Analysis")
        st.metric("Average Days Since Purchase", f"{rfm_stats['mean']['Recency']:.0f}")
        st.metric("Best Customer", f"{rfm_stats['min']['Recency']:.0f} days")
        st.metric("At-Risk Threshold", f"{rfm_stats['max']['Recency']:.0f} days")
    
    with col2:
        st.markdown("#### 📊 Frequency Analysis")
        st.metric("Average Orders", f"{rfm_stats['mean']['Frequency']:.0f}")
        st.metric("Top Customer", f"{rfm_stats['max']['Frequency']:.0f} orders")
        st.metric("One-time Buyers", f"{rfm_stats['min']['Frequency']:.0f} order")
    
    with col3:
        st.markdown("#### 💰 Monetary Analysis")
        st.metric("Average Lifetime Value", f"${rfm_stats['mean']['Monetary']:,.2f}")
        st.metric("Highest Value", f"${rfm_stats['max']['Monetary']:,.2f}")
        st.metric("Lowest Value", f"${rfm_stats['min']['Monetary']:,.2f}")
    
    # Distribution visualizations
    display_distribution_charts()

def display_distribution_charts():
    """Display RFM distribution charts."""
    
    st.markdown("### 📈 Value Distribution Analysis")
    
    # Create subplots for RFM distributions
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Recency Distribution', 'Frequency Distribution', 'Monetary Distribution')
    )
    
    # Generate sample data for demonstration (in production, use actual data)
    for i, (metric, color) in enumerate([('Recency', 'blue'), ('Frequency', 'green'), ('Monetary', 'red')], 1):
        np.random.seed(42)
        if metric == 'Recency':
            data = np.random.exponential(50, 1000)
        elif metric == 'Frequency':
            data = np.random.poisson(10, 1000)
        else:
            data = np.random.lognormal(6, 1.5, 1000)
        
        fig.add_trace(
            go.Histogram(x=data, name=metric, marker_color=color, showlegend=False),
            row=1, col=i
        )
    
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

def display_segment_analysis():
    """Display customer segment analysis."""
    
    st.markdown("### 🎯 Customer Segment Analysis")
    
    # Sample segment data (in production, calculate from actual customer data)
    segment_data = pd.DataFrame({
        'Segment': ['High-Value', 'Regular', 'Occasional', 'At-Risk'],
        'Count': [250, 450, 300, 200],
        'Percentage': [20.8, 37.5, 25.0, 16.7],
        'Avg_Revenue': [5000, 2000, 800, 400]
    })
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Enhanced pie chart for segment distribution
        fig_pie = px.pie(
            segment_data, 
            values='Count', 
            names='Segment',
            title='Customer Distribution by Segment',
            color_discrete_map={
                'High-Value': '#28a745',
                'Regular': '#17a2b8',
                'Occasional': '#ffc107',
                'At-Risk': '#dc3545'
            }
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.markdown("#### Segment Metrics")
        for _, row in segment_data.iterrows():
            st.metric(
                row['Segment'],
                f"{row['Count']} customers",
                f"${row['Avg_Revenue']:,} avg"
            )
    
    # Revenue contribution analysis
    display_revenue_contribution(segment_data)

def display_revenue_contribution(segment_data):
    """Display revenue contribution by segment."""
    
    st.markdown("### 💰 Revenue Contribution by Segment")
    
    # Calculate total revenue by segment
    fig_revenue = go.Figure(data=[
        go.Bar(
            x=segment_data['Segment'],
            y=segment_data['Count'] * segment_data['Avg_Revenue'],
            text=[f"${v:,.0f}" for v in segment_data['Count'] * segment_data['Avg_Revenue']],
            textposition='auto',
            marker_color=['#28a745', '#17a2b8', '#ffc107', '#dc3545']
        )
    ])
    fig_revenue.update_layout(
        title="Total Revenue by Segment",
        yaxis_title="Revenue ($)",
        showlegend=False
    )
    st.plotly_chart(fig_revenue, use_container_width=True)

def display_strategic_recommendations():
    """Display strategic business recommendations."""
    
    st.markdown("### 💡 Strategic Recommendations")
    
    # Comprehensive business recommendations
    recommendations = [
        {
            'priority': 'High',
            'segment': 'At-Risk',
            'action': 'Launch Immediate Win-Back Campaign',
            'impact': 'Prevent 16.7% customer churn',
            'timeline': '1-2 weeks'
        },
        {
            'priority': 'High',
            'segment': 'High-Value',
            'action': 'Implement VIP Loyalty Program',
            'impact': 'Increase LTV by 25%',
            'timeline': '1 month'
        },
        {
            'priority': 'Medium',
            'segment': 'Occasional',
            'action': 'Create Bundle Offers',
            'impact': 'Boost frequency by 40%',
            'timeline': '2-3 weeks'
        },
        {
            'priority': 'Medium',
            'segment': 'Regular',
            'action': 'Personalized Email Campaigns',
            'impact': 'Improve retention by 15%',
            'timeline': 'Ongoing'
        }
    ]
    
    # Display recommendations with priority-based styling
    for rec in recommendations:
        color = '#dc3545' if rec['priority'] == 'High' else '#ffc107'
        st.markdown(f"""
        <div style="background: {color}20; border-left: 5px solid {color}; padding: 1rem; margin: 1rem 0; border-radius: 10px;">
            <h4 style="margin: 0; color: #333;">{rec['action']}</h4>
            <p style="margin: 0.5rem 0;"><strong>Segment:</strong> {rec['segment']} | <strong>Priority:</strong> {rec['priority']}</p>
            <p style="margin: 0.5rem 0;"><strong>Expected Impact:</strong> {rec['impact']}</p>
            <p style="margin: 0;"><strong>Timeline:</strong> {rec['timeline']}</p>
        </div>
        """, unsafe_allow_html=True)

def show_advanced_analytics_page():
    """Display advanced analytics features."""
    
    st.markdown('<h2 class="sub-header">🔍 Advanced Analytics Suite</h2>', unsafe_allow_html=True)
    
    # Analytics type selector
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["Customer Lifetime Value", "Churn Prediction", "Market Basket Analysis", "Cohort Analysis"]
    )
    
    # Route to specific analysis based on selection
    if analysis_type == "Customer Lifetime Value":
        display_clv_analysis()
    elif analysis_type == "Churn Prediction":
        display_churn_analysis()
    elif analysis_type == "Market Basket Analysis":
        display_market_basket_analysis()
    else:  # Cohort Analysis
        display_cohort_analysis()

def display_clv_analysis():
    """Display Customer Lifetime Value analysis."""
    
    st.markdown("### 💰 Customer Lifetime Value Analysis")
    
    # CLV explanation
    with st.expander("📚 Understanding CLV"):
        st.markdown("""
        **Customer Lifetime Value (CLV)** represents the total revenue you can expect from a customer throughout their relationship with your business.
        
        **Formula:** CLV = Average Order Value × Purchase Frequency × Customer Lifespan
        """)
    
    # Mock CLV data by segment
    clv_data = pd.DataFrame({
        'Segment': ['High-Value', 'Regular', 'Occasional', 'At-Risk'],
        'Avg_CLV': [8500, 3200, 1200, 600],
        'Projected_CLV': [12000, 4500, 1800, 800]
    })
    
    # CLV comparison chart
    fig_clv = go.Figure()
    fig_clv.add_trace(go.Bar(name='Current CLV', x=clv_data['Segment'], y=clv_data['Avg_CLV']))
    fig_clv.add_trace(go.Bar(name='Projected CLV', x=clv_data['Segment'], y=clv_data['Projected_CLV']))
    
    fig_clv.update_layout(
        title="Customer Lifetime Value by Segment",
        yaxis_title="CLV ($)",
        barmode='group'
    )
    st.plotly_chart(fig_clv, use_container_width=True)

def display_churn_analysis():
    """Display churn risk analysis."""
    
    st.markdown("### 🔄 Churn Risk Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # High risk indicators
        st.markdown("#### 🚨 High Risk Indicators")
        risk_factors = [
            "No purchase in 90+ days",
            "Declining order frequency",
            "Reduced average order value",
            "No email engagement"
        ]
        for factor in risk_factors:
            st.markdown(f"• {factor}")
    
    with col2:
        # Churn probability gauge
        churn_prob = 0.68  # Mock value
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=churn_prob * 100,
            title={'text': "Churn Risk Score"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

def display_market_basket_analysis():
    """Display market basket analysis."""
    
    st.markdown("### 🛒 Market Basket Analysis")
    st.info("Discover which products are frequently bought together")
    
    # Mock association rules
    associations = pd.DataFrame({
        'Product A': ['Laptop', 'Phone Case', 'Coffee Maker', 'Running Shoes'],
        'Product B': ['Laptop Bag', 'Screen Protector', 'Coffee Pods', 'Sports Socks'],
        'Support': [0.15, 0.22, 0.18, 0.12],
        'Confidence': [0.85, 0.92, 0.78, 0.88],
        'Lift': [3.2, 4.1, 2.8, 3.5]
    })
    
    # Display as interactive table with styling
    st.dataframe(
        associations.style.background_gradient(subset=['Support', 'Confidence', 'Lift']),
        use_container_width=True
    )

def display_cohort_analysis():
    """Display cohort retention analysis."""
    
    st.markdown("### 📅 Cohort Retention Analysis")
    
    # Create mock cohort data
    cohorts = pd.DataFrame({
        'Cohort': ['Jan 2024', 'Feb 2024', 'Mar 2024', 'Apr 2024'],
        'Month 1': [100, 100, 100, 100],
        'Month 2': [82, 85, 78, 80],
        'Month 3': [68, 72, 65, 70],
        'Month 4': [55, 60, 52, 58]
    })
    
    # Cohort retention heatmap
    fig_cohort = px.imshow(
        cohorts.iloc[:, 1:].values,
        labels=dict(x="Months Since First Purchase", y="Cohort", color="Retention %"),
        x=['Month 1', 'Month 2', 'Month 3', 'Month 4'],
        y=cohorts['Cohort'],
        color_continuous_scale='RdYlGn'
    )
    fig_cohort.update_layout(title="Customer Retention by Cohort")
    st.plotly_chart(fig_cohort, use_container_width=True)

# ================================================================================
# APPLICATION ENTRY POINT
# ================================================================================

if __name__ == "__main__":
    main()