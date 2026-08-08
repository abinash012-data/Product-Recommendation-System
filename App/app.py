import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# --- 1. Page Config ---
st.set_page_config(page_title="Product Recommender", layout="wide")

# --- 2. Load Models with Caching ---
@st.cache_resource
def load_assets():
    try:
        scaler = joblib.load('scaler.joblib')
        kmeans = joblib.load('kmeans_model.joblib')
        product_data = joblib.load('product_data.joblib')
        nn_models = joblib.load('nn_models.joblib')
        return scaler, kmeans, product_data, nn_models
    except FileNotFoundError:
        st.error("Error: Ensure all .joblib files are in the same folder as app.py")
        return None, None, None, None

scaler, kmeans, product_data, nn_models = load_assets()

# Feature order must exactly match notebook training
FEATURES = [
    'avg_rating', 'num_ratings', 'std_rating', 'median_rating',
    'min_rating', 'max_rating', 'frac_positive', 
    'log_num_ratings', 'rating_range'
]

# --- 3. UI Layout ---
st.title("📦 Product Recommendation System")

if product_data is not None:
    # Sidebar Search
    st.sidebar.header("Search Settings")
    selected_id = st.sidebar.selectbox("Select a Product ID", product_data.index.tolist())
    
    # Display product stats
    st.sidebar.info(f"**Current Product:** {selected_id}")
    st.sidebar.write(f"Rating: {product_data.loc[selected_id, 'avg_rating']:.2f} ⭐")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Recommendations")
        if st.button("Get Recommendations"):
        # 1. Identify Cluster
            cluster_id = int(product_data.loc[selected_id, 'cluster'])
        
        # 2. Get the specific NN model for this cluster
            if cluster_id in nn_models:
                nn_model, id_list = nn_models[cluster_id]
            
            # 3. Prepare features
                input_vector = product_data.loc[[selected_id], FEATURES]
                scaled_vector = scaler.transform(input_vector)
            
            # 4. Predict
                distances, indices = nn_model.kneighbors(scaled_vector)
            
            # 5. Show Results
                st.subheader(f"Top 5 Similar Products in Cluster {cluster_id}")
                rec_cols = st.columns(5)
                count = 0
                for idx in indices[0]:
                    neighbor_id = id_list[idx]
                    if neighbor_id == selected_id: continue # Skip itself
                    if count >= 5: break
                
                    info = product_data.loc[neighbor_id]
                    with rec_cols[count]:
                        st.success(f"**{neighbor_id}**")
                        st.write(f"Rating: {info['avg_rating']:.2f}")
                        st.write(f"Reviews: {int(info['num_ratings'])}")
                    count += 1
            else:
                st.warning("No neighbors found for this specific cluster.")

    with col2:
        st.subheader("Cluster Mapping")
        st.write("Visualizing the product's position within the 2D projected feature space.")
        # --- GRAPHICAL REPRESENTATION CODE ---
        # 1. Prepare data for PCA
        scaled_features = scaler.transform(product_data[FEATURES])
        
        # 2. Apply PCA to reduce to 2 dimensions for plotting
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_features)
        
        # Create a plotting dataframe
        plot_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
        plot_df['cluster'] = product_data['cluster'].values
        
        # Find coordinates for the selected product
        selected_idx = product_data.index.get_loc(selected_id)
        selected_pca = plot_df.iloc[selected_idx]

        # 3. Create the Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Scatter plot of all clusters
        sns.scatterplot(
            data=plot_df, x='PC1', y='PC2', hue='cluster', 
            palette='viridis', alpha=0.4, ax=ax, legend='full'
        )
        
        # Highlight the selected product
        ax.scatter(
            selected_pca['PC1'], selected_pca['PC2'], 
            color='red', s=200, marker='*', label='Selected Product', edgecolor='black'
        )
        
        ax.set_title(f"PCA Cluster Map (Selected: {selected_id})")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
        
        # Show plot in Streamlit
        st.pyplot(fig)
