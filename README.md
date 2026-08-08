# Product-Recommendation-System
Machine learning-based product recommendation system that uses product rating features, K-Means clustering, and Nearest Neighbors to generate Top-N similar product recommendations.
# 🛍️ Product Recommendation System

An **item-to-item product recommendation system** built using machine learning techniques to identify products with similar user-rating behavior.

The system transforms raw user–product ratings into meaningful product-level features, groups similar products using **K-Means clustering**, and then applies **Nearest Neighbors** within each cluster to generate personalized-looking product similarity recommendations.

---

## 📌 Project Overview

Recommendation systems are widely used in e-commerce platforms to help users discover products that are similar to products they already interact with.

This project focuses on **product-to-product recommendation** rather than directly predicting a user's rating.

The approach is:

```text
Raw User-Product Ratings
          │
          ▼
    Data Cleaning
          │
          ▼
 Product-Level Features
          │
          ▼
    Feature Scaling
          │
          ▼
    K-Means Clustering
       K = 11
          │
          ▼
 Cluster-Specific
 Nearest Neighbors
          │
          ▼
 Top-N Similar Products
```

---

## 🎯 Objectives

The main objectives of this project are:

* Analyze user-product rating behavior.
* Identify meaningful characteristics of individual products.
* Group products with similar rating behavior.
* Find products that are most similar to a given product.
* Build an efficient item-to-item recommendation engine.
* Evaluate the quality and consistency of the generated recommendations.
* Save the trained models and data artifacts for later use.

---

## 📊 Dataset

The project uses a ratings dataset loaded from:

```text
rating_short.csv
```

The dataset contains four original columns:

| Column      | Description                      |
| ----------- | -------------------------------- |
| `userid`    | Unique identifier of the user    |
| `productid` | Unique identifier of the product |
| `rating`    | User rating for the product      |
| `date`      | Rating timestamp                 |

The dataset contains:

* **78,245 ratings**
* **76,430 unique users**
* **40,228 unique products**
* **No missing values**
* **No duplicate rows**

The `date` column is removed because it is not used by the recommendation approach.

### Rating Distribution

| Rating |  Count |
| -----: | -----: |
|      1 |  9,128 |
|      2 |  4,592 |
|      3 |  6,287 |
|      4 | 14,878 |
|      5 | 43,360 |

The ratings show a strong positive bias, with a median rating of **5.0**.

---

## 🔎 Exploratory Data Analysis

The analysis investigates:

* Rating distribution
* Number of unique users
* Number of unique products
* User activity
* Product activity
* Rating sparsity
* Product-level rating behavior

A major characteristic of the dataset is its sparsity.

Most users have only a small number of ratings, while some products receive substantially more ratings than others.

This makes traditional user-based collaborative filtering less suitable for this dataset.

Instead, the project focuses on **product-level behavioral features**.

---

## 🧮 Product Feature Engineering

The original dataset contains individual user-product interactions.

For clustering, these interactions are aggregated to create **one row per product**.

The following features are generated:

| Feature           | Description                                   |
| ----------------- | --------------------------------------------- |
| `avg_rating`      | Average rating received by the product        |
| `num_ratings`     | Number of ratings received                    |
| `std_rating`      | Standard deviation of ratings                 |
| `median_rating`   | Median rating                                 |
| `min_rating`      | Minimum rating                                |
| `max_rating`      | Maximum rating                                |
| `frac_positive`   | Fraction of ratings ≥ 4                       |
| `log_num_ratings` | Log-scaled rating count                       |
| `rating_range`    | Difference between maximum and minimum rating |

These features capture three important aspects of product behavior:

1. **Rating quality**
2. **Product popularity/activity**
3. **Agreement or disagreement between users**

The raw dataset is therefore reduced from **78,245 user-product interactions to 40,228 product-level observations**.

---

## ⚙️ Feature Scaling

Since the engineered features have different numerical ranges, `StandardScaler` is applied before clustering.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

The resulting feature matrix contains:

```text
40,228 products × 9 features
```

---

## 🤖 Clustering Model

### K-Means Clustering

K-Means is used to group products according to their rating behavior.

Different values of `K` were evaluated using the **Elbow Method** and **Silhouette Score**.

The silhouette scores included:

|      K | Silhouette Score |
| -----: | ---------------: |
|      2 |           0.5789 |
|      3 |           0.6964 |
|      4 |           0.7018 |
|      5 |           0.6996 |
|      6 |           0.6396 |
|      7 |           0.6541 |
|      8 |           0.7107 |
|      9 |           0.7354 |
|     10 |           0.7392 |
| **11** |       **0.7765** |
|     12 |           0.7731 |
|     13 |           0.7706 |
|     14 |           0.7732 |

The final K-Means model therefore uses:

```python
K = 11
```

with:

```python
KMeans(
    n_clusters=11,
    random_state=42,
    n_init=10
)
```

## The selected K = 11 configuration achieved a silhouette score of **0.7765**.

## 🧩 Product Cluster Interpretation

The 11 clusters represent different patterns of product rating behavior.

Examples include:

### Cluster 0 — Perfectly Rated, Low Activity

Products receiving almost exclusively 5-star ratings but having very few ratings.

### Cluster 1 — Poorly Rated, Low Activity

Products with predominantly negative ratings and limited activity.

### Cluster 3 — Highly Rated, High Activity

Popular products with high average ratings but some disagreement among users.

### Cluster 5 — Consistently Good, Low-Medium Activity

Products with high ratings and relatively low rating variance.

### Cluster 9 — Extremely Popular, Mixed Ratings

Highly active products with a good average rating but significant disagreement between users.

The clustering therefore considers more than average rating alone. It also incorporates popularity and rating variability.

---

## 🔬 Comparison of Clustering Models

Several clustering techniques were investigated:

| Model                    | Silhouette Score |
| ------------------------ | ---------------: |
| K-Means (K=11)           |       **0.7765** |
| Birch                    |           0.6236 |
| Agglomerative Clustering |           0.7628 |
| Gaussian Mixture Model   |           0.7047 |
| DBSCAN                   |           0.9038 |

Although DBSCAN produced the highest silhouette score on the sampled/PCA-reduced data, it generated many small clusters, was sensitive to the `eps` parameter, produced noise points, and did not provide direct control over the number of clusters.

## Therefore, **K-Means was selected as the final clustering model**.

# 🧠 Recommendation Engine

K-Means identifies the group to which a product belongs, but it does not directly determine which products are closest to one another.

To solve this, a **Nearest Neighbors model is trained separately inside each cluster**.

```text
Input Product
     │
     ▼
K-Means Cluster
     │
     ▼
Corresponding Cluster
Nearest Neighbors Model
     │
     ▼
Euclidean Distance
     │
     ▼
Top-N Similar Products
```

This two-stage approach combines:

* **K-Means** → coarse grouping
* **Nearest Neighbors** → fine-grained similarity

The notebook uses Euclidean distance for nearest-neighbor search.

---

## 🔧 Recommendation Function

The main recommendation function is:

```python
recommend_similar(productid, top_n=5)
```

Example:

```python
recommend_similar("1400501466")
```

Example output:

```text
[
    '1400501776',
    '7807284382',
    '1615527656',
    '9983891212',
    '1400532736'
]
```

The function:

1. Checks whether the product exists.
2. Finds its K-Means cluster.
3. Retrieves the corresponding nearest-neighbor model.
4. Calculates distances from the input product.
5. Removes the input product itself.
6. Returns the requested number of similar products.

The default output is the **Top-5 similar products**.

---

# 📏 Evaluation

The recommendation engine is evaluated using three metrics.

## 1. Cluster Consistency

Measures the proportion of recommended products that belong to the same cluster as the input product.

```text
Cluster Consistency =
Same-Cluster Recommendations / Total Recommendations
```

For the tested product:

```text
1.0
```

or:

```text
100%
```

---

## 2. Average Similarity Distance

Measures the average Euclidean distance between the input product and its nearest neighbors.

A lower value indicates greater feature similarity.

For the tested product:

```text
0.0
```

---

## 3. Feature Difference Index (FDI)

Measures the average absolute difference between the input product's feature vector and the feature vectors of its recommendations.

A lower value indicates that the recommended products have similar rating behavior.

For the tested product:

```text
0.0
```

The notebook reports 100% cluster consistency and near-zero feature difference for the tested recommendations.

> **Note:** These evaluation results are based on the tested examples in the notebook and should not be interpreted as a general offline accuracy benchmark across the entire dataset.

---

# 💾 Saved Model Artifacts

The project saves the trained components using `joblib`:

```text
scaler.joblib
kmeans_model.joblib
product_data.joblib
nn_models.joblib
```

| Artifact              | Purpose                                      |
| --------------------- | -------------------------------------------- |
| `scaler.joblib`       | Trained feature scaler                       |
| `kmeans_model.joblib` | Final K-Means clustering model               |
| `product_data.joblib` | Product feature data and cluster assignments |
| `nn_models.joblib`    | Cluster-specific Nearest Neighbors models    |

These artifacts are generated using:

```python
import joblib

joblib.dump(scaler, 'scaler.joblib')
joblib.dump(kmeans, 'kmeans_model.joblib')
joblib.dump(product_feats, 'product_data.joblib')
joblib.dump(nn_models, 'nn_models.joblib')
```

The notebook confirms successful creation of all four artifacts.

---

# 🛠️ Tech Stack

### Programming Language

* Python 3.12+

### Data Processing

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn

Used components include:

* `StandardScaler`
* `KMeans`
* `NearestNeighbors`
* `Birch`
* `AgglomerativeClustering`
* `GaussianMixture`
* `DBSCAN`
* `PCA`
* `silhouette_score`

### Model Persistence

* Joblib

---

# 📁 Suggested Repository Structure

A clean GitHub repository can be organized as:

```text
Product-Recommendation-System/
│
├── README.md
│
├── data/
│   └── rating_short.csv
│
├── notebooks/
│   └── product_recommendation_system.ipynb
│
├── models/
│   ├── scaler.joblib
│   ├── kmeans_model.joblib
│   ├── product_data.joblib
│   └── nn_models.joblib
│
├── requirements.txt
│
└── .gitignore
```

---

# 🚀 Getting Started

## 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/Product-Recommendation-System.git
cd Product-Recommendation-System
```

Replace `<your-username>` with your GitHub username.

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

Create a `requirements.txt` file containing:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
jupyter
```

Then run:

```bash
pip install -r requirements.txt
```

---

## 4. Run the Notebook

Launch Jupyter:

```bash
jupyter notebook
```

Then open:

```text
notebooks/product_recommendation_system.ipynb
```

Make sure the dataset path is correctly configured:

```python
df = pd.read_csv('rating_short.csv')
```

---

# 💡 Example Usage

After training/loading the models:

```python
recommend_similar("1400501466", top_n=5)
```

Returns a list of product IDs that are most similar to the input product based on the engineered product-level rating features.

---

# 🔍 How Recommendations Are Determined

The system does **not** use product descriptions, product categories, images, prices, or textual metadata.

Instead, similarity is based on numerical characteristics derived from user ratings:

```text
Average Rating
       +
Number of Ratings
       +
Rating Variance
       +
Median Rating
       +
Minimum / Maximum Rating
       +
Positive Rating Fraction
       +
Log Popularity
       +
Rating Range
       ↓
Product Feature Vector
       ↓
K-Means Cluster
       ↓
Nearest Neighbors
       ↓
Similar Products
```

Therefore, two products are considered similar when they exhibit similar **user-rating behavior**.

---

# ⚠️ Limitations

The current implementation has several limitations:

### 1. Sparse User Interactions

Most users have very few ratings, which limits the usefulness of user-based collaborative filtering.

### 2. No Product Metadata

The model does not use:

* Product descriptions
* Categories
* Brands
* Prices
* Images
* Product specifications

Consequently, recommendations are based entirely on rating behavior.

### 3. Cold-Start Problem

A completely new product without historical ratings cannot be represented reliably using the current feature-engineering pipeline.

### 4. Rating Bias

The dataset has a strong positive rating bias, with many 5-star ratings.

### 5. Evaluation Scope

The reported recommendation metrics are demonstrated on tested products rather than a comprehensive train/test recommendation benchmark.

### 6. Cluster-Based Recommendation

Products are searched within their assigned cluster. This improves computational efficiency and consistency but may exclude potentially similar products located in another cluster.

---

# 🔮 Future Improvements

Potential improvements include:

* Add product metadata and build a **content-based recommender**.
* Implement traditional **collaborative filtering**.
* Experiment with matrix factorization techniques such as SVD.
* Develop a hybrid recommendation system.
* Address the cold-start problem.
* Introduce time-aware recommendations.
* Add product popularity and recency signals.
* Build a REST API using FastAPI or Flask.
* Create a web interface using Streamlit.
* Add automated offline evaluation using train/test splits.
* Evaluate Precision@K, Recall@K, MAP@K, and NDCG@K.
* Add recommendation explanations.
* Optimize nearest-neighbor search for larger datasets.

---

# 📈 Key Results

The final approach achieved:

| Component                                     | Result     |
| --------------------------------------------- | ---------- |
| User-product interactions                     | **78,245** |
| Unique users                                  | **76,430** |
| Unique products                               | **40,228** |
| Product features                              | **9**      |
| Final K-Means clusters                        | **11**     |
| K-Means silhouette score                      | **0.7765** |
| Recommendation size                           | **Top-5**  |
| Cluster consistency in tested example         | **100%**   |
| Average similarity distance in tested example | **0.0**    |
| Feature Difference Index in tested example    | **0.0**    |

---

# 👨‍💻 Project Workflow

```text
                 ┌─────────────────────┐
                 │ rating_short.csv    │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Data Cleaning & EDA │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Product Aggregation │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ 9 Product Features  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ StandardScaler      │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ K-Means (K = 11)    │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Product Clusters    │
                 └──────────┬──────────┘
                            │
                            ▼
              ┌─────────────────────────────┐
              │ Nearest Neighbors per       │
              │ Cluster                     │
              └──────────────┬──────────────┘
                             │
                             ▼
                 ┌─────────────────────┐
                 │ Top-N Recommendations│
                 └─────────────────────┘
```

---

# 📚 Conclusion

This project demonstrates an **item-to-item recommendation approach** built from product rating behavior.

The system first summarizes user interactions into product-level statistical features, then uses **K-Means clustering** to identify groups of products with similar rating patterns. A **Nearest Neighbors** model is subsequently applied within each cluster to identify the closest products.

The final architecture provides a practical two-stage recommendation strategy:

> **K-Means for product grouping + Nearest Neighbors for product similarity**

The approach is particularly suited to the structure of the available dataset, where user interactions are highly sparse but product-level rating behavior can still provide useful signals for finding similar products.
