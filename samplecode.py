Sample code:
# ================================
# 1. Import Libraries
# ================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

 ================================
# 2. Load Dataset
 ================================
 Example CSV format:
 Food,Calories,Protein,Fat,Carbs,VitaminC,Calcium,Iron
df = pd.read_csv("/Users/samyuktha/Desktop/FoodNutrientAnalysis/Book.xlsx")

print("First 5 rows of dataset:")
print(df.head())

#================================
# 3. Data Preprocessing
 ================================
# Select only numerical nutrient columns
nutrient_cols = ['Calories', 'Protein', 'Fat', 'Carbs', 'VitaminC', 'Calcium', 'Iron']
X = df[nutrient_cols]

# Handle missing values (if any)
X = X.fillna(X.mean())

# Standardize (important for KMeans)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

 ================================
# 4. Elbow Method to Find Optimal K
#================================
inertia = []
K = range(1, 11)

for k in K:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.figure(figsize=(6,4))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()

 ================================
# 5. Apply KMeans Clustering
 ================================
optimal_k = 4   # you can change based on elbow curve
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

print("\nClustered Data Sample:")
print(df.head())

================================
# 6. Visualization with PCA (2D)
 ================================
pca = PCA(2)  # reduce dimensions to 2 for visualization
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=df['Cluster'], cmap='viridis', s=50)
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title(f"KMeans Clustering of Foods (k={optimal_k})")
plt.colorbar(label="Cluster")
plt.show()

 ================================
# 7. Cluster Analysis
 ================================
cluster_summary = df.groupby('Cluster')[nutrient_cols].mean()
print("\nCluster Nutrient Averages:")
print(cluster_summary)
