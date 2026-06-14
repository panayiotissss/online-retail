from features import build_features
from sklearn.preprocessing import StandardScaler
import pandas as pd
import umap
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score ,davies_bouldin_score , adjusted_rand_score
import joblib


#Import data
df = build_features(path='data/online_retail_II.csv')


print(df.head())

scaler = StandardScaler()

X_scaled = pd.DataFrame(
    scaler.fit_transform(df),
    columns=df.columns,
    index=df.index
)


print(X_scaled.head())


# # 2D — for visualisation only
# reducer_2d = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=42)
# X_2d = reducer_2d.fit_transform(X_scaled)

# Plot it immediately so you can see the structure

# plt.scatter(X_2d[:, 0], X_2d[:, 1], s=5, alpha=0.5)
# plt.title('UMAP 2D projection')
# plt.savefig('umap_2d.png')
# plt.show()

#K-Means
inertias = []

for i in range(2,10):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    print(inertias)

plt.plot(range(2,10), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.savefig('Inertia Elbow.png')
#plt.show()
plt.close()




# for k in [3, 4, 5]:
#     km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
#     labels = km.fit_predict(X_scaled)
#     score = silhouette_score(X_scaled, labels)
#     print(f"k={k}, silhouette={score:.4f}")



km = KMeans(n_clusters =4 , init ='k-means++', n_init = 10 , random_state=42)
labels_run1 = km.fit_predict(X_scaled)
print(pd.Series(labels_run1).value_counts().sort_index())


#Silhouette
score = silhouette_score(X_scaled, labels_run1)
print(f"k={4}, silhouette={score:.4f}")

#David Boulding
score = davies_bouldin_score(X_scaled , labels_run1)
print(f"k={4}, David={score:.4f}")

df['Cluster'] = labels_run1
profile = df.groupby('Cluster').mean()
print(profile.round(2))



#Stability
km = KMeans(n_clusters =4 , init ='k-means++', n_init = 10 , random_state=15)
labels_run2 = km.fit_predict(X_scaled)
print(pd.Series(labels_run2).value_counts().sort_index())

# Compare ARI
ARI = adjusted_rand_score(labels_run1,labels_run2)
print(f'ARI ={ARI}')



cluster_names = {
    0: 'Champions',
    1: 'At Risk',
    2: 'Lost',
    3: 'Loyal but Slow'
}
df['Segment'] = df['Cluster'].map(cluster_names)


# 2D to see the colours per cluster
reducer_2d = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=42)
X_2d = reducer_2d.fit_transform(X_scaled)

plt.figure()
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels_run1, cmap='tab10', s=5, alpha=0.5)
plt.title('UMAP 2D projection')
plt.savefig('umap_2d.png')
plt.show()
plt.close()

#Save the model
joblib.dump(km,'models/kmeans.pkl')
joblib.dump(scaler,'models/scaler.pkl')
joblib.dump(reducer_2d,'models/umap_reducer.pkl')



#Agglomerative