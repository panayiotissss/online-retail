from features import build_features
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans , AgglomerativeClustering , DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score ,davies_bouldin_score , adjusted_rand_score , calinski_harabasz_score
import joblib
import hdbscan


#Import data
df = build_features(path='data/online_retail_II.csv')


scaler = StandardScaler()

X_scaled = pd.DataFrame(
    scaler.fit_transform(df),
    columns=df.columns,
    index=df.index
)





# # 2D — for visualisation only
reducer_2d = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=42)
X_2d = reducer_2d.fit_transform(X_scaled)

joblib.dump(scaler,'models/scaler.pkl')
joblib.dump(reducer_2d,'models/umap_reducer.pkl')


#Plot it immediately so you can see the structure

plt.scatter(X_2d[:, 0], X_2d[:, 1], s=5, alpha=0.5)
plt.title('UMAP 2D projection')
plt.savefig('plots/umap_2d.png')
#plt.show()

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
plt.savefig('plots/Inertia Elbow.png')
#plt.show()
plt.close()




for k in [3, 4, 5]: #k =4
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    caliniski =  calinski_harabasz_score(X_scaled, labels)
    print(f"k={k}, silhouette={score:.4f}")
    print(f"k={k}, calinski={caliniski:.4f}")


km = KMeans(n_clusters =4 , init ='k-means++', n_init = 10 , random_state=42)
labels_run1 = km.fit_predict(X_scaled)
print(pd.Series(labels_run1).value_counts().sort_index())


#Silhouette
score = silhouette_score(X_scaled, labels_run1)
print(f"k={4} K-Means, silhouette={score:.4f}")

#David Boulding
score = davies_bouldin_score(X_scaled , labels_run1)
print(f"k={4} K-Means, David={score:.4f}")

df['Cluster'] = labels_run1
profile = df.groupby('Cluster').mean()
print(profile.round(2))



#Stability
km = KMeans(n_clusters =4 , init ='k-means++', n_init = 10 , random_state=15)
labels_run2 = km.fit_predict(X_scaled)
print(pd.Series(labels_run2).value_counts().sort_index())

# Compare ARI
ARI = adjusted_rand_score(labels_run1,labels_run2)
print(f'K-Means ARI ={ARI}')



cluster_names = {
    0: 'Champions',
    1: 'At Risk',
    2: 'Lost',
    3: 'Loyal but Slow'
}
# df['Segment'] = df['Cluster'].map(cluster_names)


#Plot to see colours per cluster

plt.figure()
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels_run1, cmap='tab10', s=5, alpha=0.5)
plt.title('UMAP 2D projection')
plt.savefig('plots/umap_2d_kmeans.png')
#plt.show()
plt.close()

#Save the model
joblib.dump(reducer_2d,'models/umap_reducer.pkl')



#Agglomerative  k=3

# for k in [3, 4, 5]:
#     Ag = AgglomerativeClustering(n_clusters=k, linkage = 'ward')
#     labels = Ag.fit_predict(X_scaled)
#     score = silhouette_score(X_scaled, labels)
#     print(f"k={k}, silhouette={score:.4f}")


Ag = AgglomerativeClustering(n_clusters=3, linkage = 'ward')
labels = Ag.fit_predict(X_scaled)
print(pd.Series(labels).value_counts().sort_index())


#Silhouette
score = silhouette_score(X_scaled, labels)
print(f"k={3} Agglomerative, silhouette={score:.4f}")

#David Boulding
score = davies_bouldin_score(X_scaled , labels)
print(f"k={3} Agglomerative, David={score:.4f}")

df['Cluster'] = labels
profile = df.groupby('Cluster').mean()
print(profile.round(2))


# 2D to see the colours per cluster

plt.figure()
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='tab10', s=5, alpha=0.5)
plt.title('UMAP 2D -Agglomerative')
plt.savefig('plots/umap_2d_agglo.png')
#plt.show()
plt.close()



cluster_names = {
    0: 'Inactive/Lost',
    1: 'Champions',
    2: 'At Risk'
}
# df['Segment'] = df['Cluster'].map(cluster_names)

#Save the model
joblib.dump(Ag,'models/Agglomerative.pkl')



#DBSCAN  k = 1 -> Failed due to global eps and varying cluster sizes (also convex clusters)
# n_neibors = 2*n_features 
# n_neigbors = min_samples

#Calculate eps (where is shoots up that is the eps)
nn = NearestNeighbors(n_neighbors = 20)
nn.fit(X_scaled)

distances, _ = nn.kneighbors(X_scaled)
k_distances = np.sort(distances[:, 4])
plt.figure()
plt.plot(k_distances)
plt.title('K-Distance Graph')
plt.xlabel('Points sorted by distance')
plt.ylabel('5th nearest neighbour distance')
plt.savefig('plots/kdistance.png')
#plt.show()
plt.close()

#Fit dbscan
db = DBSCAN(eps=2, min_samples=20)
labels_db = db.fit_predict(X_scaled)
print(pd.Series(labels_db).value_counts().sort_index())


#HDBSCAN
#min_cluster is 1% of data

# Essentially same result as 2D so 2d captures it correctly - 5 just to check
# reducer_5d = umap.UMAP(n_components=5, n_neighbors=30, min_dist=0.0, random_state=42)
# X_5d = reducer_5d.fit_transform(X_scaled)

hdb = hdbscan.HDBSCAN(min_cluster_size=50)
labels_hdb = hdb.fit_predict(X_2d)
print(pd.Series(labels_hdb).value_counts().sort_index())

mask = labels_hdb != -1
#Silhouette
score = silhouette_score(X_scaled[mask], labels_hdb[mask])
print(f"k={2}, silhouette={score:.4f}")

#David Boulding
score = davies_bouldin_score(X_scaled[mask] , labels_hdb[mask])
print(f"k={2}, David={score:.4f}")

df['Cluster'] = labels_hdb
profile = df.groupby('Cluster').mean()
print(profile.round(2))


# 2D to see the colours per cluster

plt.figure()
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels_hdb, cmap='tab10', s=5, alpha=0.5)
plt.title('UMAP 2D -HDBSCAN')
plt.savefig('plots/umap_2d_hdbscan.png')
#plt.show()
plt.close()



cluster_names = {
    0: 'Champions',
    1: 'Inactive/Lost',
}

# df['Segment'] = df['Cluster'].map(cluster_names)

# #Save the model
joblib.dump(hdb,'models/Hdbscan.pkl')



#Deploy k-means more business utility similar metrics to hdbscan

# ┌───────────────┬─────┬────────────┬────────────────┬─────────────────────────────────────────────┐
# │   Algorithm   │  k  │ Silhouette │ Davies-Bouldin │                    Notes                    │
# ├───────────────┼─────┼────────────┼────────────────┼─────────────────────────────────────────────┤
# │ K-Means       │ 4   │ 0.2550     │ 1.2789         │ Stable ARI=0.998, 4 interpretable segments  │ -> Winner
# ├───────────────┼─────┼────────────┼────────────────┼─────────────────────────────────────────────┤
# │ Agglomerative │ 3   │ 0.2296     │ 1.3448         │ Deterministic, 3 segments                   │
# ├───────────────┼─────┼────────────┼────────────────┼─────────────────────────────────────────────┤
# │ DBSCAN        │ -   │ failed     │ failed         │ One giant cluster, wrong tool for blob data │
# ├───────────────┼─────┼────────────┼────────────────┼─────────────────────────────────────────────┤
# │ HDBSCAN       │ 2   │ 0.2849     │ 1.2661         │ Best metrics, found noise, 2 broad segments │
# └───────────────┴─────┴────────────┴────────────────┴─────────────────────────────────────────────┘