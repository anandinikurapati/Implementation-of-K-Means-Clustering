# Implementation-of-K-Means-Clustering
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs 
data=make_blobs(n_samples=200,n_features=2,centers=4,
cluster_std=1.8, random_state=101)
kmeans = KMeans(n_clusters=4) 
kmeans.fit(data[0])
print("K-Means Cluster Centers") 
print(kmeans.cluster_centers_)
print("K-Meams Lables") 
print(kmeans.labels_)
