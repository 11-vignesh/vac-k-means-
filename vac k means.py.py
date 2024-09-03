#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


dataset = pd.read_csv('Iris.csv')
X = dataset.iloc[:, [3, 4]].values # annual income , spending score
X


# In[3]:


from sklearn.cluster import KMeans
# WCSS - within summ of squares
wcss = [] # intialzing a empty list ,

# We are going to do 10 different iterations
for i in range(1, 11):
    # no of clusters (i) , random intilization - to avoid random intilization trap we use 'K means ++' ,
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X) # we need to fit it to data x
    wcss.append(kmeans.inertia_)# appending the values of 'i' to WCSS for for 1st iteration , i is 1
    #inertia_ will calculate the WCSS
    
plt.plot(range(1, 11), wcss) # no of clusters vs WCSS
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[4]:


# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)

#we need to fit to data x, fit_predict that returns for each observation which clusters it belongs to ,that means each datapoint on our dataset , it going to tell which cluster it belongs to .
y_kmeans = kmeans.fit_predict(X)
y_kmeans


# In[5]:


# Visualising the clusters
# plt.scatter(X[y_kmeans == cluster number,  x_coodinate(index of y , which is 0)], X[y_kmeans == cluster number, y_coodinate(index of y , which is 1)],
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


# In[6]:


# cluster 1 - high income , low spending score # careful segement
# cluster 2 - average income , average spending score # standard segment
# cluster 3 - high income , high spending score # target segment
# cluster 4 - low income , high spending score # careless segment
# cluster 5 - low income , low spending score # sensible segment


# In[7]:


# Visualising the clusters
# plt.scatter(X[y_kmeans == cluster number,  x_coodinate(index of y , which is 0)], X[y_kmeans == cluster number, y_coodinate(index of y , which is 1)],
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Carefull Segment')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Standard Segment')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Target Segment')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Careless Segment')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Sensible Segment')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


# In[ ]:




