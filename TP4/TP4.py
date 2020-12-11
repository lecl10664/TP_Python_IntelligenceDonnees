#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 09:04:35 2020

@author: leopoldclement
"""

import matplotlib as plt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import sklearn as sk
import sklearn.decomposition
from sklearn.model_selection import train_test_split



# Creation de 2 variables différentes pour notre jeux de données
df =  pd.read_csv('iris.csv')

dfIris = pd.read_csv('iris.csv', usecols=range(4))
print(dfIris.head())
dfIrisClass = pd.read_csv('iris.csv', usecols=range(4,5))
print(dfIrisClass.head())


scaler = StandardScaler()

pca = sklearn.decomposition.PCA(n_components = 2)
PCA_val = pca.fit_transform(scaler.fit_transform(dfIris))
df_Iris_PCA = pd.DataFrame(data=PCA_val, columns=['PC1', 'PC2'])
df_Iris_PCA_class = pd.concat([df_Iris_PCA, dfIrisClass], axis = 1)


print(df_Iris_PCA)
print(df_Iris_PCA_class)



species = np.unique(dfIrisClass.values)

colors = ['navy', 'turquoise', 'darkorange']
color_dict = {}
for color, specy in zip(colors, species):
    color_dict[specy] = color
print(color_dict)    


fig, ax = plt.subplots();
for specy in species:
        ax.scatter(PCA_val[df["Class"]==specy, 0], PCA_val[df["Class"]==specy, 1],
               c=color_dict[specy], label=specy);
ax.legend();
ax.set_title('PCA space', fontsize=12);




# 5) K-Means
from sklearn.cluster import KMeans


kmeans = KMeans(n_clusters=3, n_init=5, max_iter=300,random_state=3).fit(dfIris)
kmeans.score(dfIris)
prediction = kmeans.predict(dfIris)

print(prediction)


fig, ax = plt.subplots();
plt.scatter(df_Iris_PCA['PC1'], df_Iris_PCA['PC2'], c = prediction)
plt.title("Cluster avec K-Means, random_state = ")





# Matrice de confusion K-Means


iris = load_iris()
X = iris.data
y = iris.target

print('classes des iris : ')
print( y)
print('prediction des classes avec K-Means : ')
print(prediction)


# Plot normalized confusion matrix

from sklearn.metrics import confusion_matrix


cm = confusion_matrix(y, prediction, labels=[0,1,2], normalize='true')
print (cm)

from seaborn import heatmap 

fig, ax= plt.subplots() 
heatmap(cm, cmap="Blues", annot=True, ax = ax); #annot=True to annotate cells 

# labels, title and ticks 
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix K-Means'); 
ax.xaxis.set_ticklabels(species); ax.yaxis.set_ticklabels(species); 



print('------------------------')

# Cluster avec méthode DBSCAN
from sklearn.cluster import DBSCAN

dbprediction = DBSCAN(eps=0.5, min_samples=5).fit_predict(dfIris)


for i in range(len(dbprediction)):
    if dbprediction[i] == -1:
        dbprediction[i] = 2
    
print(dbprediction)


#Plot de la méthode DBSCAN
fig, ax = plt.subplots();
plt.scatter(df_Iris_PCA['PC1'], df_Iris_PCA['PC2'], c = dbprediction)
plt.title("Cluster avec DBSCAN")


# Matrice de confusion DBSCAN
cm_db = confusion_matrix(y, dbprediction, labels=[0,1,2], normalize='true')
print (cm_db)


fig, ax= plt.subplots() 
heatmap(cm_db, cmap="Blues", annot=True, ax = ax); #annot=True to annotate cells 
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix DBSCAN'); 
ax.xaxis.set_ticklabels(species); ax.yaxis.set_ticklabels(species); 



print('------------------------ Spectral Cluster')
# Compute Spectral Cluster
from sklearn.cluster import SpectralClustering

spprediction = SpectralClustering(n_clusters=3, random_state=1).fit_predict(dfIris)
print(spprediction)

        

#Plot de la méthode Spectral
fig, ax = plt.subplots();
plt.scatter(df_Iris_PCA['PC1'], df_Iris_PCA['PC2'], c = spprediction)
plt.title("Cluster avec Spectral")


#Matrice de confusion
cm_sp = confusion_matrix(y, spprediction, labels=[0,1,2], normalize='true')
print (cm_sp)


fig, ax= plt.subplots() 
heatmap(cm_sp, cmap="Blues", annot=True, ax = ax); #annot=True to annotate cells 
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix Spectral Cluster'); 
ax.xaxis.set_ticklabels(species); ax.yaxis.set_ticklabels(species); 


print('------------------------ HCA')
# Compute HCA
from sklearn.cluster import AgglomerativeClustering

HCAprediction = AgglomerativeClustering(n_clusters=3).fit_predict(df_Iris_PCA)

print(HCAprediction)

#Matrice de confusion HCA
cm_HCA = confusion_matrix(y, HCAprediction, labels=[0,1,2], normalize='true')
print (cm_HCA)


fig, ax= plt.subplots() 
heatmap(cm_HCA, cmap="Blues", annot=True, ax = ax); #annot=True to annotate cells 
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix HCA'); 
ax.xaxis.set_ticklabels(species); ax.yaxis.set_ticklabels(species); 




print('------------------------')
# Compute GMM
from sklearn.mixture import GaussianMixture

GMMprediction = GaussianMixture(n_components=3, covariance_type='tied',random_state=3).fit_predict(df_Iris_PCA)

print(GMMprediction)

#Matrice de confusion GMM
cm_GMM = confusion_matrix(y, GMMprediction, labels=[0,1,2], normalize='true')
print (cm_GMM)


fig, ax= plt.subplots() 
heatmap(cm_GMM, cmap="Blues", annot=True, ax = ax); #annot=True to annotate cells 
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix GMM'); 
ax.xaxis.set_ticklabels(species); ax.yaxis.set_ticklabels(species); 


#Silhouette score du meilleur cluster


from sklearn.metrics import silhouette_score

score = silhouette_score(dfIris, spprediction, metric='sqeuclidean')

print("Silhouette Coefficient: %0.3f"% score)



#Exercice 4


#1

data = pd.read_csv('exo4_atm_extr.csv', sep=';')
print(data)


atm =  pd.read_csv('exo4_atm_extr.csv', sep=';', usecols=range(11))
print(atm)
atmClass = pd.read_csv('exo4_atm_extr.csv', sep=';', usecols=range(11,12))
print(atmClass)

types = np.unique(atmClass.values)
print(types)



#2



#PCA 


pca = sklearn.decomposition.PCA(n_components = 2)
PCA_val = pca.fit_transform(atm)
atm_PCA = pd.DataFrame(data=PCA_val, columns=['PC1', 'PC2'])


print(atm_PCA)




colors = ['navy', 'turquoise', 'darkorange', 'blue', 'red']
color_dict = {}
for color, specy in zip(colors, types):
    color_dict[specy] = color
print(color_dict)    


fig, ax = plt.subplots();
for specy in types:
        ax.scatter(PCA_val[data["Type"]==specy, 0], PCA_val[data["Type"]==specy, 1],
               c=color_dict[specy], label=specy);
ax.legend();
ax.set_title('PCA space', fontsize=12);





#K_MEANS

from sklearn import metrics

clusterer = KMeans(n_clusters=4, random_state=10)
cluster_labels = clusterer.fit_predict(atm)
print(cluster_labels)



fig, ax = plt.subplots();
plt.scatter(atm_PCA['PC1'], atm_PCA['PC2'], c = cluster_labels)
plt.title("Cluster avec K-Means avec 4 clusters")






range_n_clusters = [2, 3, 4, 5, 6, 7]

for n_clusters in range_n_clusters:

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(atm)



    # Calinski-Harabasz Index
    c_h = metrics.calinski_harabasz_score(atm, cluster_labels)
    d_b = metrics.davies_bouldin_score(atm, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The Calinski-Harabasz score is :", c_h)

    print("and the Davies-Bouldin score is :", d_b)


    fig, ax = plt.subplots()
    plt.scatter(atm_PCA['PC1'], atm_PCA['PC2'], c = cluster_labels)
    plt.title("Cluster avec K-Means")



#Exercice 3


data = pd.read_csv('wdbc.data', sep=',')
print(data)



wdbc =  pd.read_csv('wdbc.data', sep=',', usecols=range(2,32))
print(wdbc)



wdbcClass = pd.read_csv('wdbc.data', sep=',', usecols=range(1,2))
print(wdbcClass)

types = np.unique(wdbcClass.values)
print(types)



#PCA 


pca = sklearn.decomposition.PCA(n_components = 2)
PCA_val = pca.fit_transform(scaler.fit_transform(wdbc))
wdbc_PCA = pd.DataFrame(data=PCA_val, columns=['PC1', 'PC2'])


print(wdbc_PCA)


colors = ['blue', 'red']
color_dict = {}
for color, specy in zip(colors, types):
    color_dict[specy] = color
print(color_dict)    


fig, ax = plt.subplots();
for specy in types:
        ax.scatter(PCA_val[data.iloc[:, 1]==specy, 0], PCA_val[data.iloc[:, 1]==specy, 1],
               c=color_dict[specy], label=specy);
ax.legend();
ax.set_title('PCA space', fontsize=12);





# K-Means
from sklearn.cluster import KMeans


kmeans = KMeans(n_clusters=2, random_state=0).fit(wdbc)
kmeans.score(wdbc)
prediction = kmeans.predict(wdbc)

print(prediction)


fig, ax = plt.subplots();
plt.scatter(wdbc_PCA['PC1'], wdbc_PCA['PC2'], c = prediction)
plt.title("Cluster avec K-Means")


print('------------------------')



# Cluster avec méthode DBSCAN
from sklearn.cluster import DBSCAN

dbprediction = DBSCAN().fit_predict(wdbc)

print(dbprediction)

fig, ax = plt.subplots();
plt.scatter(wdbc_PCA['PC1'], wdbc_PCA['PC2'], c = dbprediction)
plt.title("Cluster avec DBSCAN")


print('------------------------')

# Spectral Cluster
from sklearn.cluster import SpectralClustering

spprediction = SpectralClustering(n_clusters=2, random_state=0).fit_predict(wdbc)
print(spprediction)


fig, ax = plt.subplots();
plt.scatter(wdbc_PCA['PC1'], wdbc_PCA['PC2'], c = spprediction)
plt.title("Cluster avec Spectral")


print('------------------------')


# Compute HCA
from sklearn.cluster import AgglomerativeClustering

HCAprediction = AgglomerativeClustering(n_clusters=2).fit_predict(wdbc)

print(HCAprediction)

fig, ax = plt.subplots();
plt.scatter(wdbc_PCA['PC1'], wdbc_PCA['PC2'], c = HCAprediction)
plt.title("Cluster avec HCA")




print('------------------------')
# Compute GMM
from sklearn.mixture import GaussianMixture

GMMprediction = GaussianMixture(n_components=2, covariance_type='tied',random_state=3).fit_predict(wdbc)

print(GMMprediction)

fig, ax = plt.subplots();
plt.scatter(wdbc_PCA['PC1'], wdbc_PCA['PC2'], c = GMMprediction)
plt.title("Cluster avec GMM")




