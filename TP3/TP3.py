import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import seaborn as sns; sns.set()
import numpy as np
from sklearn import decomposition
from sklearn.datasets import load_digits
from sklearn.manifold import MDS
from sklearn import decomposition
from sklearn import preprocessing
from sklearn.manifold import Isomap
from sklearn.manifold import MDS
from matplotlib.collections import LineCollection
import seaborn as sns; sns.set()
from sklearn.manifold import LocallyLinearEmbedding

#Exercice A - cf word

#Exercice B

#1. Importation du ficher csv 

ir = pd.read_csv("iris.csv", sep = ',')
df = pd.read_csv("iris.csv", usecols=[0, 1, 2, 3])
print (ir)
print (df)


#2. Séparation des données avec zip

#cf le 4

#3. Normalisation des données

scaler = StandardScaler()
data = scaler.fit_transform(df)

#4. ACP 

def display_circles(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):
    for d1, d2 in axis_ranks: # On affiche les 3 premiers plans factoriels, donc les 6 premières composantes
        if d2 < n_comp:

            # initialisation de la figure
            fig, ax = plt.subplots(figsize=(7,6))

            # détermination des limites du graphique
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else :
                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])

            # affichage des flèches
            # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
            if pcs.shape[1] < 30 :
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                   pcs[d1,:], pcs[d2,:], 
                   angles='xy', scale_units='xy', scale=1, color="grey")
                # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))
            
            # affichage des noms des variables  
            if labels is not None:  
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        plt.text(x, y, labels[i], fontsize='14', ha='center', va='center', rotation=label_rotation, color="blue", alpha=0.5)
            
            # affichage du cercle
            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)

            # définition des limites du graphique
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
        
            # affichage des lignes horizontales et verticales
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Cercle des corrélations (F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)
        
def display_factorial_planes(X_projected, n_comp, pca, axis_ranks, labels=None, alpha=1, illustrative_var=None):
    for d1,d2 in axis_ranks:
        if d2 < n_comp:
        
            # affichage des points
            if illustrative_var is None:
                plt.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha)
            else:
                illustrative_var = np.array(illustrative_var)
                for value in np.unique(illustrative_var):
                    selected = np.where(illustrative_var == value)
                    plt.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value)
                plt.legend()

            # affichage des labels des points
            if labels is not None:
                for i,(x,y) in enumerate(X_projected[:,[d1,d2]]):
                    plt.text(x, y, labels[i],
                              fontsize='14', ha='center',va='center') 
                
            # détermination des limites du graphique
            boundary = np.max(np.abs(X_projected[:, [d1,d2]])) * 1.1
            plt.xlim([-boundary,boundary])
            plt.ylim([-boundary,boundary])
        
            # affichage des lignes horizontales et verticales
            plt.plot([-100, 100], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-100, 100], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Projection des individus (sur F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)

def display_scree_plot(pca):
    scree = pca.explained_variance_ratio_*100
    plt.bar(np.arange(len(scree))+1, scree)
    plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')
    plt.xlabel("rang de l'axe d'inertie")
    plt.ylabel("pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres")
    plt.show(block=False)



# Projection des individus - PCA avec couleurs

n_components = 2
features = df.columns
names = df.index

df2 = pd.read_csv('iris.csv', names = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "Class"])

pca = decomposition.PCA(n_components)
dfprojected = pca.fit_transform(data)
x = pd.DataFrame(data = dfprojected, columns = ['F1', 'F2'])
y = pd.concat([x, df2['Class']], axis = 1)
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
targets = ['setosa', 'versicolor', 'virginica']
colors = ['orange','yellow', 'pink']
for target, color in zip(targets, colors):
  Val = y['Class'] == target 
  ax.scatter(y.loc[Val, 'F1'], y.loc[Val, 'F2'], c=color, s = 30)
ax.legend(targets)
ax.grid()

# Cercle des corrélations

 
n_components = 3
features = df.columns
names = df.index

pca = decomposition.PCA(n_components)
dfprojected = pca.fit_transform(df)
pd.DataFrame(data=dfprojected, index=df.index, columns=["F" + str(i+1) for i in range (3)])

X_projected = pca.transform(df)
display_factorial_planes(dfprojected, n_components, pca, [(0,1),(2,3)], labels = np.array(names))

plt.show()

# Projection des individus - PCA sans couleurs

pcs = pca.components_
display_circles(pcs, n_components, pca, [(0,1),(2,3)], labels = np.array(features))

plt.show()










#Exercice C

#1. Importation du ficher csv 

gl = pd.read_csv("golub_data.csv", sep = ',') 

#2. Importation des labels 

gl2 = pd.read_csv("golub_class2.csv", usecols=[1])
print(gl2)

# Projection des individus - Globul



n_components = 2
names = gl.index



# Centrage et Réduction
std_scale = preprocessing.StandardScaler().fit(gl)
X_scaled = std_scale.transform(gl)

# Calcul des composantes principales
pca = decomposition.PCA(n_components)
pca.fit(X_scaled)


# Projection des individus

X_projected = pca.transform(X_scaled)
display_factorial_planes(X_projected, n_components, pca, [(0,1),(2,3),(4,5)], labels = np.array(names))

plt.show()


# Projection des individus - Globul avec label

gl2 = pd.read_csv("golub_class2.csv", usecols=[1])
gl3 = class2 = pd.read_csv('golub_class2.csv', names = ["Ex", "Dt"])

n_components = 2


pca = decomposition.PCA(n_components) 
x = pd.DataFrame(data = X_projected, columns = ['F1', 'F2'])
y = pd.concat([x, gl3[['Dt']]], axis = 1)
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
targets = ['ALL', 'AML']
colors = ['blue','green']
for target, color in zip(targets, colors):
  Val = y['Dt'] == target 
  ax.scatter(y.loc[Val, 'F1'], y.loc[Val, 'F2'], c=color, s = 30)
ax.legend(targets)
ax.grid()







# EXERCICE D



#1. Importation du ficher csv 

alon = pd.read_csv("alon.csv", sep = ';')
print(alon)

#2. Importation des labels 

alon_class = pd.read_csv("alon_class.csv")
print(alon_class)

# Projection des individus - Globul



n_components = 2


# Centrage et Réduction
std_scale = preprocessing.StandardScaler().fit(alon)
X_scaled = std_scale.transform(alon)

# Calcul des composantes principales
pca = decomposition.PCA(n_components)
pca.fit(X_scaled)


# Projection des individus

X_projected = pca.transform(X_scaled)
display_factorial_planes(X_projected, n_components, pca, [(0,1),(2,3),(4,5)])

plt.show()


# Projection des individus - Globul avec label


alon_class2 = pd.read_csv('alon_class.csv', names = ["Col"])

n_components = 2


pca = decomposition.PCA(n_components) 
x = pd.DataFrame(data = X_projected, columns = ['F1', 'F2'])
y = pd.concat([x, alon_class2[['Col']]], axis = 1)
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
targets = ['t', 'n']
colors = ['blue','green']
for target, color in zip(targets, colors):
  Val = y['Col'] == target 
  ax.scatter(y.loc[Val, 'F1'], y.loc[Val, 'F2'], c=color, s = 30)
ax.legend(targets)
ax.grid()



# MDS - Globul avec label

n_components = 3

embedding = MDS(n_components)
glprojected = embedding.fit_transform(X_scaled)
x = pd.DataFrame(data = glprojected, columns = ['F1', 'F2','F3'])
y = pd.concat([x, alon_class2[['Col']]], axis = 1)
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
targets = ['t', 'n']
colors = ['blue','green']
for target, color in zip(targets, colors):
    Val = y['Col'] == target
    ax.scatter(y.loc[Val, 'F1'], y.loc[Val, 'F2'], c=color, s = 30)
ax.legend(targets)
ax.grid()

plt.show()


# utilisation de LocallyLinearEmbedding

n_components = 3

embedding = LocallyLinearEmbedding(n_components)
glprojected = embedding.fit_transform(X_scaled)
x = pd.DataFrame(data = glprojected, columns = ['F1', 'F2'])
y = pd.concat([x, alon_class2[['Col']]], axis = 1)
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
targets = ['t', 'n']
colors = ['blue','green']
for target, color in zip(targets, colors):
    Val = y['Col'] == target
    ax.scatter(y.loc[Val, 'F1'], y.loc[Val, 'F2'], c=color, s = 30)
ax.legend(targets)
ax.grid()

plt.show()


# utilisation de ISOMAP
n_components = 30

embedding = Isomap(n_components=2)
glprojected = embedding.fit_transform(X_scaled)
x = pd.DataFrame(data = glprojected, columns = ['F1', 'F2'])
y = pd.concat([x, alon_class2[['Col']]], axis = 1)
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
targets = ['t', 'n']
colors = ['blue','green']
for target, color in zip(targets, colors):
    Val = y['Col'] == target
    ax.scatter(y.loc[Val, 'F1'], y.loc[Val, 'F2'], c=color, s = 30)
ax.legend(targets)
ax.grid()

plt.show()



