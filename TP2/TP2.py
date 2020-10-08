# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set()

df = pd.read_csv('iris.csv', sep=',')
print(df.head())


sns.distplot(df.sepal_length, color='r')
plt.show()
sns.distplot(df.sepal_width, color='g')
plt.show()
sns.distplot(df.petal_length, color='b')
plt.show()
sns.distplot(df.petal_width, color='y')
plt.show()




#3. Commande corr() 
print(df.corr())

#4. Commande pairplot et heatmap
sns.pairplot(df)
sns.heatmap(df.corr())

#5.Intervale de confiance

def IC95(data) :
    r = data.corr()
    Z = (np.log(1+r) - np.log(1-r))/2
    sZ = np.sqrt(1/(len(r)-3))
    Zinf = Z - 1.96*sZ
    Zsup = Z + 1.96*sZ


    IC  = [(np.exp(2*Zinf) -1)/(np.exp(2*Zinf)+1),
      (np.exp(2*Zsup) -1)/(np.exp(2*Zsup)+1) ]

    return IC

IC95(df)


# EXERCICE B

#1
df2 = pd.read_csv('mansize.csv', sep=';')
print(df2.head())




#2
 

#3


plt.title('Histogramme de la répartition des ages')
plt.hist(df2.A, facecolor='g',alpha = 0.75, ec='black')
plt.grid('true')
plt.show()
plt.title('Histogramme des tailles')
plt.hist(df2.B, facecolor='b',alpha = 0.75, ec='black')
plt.grid('true')   
plt.show()
plt.title('Histogramme des poids')
plt.hist(df2.C, facecolor='r',alpha = 0.75, ec='black')
plt.grid('true')   
plt.show()
plt.title('Histogramme des longueurs de femurs')
plt.hist(df2.D, facecolor='b',alpha = 0.75, ec='black')
plt.grid('true')   
plt.show()
plt.title('Histogramme des longueurs de penis')
plt.hist(df2.I, facecolor='y',alpha = 0.75, ec='black')
plt.grid('true')   
plt.show()


#4
print(df2.corr())


sns.pairplot(df2)
sns.heatmap(df2.corr())


#5

IC95(df2)






#Exercice C

#1. Importation du ficher csv 
wt = pd.read_csv("weather.csv", sep = ';')



#2.Commande Crosstab
table = pd.crosstab(wt['Outlook'], wt['Temperature'])




