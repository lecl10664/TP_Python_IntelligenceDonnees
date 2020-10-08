#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 09:14:16 2020

@author: lecl10664
"""

import matplotlib as plt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from math import sqrt, pi, exp

import csv

df = pd.read_csv('tabBats.csv')
print(df.head())
print('----------------')
print(df.shape)
print('')


print("Moyenne BOW =", np.mean(df.BOW))
print("Variance BOW =", np.var(df.BOW))
print("Ecart-type BOW =", np.std(df.BOW))
print("Poids median cerveau =", np.median(df.BRW), "ug")

print('')

plt.title('Histogramme des volumes du noyau auditif')
plt.hist(df.AUD, facecolor='g',alpha = 0.75, ec='black')
plt.grid('true')
plt.show()
plt.title('Histogramme des volumes du bulbe olfactif')
plt.hist(df.MOB, facecolor='b',alpha = 0.75, ec='black')
plt.grid('true')   
plt.show()
plt.title("Histogramme des volumes de l'hippocampe")
plt.hist(df.HIP, facecolor='r',alpha = 0.75, ec='black')
plt.grid('true')
plt.show()



# Exercice 2 : 

df1 = pd.read_csv('notes.csv', sep=';')
print(df1.head())
print(df1.shape)

plt.title('Histogramme des notes du TP1')
plt.hist(df1.Lab1, 59, facecolor='g',alpha = 0.75, ec='black')
plt.grid('true')
plt.show()

plt.title('Histogramme des notes du TP3')
plt.hist(df1.Lab3, 59, facecolor='b',alpha = 0.75, ec='black')
plt.grid('true')
plt.show()

plt.title('Histogramme des notes du TP7')
plt.hist(df1.Lab7, 59, facecolor='r',alpha = 0.75, ec='black')
plt.grid('true')
plt.show()

print("Note moyenne des projets =", np.mean(df1.Project))
print("Note minimal des projets =", np.min(df1.Project)) 
print("Note maximal des projets =", np.max(df1.Project)) 
print("Ecart-typpe des projets =", np.std(df1.Project)) 

nA = 0
nB = 0
nC = 0
nD = 0
nE = 0
nF = 0
for i in df1.GPA :
    if (i == "A") :
        nA +=  1
    if (i == "B") :
        nB += 1
    if (i == "C") :
        nC += 1
    if (i == "D") :
        nD += 1
    if (i == "E") :
        nE += 1
    if (i == "F") :
        nF += 1
        
        
x = [nA,nB,nC,nD,nE, nF]
plt.pie(x, labels = ['A', 'B', 'C', 'D', 'E', 'F'], autopct='%1.1f%%')
plt.title("Diagramme du nombre d'eleve par groupe")














