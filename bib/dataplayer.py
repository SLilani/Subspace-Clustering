
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import math
import random
import matplotlib.pyplot as plt


# In[2]:

# Fonction de normalisation des données 
# Prend en paramétre un DataFrame Panda

def normalisation(dff) :
    df = dff.copy()
    for i in df.columns :
        mi = min(df[i])
        ma = max(df[i])
        df[i] =  (df[i]- mi) / (ma - mi)
    
    return df


# In[ ]:

# Fonction d'affichage 
# Prend en paramétre centroides : dictionnaire contenant le centroide de chaque cluster
# dictAffectation :dictionnaire contenant  Les indices de tous les points dans la base  pour chque cluster
# base :  DataFrame contenant niotre base d'exemples  

import matplotlib 
import colorsys

def affichagesClusters(centroides,dictAffectation,base):
    ind_points = []
    for cle,val in dictAffectation.items() :
        ind_points += val
    #if(len(ind_points) < len(base)) :
        #base_real = pd.DataFrame()
        #base_real = base_real.append(base.iloc[ind_points[0:]])
        #M_data2D = base_real.as_matrix() 
    #else : 
    M_data2D = base.as_matrix() 
    colonne_X= M_data2D[0:,0] 
    colonne_Y= M_data2D[0:,1] 
    #Nombre de couluers
    colorNbr = len(dictAffectation)
    #Choix aléatoire des couleurs
    colorNames = ['red','blue','darkgreen','deeppink','orangered','orchid',"black"]
    colors = [i for i in range(len(colorNames))]
    colors =colors[0:colorNbr]   
    colorMap = ["black"] 
    for i in range(colorNbr) :
        colorMap.append(colorNames[colors[i]])
    colorMap = np.array(colorMap)

    #categories = np.zeros(len(base))
    #categories = np.array(categories)
    categories = {}
    for key,values in dictAffectation.items() :
        for indice in values :
            categories[indice] = int(key)+ 1
    #categories = categories.astype(int)
    a = np.zeros(len(base))
    for i in categories.keys() :
        a[i]= categories[i]
    a = np.array(a)
    
    plt.scatter(colonne_X,colonne_Y,s=100,c=colorMap[a.astype(int)])
    
    
    if(len(centroides) >0): 
        #Affichage des centroides finaux en noir 
        M_data2D = centroides.as_matrix()
        colonne_X= M_data2D[0:,0]
        colonne_Y= M_data2D[0:,1]
        plt.scatter(colonne_X,colonne_Y,color='black')


# Fonction d'affichage des résultats
def affichageResults(les_centres, l_affectation,nb_iter,convergence) :
    print("Le nombre d'itération : ",nb_iter)
    print("convergence : ", convergence)
    print("Les centres : \n",les_centres)
    print("Les affectations : \n")

    for c,v in l_affectation.items() :
        print("Cluster",c," : ",v)

    AffichagesClusters(les_centres, l_affectation,points)


# In[ ]:

#    ###############################################################################################################     #
#    ######################## Fonctions utiles pour la génération de jeux de donnéés ###############################     # 
#    ######################## selon différentes gaussiennes pour les tests. ########################################     #  
#    ###############################################################################################################     #

    
# Fonction la génération d'une gaussienne 
# Pred en paramétre centre : Centre de la gaussienne, sigma : Variance et nb_points : Nombre de points

def createGaussianDataFrame(center,sigma,nb_points) :
    return pd.DataFrame(np.random.multivariate_normal(center,sigma,nb_points))

# Fonction la génération d'un jeu de donnéés XOR (rond)
# Nombre de points en sortie = 4 * nb_points

def create_xor(nb_points,var) :
    G1 = createGaussianDataFrame(np.array([1,1]),np.array([[var,0],[0,var]]),nb_points)
    G2 = createGaussianDataFrame(np.array([0,0]),np.array([[var,0],[0,var]]),nb_points)
    G3 = createGaussianDataFrame(np.array([0,1]),np.array([[var,0],[0,var]]),nb_points)
    G4 = createGaussianDataFrame(np.array([1,0]),np.array([[var,0],[0,var]]),nb_points)
    return G1.append(G2.append(G3.append(G4,ignore_index=True),ignore_index=True),ignore_index = True)


def create_gauss_ronde(nb_points,var) :
    G1 = createGaussianDataFrame(np.array([1,1]),np.array([[var,0],[0,var]]),nb_points)
    G2 = createGaussianDataFrame(np.array([0,0]),np.array([[var,0],[0,var]]),nb_points)
    return G1.append(G2, ignore_index = True)

# Fonction la génération de deux gaussiennes verticales (elipse)
# Nombre de points en sortie = 2 * nb_points

def create_gauss_vertical(nb_points) :
    positive_points = np.random.multivariate_normal(np.array([0,0]),np.array([[0,1],[0.005,0]]),50)
    negative_points = np.random.multivariate_normal(np.array([1,0]),np.array([[0,1],[0.005,0]]),50)
    points =pd.DataFrame(np.concatenate((positive_points, negative_points), axis=0))
    return points
    
# Fonction la génération de deux gaussiennes horizentales (elipse)
# Nombre de points en sortie = 2 * nb_points

def create_gauss_horizontal(nb_points) :
    positive_points = np.random.multivariate_normal(np.array([0,1]),np.array([[1,0],[0,0.005]]),50)
    negative_points = np.random.multivariate_normal(np.array([0,0]),np.array([[1,0],[0,0.005]]),50)
    points =pd.DataFrame(np.concatenate((positive_points, negative_points), axis=0))
    return points   
    
# Fonction la génération de deux gaussiennes une verticale (elipse) et l'autre ronde
# Nombre de points en sortie = 2 * nb_points

def create_gauss_vertical_cent(nb_points) :
    positive_points = np.random.multivariate_normal(np.array([0,0]),np.array([[0,1],[0.005,0]]),50)
    negative_points = np.random.multivariate_normal(np.array([1,0]),np.array([[0.01,0],[0,0.06]]),50)
    points =pd.DataFrame(np.concatenate((positive_points, negative_points), axis=0))
    return points     
    
# Fonction la génération de deux gaussiennes une horizentale (elipse) et l'autre ronde
# Nombre de points en sortie = 2 * nb_points

def create_gauss_horizontal_cent(nb_points) :
    positive_points =  np.random.multivariate_normal(np.array([0,1]),np.array([[1,0],[0,0.005]]),50)
    negative_points =  np.random.multivariate_normal(np.array([1,0]),np.array([[0.06,0],[0,0.01]]),50)
    points =pd.DataFrame(np.concatenate((positive_points, negative_points), axis=0))
    return points        
    
# Fonction la génération de deux gaussiennes une horizentale (elipse) 
# etl'autre verticale (elispe), qui s'intersectent 
# Nombre de points en sortie = 2 * nb_points

def create_gauss_cross(nb_points) :   
    positive_points = np.random.multivariate_normal(np.array([0,1]),np.array([[0,1],[0.08,0]]),50)
    negative_points = np.random.multivariate_normal(np.array([0,0]),np.array([[1,0],[0,0.08]]),50)
    points =pd.DataFrame(np.concatenate((positive_points, negative_points), axis=0))   
    return points

