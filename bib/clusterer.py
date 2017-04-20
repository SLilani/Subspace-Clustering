# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 20:26:14 2017

@author: MDK
"""

import numpy as np
import matplotlib.pyplot as plt

# - - Classe clusterer : class racine 
class clusterer :
   
    #initialisation
    def __init__(self,base) :
        """ base : base d'exemples
        """
        self.base = base
    #Affichage    
    def __str__(self) :
        print("########################## Affichage ###############################")
    
    
    # fonction d'affichage des clusters 
    def plot_clusters(self,dictAffectation):
        ind_points = []
        for cle,val in dictAffectation.items() :
            ind_points += val 
        M_data2D = self.base.as_matrix() 
        colonne_X= M_data2D[0:,0] 
        colonne_Y= M_data2D[0:,1] 
        #Nombre de couluers
        colorNbr = len(dictAffectation)
        #Choix aléatoire des couleurs
        colorNames = ['red','blue','darkgreen','deeppink','orangered','orchid','orange','green',"black"]
        colors = [i for i in range(len(colorNames))]
        colors =colors[0:colorNbr]   
        colorMap = ["black"] 
        
        for i in range(colorNbr) :
            colorMap.append(colorNames[colors[i]])
        colorMap = np.array(colorMap)
        categories = {}
        
        for key,values in dictAffectation.items() :
            for indice in values :
                categories[indice] = int(key)+ 1
    
        a = np.zeros(len(self.base))
        for i in categories.keys() :
            a[i]= categories[i]
        a = np.array(a)
        
        plt.scatter(colonne_X,colonne_Y,s=100,c=colorMap[a.astype(int)])
    
    # Execution
    def run(self) :
        raise NotImplementedError("Please Implement this method")
    
