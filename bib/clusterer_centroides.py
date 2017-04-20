# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 20:51:56 2017

@author: MDK
"""
import matplotlib.pyplot as plt

# - - Classe clusterer_centroides
class clusterer_centroides :
    
    #Affichage  details des centroides  
    def __str__(self) :
        print("########################## Centroides ###############################")
    
    # Affichage des centroides 
    def plot_centroides(self,centroides) :
        # Affichage des centroides finaux en noir 
        M_data2D = centroides.as_matrix()
        colonne_X= M_data2D[0:,0]
        colonne_Y= M_data2D[0:,1]
        plt.scatter(colonne_X,colonne_Y,color='black')