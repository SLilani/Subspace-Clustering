# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 20:41:05 2017

@author: MDK
"""
from clusterer import clusterer

# - - Classe clusterer : class racine 
class clusterer_base(clusterer) :
   
    #initialisation
    def __init__(self,base) :
        """ base : base d'exemples
        """
        clusterer.__init__(self,base)
    
    #Affichage    
    def __str__(self) :
        print("########################## Clustering ###############################")
    
    # fonction d'affichage des clusters 
    def plot_clusters(self,dictAffectation) :
        clusterer.plot_clusters(self,dictAffectation)
    
    # Execution
    def run(self) :
        raise NotImplementedError("Please Implement this method")

