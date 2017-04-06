# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 20:02:49 2017

@author: MDK
"""
from bib.Evaluateur import Evaluateur
import bib.outils as tools
import pandas as pd


class Evaluateur_inertie(Evaluateur) :
    def __init__(self,base,dict_affectations,centroides) :
        """ base : DataFrame d'exemples
        """
        Evaluateur.__init__(self,base)
        self.dict_affectations = dict_affectations
        self.centroides = centroides
       
    # Inertie d'un cluster 
    def inertie_cluster(self , df,centroide ,fonc ) :
        inertie = 0
        for i,ligne in df.iterrows() :
            inertie = inertie + pow(tools.distance(fonc,pd.DataFrame(ligne).T,centroide),2)
        return inertie
    
    # Inertie intra-classes 
    def inertie_globale(self,fonc = tools.euclidienne) :
        inertie_glob = 0
        i = 0 
        for key,values in self.dict_affectations.items():
            inertie_glob += self.inertie_cluster(pd.DataFrame(self.base.iloc[self.dict_affectations[key]]),self.centroides.ix[i],fonc)
            i += 1
        return inertie_glob
    
    # Inertie inter-classes
    def inertie_inter_clusters(self,fonc = tools.euclidienne) :
        centroide = tools.centroide(self.centroides)
        return self.inertie_cluster(self.centroides,centroide,fonc)
            
    # Inertie Totale = Inertie inter-classes + # Inertie intra-classes 
    def inertie_totale(self,fonc = tools.euclidienne) :
        return self.inertie_globale(fonc) + self.inertie_inter_clusters(fonc)
   
    #@classmethod
    def Evaluate(self,fonc)  :
        results = {}
        results["IW"] = self.inertie_globale(fonc) # Inertie intra-classes
        results["IR"] = self.inertie_inter_clusters(fonc)  # Inertie inter-classe
        results["IT"] = self.inertie_totale(fonc) # Inertie totale
        Evaluateur.Evaluate(self,results)
    
    #@classmethod
    def store_results(self,filename,libelle,i) :
        with open(filename, 'a') as fichier:
            fichier.write(libelle+" number "+str(i)+"\n")
            fichier.write("Inertie intra-classes : "+str(self.results["IW"]))+"\n"
            fichier.write("Inertie inter-classes : "+str(self.results["IR"]))+"\n"
            fichier.write("Inertie totale-classes : "+str(self.results["IT"]))+"\n"
    #@classmethod
    def __str__(self) : 
            print("Résultat de l'évaluation : \n")
            print("Inertie intra-classes : "+str(self.results["IW"])+"\n")
            print("Inertie inter-classes : "+str(self.results["IR"])+"\n")
            print("Inertie totale-classes : "+str(self.results["IT"])+"\n")
        
            
        