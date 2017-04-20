# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 08:41:00 2017

@author: MDK
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 20:02:49 2017

@author: MDK
"""
from Evaluateur import Evaluateur
import outils as tools
import pandas as pd


class Evaluateur_centroide(Evaluateur) :
    def __init__(self,centroides_attendus,centroides_obtenus) :
        """ base : DataFrame d'exemples
        """
        self.centroides_attendus = centroides_attendus.as_matrix()
        self.centroides_obtenus = centroides_obtenus.as_matrix()
       
    def Evaluate(self) :
        res = 0
        for centre_attendu in self.centroides_attendus :
            min = 9999
            for centre_obtenu in self.centroides_obtenus :
                distance = tools.euclidienne(pd.DataFrame(centre_obtenu),pd.DataFrame(centre_attendu))
                if(min > distance ) :
                    min = distance
            res += min
        Evaluateur.Evaluate(self,res)
        return res
                        
    
    #@classmethod
    def store_results(self,filename,libelle,i) :
        with open(filename, 'a') as fichier:
            fichier.write(libelle+" number "+str(i)+"\n")
            fichier.write("Différence entre les centroides attendus et les centroides obtenus : ",self.results,"\n")
    #@classmethod
    def __str__(self) : 
            print("Résultat de l'évaluation : \n")
            print("La différences des distances entre les centroides attendus et les centroides obtenus  : ",self.results,"\n")
            
        
            
        