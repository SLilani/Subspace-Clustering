
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
from datetime import datetime as dt
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import bib.dataplayer as dtp
import bib.outils as tools


# In[2]:

# - - Classe pour représenter une cellule (units)
class output :
    def __init__(self,parameters,name) :
        self.parameters = parameters 
        self.clusters = {}
        self.centroides = {}
        self.name = name
        if(name == "proclus") :
            self.dim_best = {}
            self.best_objectif = 0
        
    # revoie le nombre de dimensions
    
    def get_dim_parameters(self):
        return len(self.parameters)
    
    def construct_clusters(self) :
        self.clusters = self.parameters[0]
        if(self.name == "proclus"):
            self.centroides = self.parameters[1]
            self.dim_best = self.parameters[2]
            self.best_objectif = self.parameters[3]
    # affichage d'un cluster
    def get_output(self) :
        if(self.name == "proclus"):
            return self.clusters,self.centroides,self.dim_best,self.best_objectif
        else : 
            return self.clusters,self.centroides
    

