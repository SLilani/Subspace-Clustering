
# coding: utf-8

# In[5]:

import numpy as np
import bib.outils as tools
from random import randrange
import bib.dataplayer as dtp
import matplotlib.pyplot as plt
import pandas as pd

# In[6]:

# - - Classe pour représenter une cellule (units)
class unit :
    def __init__(self,id,i,j) :
        """ d : Nombre de dimensions
            lh_Array : Tableau représentant le début et la fin de l'interavalle pour chaque attribut  
        """
        self.lh_Array = []
        self.id = id
        self.dense = False
        self.nuCluster = -1
        self.connectedCells = []
        self.indices =(i,j)
        
    # revoie le début  et la fin de l'intervalle de la dimension indice
    def getLH(self,indice) :
        return self.lh_Array[indice]
    
    # rajoute le début  et la fin de l'intervalle de la dimension indice
    def addLH(self,lh) :
        self.lh_Array.append(lh)
    
    # affichage d'une cellule 
    def display(self) :
        print("cellule id : ",self.id)
        for ligne in self.lh_Array : 
            print("intervalle : ",ligne)
        print("\n density : ",self.dense)
    
    # Fonction qui renvoie true si un point appartient a la cellule pour un ensemble de dimensions , False sinon
    def contains(self,x,dim) :
        for d in dim :  
            if(x[d] >= self.lh_Array[0][d][1] or x[d] < self.lh_Array[0][d][0]):
                return False
        return True
    
    # Fonction qui renvoie true si la cellule est dense pour un ensemble de dimensions, False sinon
    def is_a_dense_cell(self,base,dim,taux) :
        nb_in = 0
        for x in base.values :
            if(self.contains(x,dim)):
                nb_in += 1
        #print "la cellule ",self.id,"contient ",nb_in ," points"
        #print "la cellule ",self.id , "est dans l'intervallle ",self.lh_Array
        return ( nb_in / (len(base) * 1.0) ) > taux
    
    # Fonction qui renvoie true si la cellule a au moins une face en commun avec la cellule donner en paramétre
    # pour un ensemble de dimension, False sinon
    def has_commonFace_with(self,cell,dim):
        # Nombre d'intervalles différents
        if(not cell.dense) : return False
        nb_dif = 0
        
        # htk = l'tk or h'tk = ltk
        dif = False
        if(len(dim) == 1 ) :
            d = dim[0]
            dif = dif or  self.lh_Array[0][d][0] == cell.lh_Array[0][d][0] and self.lh_Array[0][d][1] == cell.lh_Array[0][d][1]
            if(dif) : return True
        
        for d in dim :
            if(self.lh_Array[0][d][0] != cell.lh_Array[0][d][0] or self.lh_Array[0][d][1] != cell.lh_Array[0][d][1]) :
                nb_dif += 1
            if( nb_dif > 1 ) :# Si le nombre d'intervalles différents est superieur a 1 on renvoie False
                return False
           
            else : # Sinon on verifie que l'intervalle en question relie les deux cellules
                if dif == False :
                    dif = (self.lh_Array[0][d][0] == cell.lh_Array[0][d][1] or  self.lh_Array[0][d][1] == cell.lh_Array[0][d][0])
        #print "cellule ",self.id," avec ",cell.id," = ",dif
        return dif          
    

    # Fonction qui renvoie true si la cellule a au moins une face en commun avec la cellule donner en paramétre
    # pour un ensemble de dimension, False sinon
    def is_connected_with(self,cell,dim,cellules ):
        cellules_copy = list(cellules)
        if(not cell.dense) : return False  
        #print "face communes entre ",self.id," est ", cell.id, " = ",self.has_commonFace_with(cell,dim) or self.id == cell.id
        if(self.has_commonFace_with(cell,dim) or self.id == cell.id) : return True
        for c in cell.all_common_faces(dim,cellules) :
            if(self.has_commonFace_with(c,dim)): return True
        if(len(cellules_copy) == 0 ) : return False
        
        for c in cellules_copy :
            if(self.has_commonFace_with(c,dim) and cell.has_commonFace_with(c,dim)) : return True
        
        c = cellules_copy.pop(0)
        if(self.is_connected_with(c,dim,cellules_copy) and cell.is_connected_with(c,dim,cellules_copy) ) :
            return True
        return False
    
    # Fonction qui renvoie l'ensemble des element qui ont une face en commun avec la cellule
    def all_common_faces(self,dim,cellules ):
        res = []
        for c in cellules :
            if(self.has_commonFace_with(c,dim)):
                res.append(c)
        return res
    
    # Fonction qui renvoie une liste contenant tout les id des cellules avec lesquelles elle est connectée
    # pour un ensemble de dimension
    def all_connected(self,dim,cellules) :
        res =[]
        indices = []
        for c in cellules :
            if(self.is_connected_with(c,dim,cellules)) :
                res.append(c.id)
                indices.append(c.indices)
        return res,indices
    
    # Fonction qui renvoie l'ensemble des points contenus dans une cellules pour un ensemble de dimensions donné
    
    def get_points(self,dim,base) :
        res = []
        for i,x in base.iterrows() :
            if(self.contains(x,dim)) :
                res.append(i)
        #print "les point dans ",self.id," sont ",res
        return list(set(res))
            


# In[7]:




# In[8]:



# - - Classe pour représenter une grille de cellules (Grid)
class grid :
    def __init__(self,base,nb_intervalles,taux) :
        """ base : DataFrame contenant notre ensemble de données 
        """
        self.base = base
        self.taux = taux
        self.nb_intervalles = nb_intervalles
        self.dim = len(base.max())
        # Noms des dimensions (attributs)
        self.dNames = [i for i in self.base.columns]
        
    # Renvoie le nom de l'attributs indice
    def create_grid (self) :
        # Tableaux contennat la taille de chaque intervalles pour les dimensions
        l_array = self.base.max()
        h_array = self.base.min()
        intervalles = ( l_array - h_array )   / self.nb_intervalles 
        # Grille contenant nb_intervalles^dim cellules
        self.grille = []
        # Array contenant toutes les valeurs successives de chaque attribut dans chaque intervalle (dimension) 
        array_bornes = np.array([list(h_array + intervalles * i) for i in range(self.nb_intervalles +1)])
        array_bornes[0] -= 1
        array_bornes[len(array_bornes) - 1 ] += 1
        # Liste contenant toutes les valeurs d'intervalles prises par chacune des cellules
        cell_values = tools.comblistes(tools.getIntervalles(array_bornes))
    
        for id in range(self.nb_intervalles**self.dim) :    
                i = id // self.nb_intervalles
                j = id % self.nb_intervalles
                u = unit(id,i,j)
                u.addLH(cell_values[id])
                self.grille.append(u)
    
    # Fonction qui marque toutes les cellules dense pour un certaint taux et pour un ensemble de dimensions comme danse 
    def mark_cells(self,dim) :
        for c in self.grille :
            if(c.is_a_dense_cell(self.base,dim,self.taux)) :
                c.dense = True  
            else : c.dense = False
    
    # Fonction qui renvoie les cellules qui forment un polygone maximal,
    # parmi toutes les cellules danses a partir d'une cellule tirée au hasard
    
    def max_polygone(self, marked_cells,dim ) :
        res = []
        if(len(marked_cells) > 0 ) :
            nombreAleatoire = randrange(0,len(marked_cells))
            #mini_i,mini_j,indice_i,indice_j = 0,0,0,0
            #for i in range(len(marked_cells)):
              #  ind_i,ind_j = marked_cells[i].indices
               # if(ind_i < mini_i) :
              #      indice_i = i
               # elif(ind_j < mini_j):
               #ndice_j =i
            
            c = marked_cells[nombreAleatoire]
            I,J = c.indices
            
            #print("IJIJIJ",I,J)
            
            res,indices = c.all_connected(dim,marked_cells)
            
            max_i = max([i for i,J in indices])
            min_i =min([i for i,J in indices])
            max_j = max([j for I,j in indices])
            min_j =min([j for I,j in indices])
            #print("diiiiim : ",dim)
            #print("maxi = ",max_i,[i for i,J in indices])
            #print("mini = ",min_i)
            #print("maxj = ",max_j,[j for I,j in indices])
            #print("minj = ",min_j)
            for k in range(len(indices)) :
                i,j = indices[k]
                if((i not in range(min_i,max_i + 1 , 1))or(j not in range(min_j,max_j + 1 , 1))) :
                    res.pop(k)
        return res
    
    # Fonction qui renvoie les Clusters et leurs Cellules pour un ensemble de dimensions donné
    def get_clusters(self,dim) :
        clusters = dict()
        for c in self.grille :
            c.dense = False 
        # On marque toutes les cellules dense 
        self.mark_cells(dim)
        # Liste des cellules marquées 
        cells_marked = []
        for c in self.grille :
            if(c.dense == True ) : 
                cells_marked.append(c)
        key = 0
        while(len(cells_marked) > 0 ):
            r = self.max_polygone(cells_marked,dim)
            cell = cells_marked
            cells_marked =[]
            clusters[key] = r 
            for e in cell :
                if(e.id not in r ):
                    cells_marked.append(e)      
            key +=1
        #print("les cluster dans dim : ",dim,"sont : ",clusters)
        return clusters
   
    # Fonction qui renvoie toutes les cellules dont les id sont dans la liste donnée en parametre 
    def get_cells(self,list_id) :
        res = []
        for c in self.grille :
            if(c.id in list_id):
                res.append(c)
        return res
    
    # Fonction qui renvoie l'ensemble des points pour un cluster  (a partir de ses cellules ) pour un ensemble de dimensions donné
    def get_cluster_points(self,dim,cells_ids) :
        cells = self.get_cells(cells_ids)
        points = []
        for i in range(len(cells)) :
            points += cells[i].get_points(dim,self.base)
        points = list(set(points))
        #df = pd.DataFrame(self.base.iloc[points[0:]])
        return points

    # Fonction qui renvoie l'ensemble des points pour tous les cluster  (a partir de leurs cellules ) pour un ensemble de dimensions donné
    def get_all_clusters_points(self,dim,dict_clusters_id_cells) :
        clusters_points = {}
        for c,v in dict_clusters_id_cells.items() :
            clusters_points[c] = self.get_cluster_points(dim,v)
        dim_name =""
        for d in dim :
            dim_name += str(d)
        return clusters_points,dim_name
     
    # Renvoie le nombre de dimensionscell_values
    def get_dimension(self) :
        return self.dim
    
    # Renvoie la grille 
    def get_grid(self) :
        return self.grille

    # Affichage d'une grille
    def display(self) :
        print("############### GRID ##################\n")
        for cell in self.grille :
            cell.display()


# In[9]:

# - - Classe pour Clique
class clique :
    def __init__(self,base,nb_intervalles,taux) :
        """ base : DataFrame contenant notre ensemble de données 
        """
        self.base = base
        self.taux = taux
        self.nb_intervalles = nb_intervalles
        self.dim = len(base.max())
        # Noms des dimensions (attributs)
        self.dNames = [i for i in self.base.columns]
        self.grid = grid(base,nb_intervalles,taux)
        self.results = {}
    
    def affichage_clique(self) :
        for key,value in self.results.items() :
            print("les clusters pour la dimension ",key," sont  : ")
            if(len(self.results[key]) != 0 ):
                plt.figure()
                dtp.affichagesClusters([],self.results[key],self.base)
                plt.show()
    
    
    
    
    # Fonction qui renvoie un dictionnaire de dictionnaires, 
    # chaque dictionnaire contient tout les clusters pour un certain ensemble de dimensions
    def run(self) :
        self.grid.create_grid()
        dict_all_clust_all_dim = {}
        for dim in tools.partiesliste([i for i in range(self.dim)],self.dim) :
            if( any([self.grid.get_clusters([x]) for x in dim])) :
                # Récupération de tout les clusters pour dim (cellules)
                dict_clusters_id_cells = self.grid.get_clusters(dim)
                # Récupération de tout les clusters pour dim (points)
                value,name = self.grid.get_all_clusters_points(dim,dict_clusters_id_cells)
                dict_all_clust_all_dim[name] = value
        self.results = dict_all_clust_all_dim
        return self.results
    
    
    
    def output_for_eval_inertie(self) :
        dict_affectations = {}
        centroides = pd.DataFrame()
        for key ,value in self.results.items() :
            for key1,value1 in value.items() :
                dict_affectations[str(key)+"-"+str(key1)] = value1
                centroides = centroides.append(tools.centroide(self.base.iloc[value1]),ignore_index=True)
        return dict_affectations,centroides
        

