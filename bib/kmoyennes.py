
# coding: utf-8

# In[1]:
import pandas as pd
import random
import outils as tools
from clusterer_base import clusterer_base
from clusterer_centroides import clusterer_centroides
import matplotlib.pyplot as plt



#############################  Algorithme des K-moyennes  ###################################################
#    Fonction kmoyennes() qui prend en argument un entier K>1, une base d'apprentissage, un nom de distance,#
#    ainsi que deux nombres: un réel epsilon>0 et un entier iter_max>1,                                     #
#    et qui rend un ensemble de centroides et une matrice d'affectation.                                    #
#############################################################################################################

class kmoyennes(clusterer_base,clusterer_centroides) :
    
    def __init__(self,k,base,fdist,epsilon,iter_max) :
        """ base : DataFrame contenant notre ensemble de données 
        """
        clusterer_base.__init__(self,base)
        self.k = k 
        self.fdist = fdist
        self.epsilon = epsilon
        self.iter_max = iter_max
        self.results = {}
        self.iter = -1
        self.centroides = {}
        self.convergence = -1
        self.dict_affectation = {}
    # Fonction inertie_cluster qui, étant donné une chaîne de caractères donnant le nom de la distance à utiliser,
    # et un DataFrame contenant un ensemble d'exemples, rend la valeur de l'inertie de cet ensemble.
    def inertie_cluster(self,df) :
        inertieCluster = 0
        centroide_ = tools.centroide(df)
        for i,ligne in df.iterrows() :
            inertieCluster += pow(tools.distance(self.fdist,centroide_,pd.DataFrame(ligne).T),2)
        return inertieCluster
    
    
    # Fonction choix_depart() qui étant donné un entier K>1 et une base d'apprentissage (sous la forme d'un DataFrame)
    # de n exemples rend ensemble de centroides contenant K exemples tirés aléatoirement dans la base.
    def choix_depart(self) :
        v = [i for i in range(len(self.base))]
        random.shuffle(v)
        return pd.DataFrame(self.base.iloc [v[0:self.k]])
    
    # Fonction plus_proche() qui, étant donné une fonction de distance à utiliser,
    # un exemple et un ensemble de centroides,
    # rend l'indice (dans l'ensemble) du centroide dont l'exemple est le plus proche. En cas d'égalité de distance, 
    # le centroide de plus petit indice est choisi.
    
    def plus_proche(self,x,centroides):
        distanceMinimum = -1
        indiceMin = 0
        for i,ligne in centroides.iterrows():
            distance_ = tools.distance(self.fdist,x,pd.DataFrame(ligne).T)
            if (distanceMinimum > distance_ or distanceMinimum < 0) :
                distanceMinimum = distance_
                indiceMin = i
        return indiceMin
    
    
    # Fonction affecte_cluster() qui, étant donné un nom de distance,
    # une base d'apprentissage et un ensemble de centroïdes,
    # rend la matrice d'affectation des exemples de la base aux clusters représentés par chaque centroïde.
    def affecte_cluster(self,centroides):
        d = {}
        for i , c in centroides.iterrows():
            d[i]=[]
        for j,x in self.base.iterrows():
            centroideProche = self.plus_proche(pd.DataFrame(x).T,centroides)
            d[centroideProche].append(j)
        return d
    
    #Fonction nouveaux_centroides() qui, étant donné un chaîne de caractères donnant le nom d'une distance,
    # une base d'apprentissage et une matrice d'affectation, rend l'ensemble des centroides correspondant.
    def nouveaux_centroides(self):
        new_centroides = pd.DataFrame()
        c = []
        for i in self.dict_affectation.keys():
            c = self.dict_affectation[i]
            print("c",c)
            new_centroides = new_centroides.append(tools.centroide(pd.DataFrame(self.base.iloc[c[0:]])),ignore_index=True) 
        return new_centroides
    
    # Fonction inertie_cluster(), écrire la fonction inertie_globale() qui,
    # étant donné un nom de distance, une base d'apprentissage et une matrice d'affectation,
    # rend la valeur de l'inertie globale du partitionnement correspondant.
    def inertie_globale(self):
        inertieGlobale = 0
        for i in self.dict_affectation.keys():
            ine = self.inertie_cluster(pd.DataFrame(self.dict_affectation[i]))
            inertieGlobale += ine
        return inertieGlobale
        
    
        
    def plot_clusters(self) :
        plt.figure()
        clusterer_base.plot_clusters(self,self.dict_affectation)
        clusterer_centroides.plot_centroides(self,self.centroides)
        plt.show()
    
    
    #Affichage details    
    def __str__(self) :
        clusterer_base.__str__(self)
        for key,value in self.results.items() :
            print("cluster n° ",key," : ")
            print(value)
        print("Convergence : ",self.convergence)
        print("Nombre d'itérations : ",self.iter)
            
    
    def run(self) :
        new_centroides = {}
        dict_affectation = {}
    
        # initialisation 
        clusters_init = self.choix_depart()
        self.dict_affectation = self.affecte_cluster(clusters_init)
        
        #Boucle
        convergence = 1
        while(convergence > self.epsilon and self.iter < self.iter_max ) :
            new_centroides = self.nouveaux_centroides()
            convergence = self.inertie_globale()
            self.dict_affectation = self.affecte_cluster(new_centroides)
            convergence = abs(convergence - self.inertie_globale())
            self.iter += 1
        self.results = dict_affectation
        self.centroides = new_centroides
        self.convergence = convergence 
        return new_centroides,dict_affectation
    


