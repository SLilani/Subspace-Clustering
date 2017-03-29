
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import math
import random
import bib.outils as outils
import bib.output as o
# Un petit algorithme d initialisation 
# Prend en argument une base d exemple un entier et une fonction de distence 
# Retourn un ensemble de points de taille egale au nombre passe en parametre

def Greedy(base , nclust ,fdist):
    mat = base.as_matrix(columns = None)
    for exemple in mat :
        exemple = list(exemple)
    m = list()
    dist = 0
    exMax = []
    for exemple in mat :
        if dist < fdist(pd.DataFrame(exemple),pd.DataFrame([0,0])) :
            exMax = exemple
            dist =  fdist(pd.DataFrame(exemple),pd.DataFrame([0,0]))  
    dicoDist = {}
    m.append(list(exMax))
    maxe = 0
    dicoidEx = {}
    dicoidDist = {}
    for i in range(len(mat)) :
        if outils.notIN(list(mat[i]) , m) :
            dicoidEx[i] = mat[i]
    for i in range(len(mat)) :
        if  outils.notIN(list(mat[i]),m) :
            dicoidDist[i] = fdist(pd.DataFrame(mat[i]),pd.DataFrame(exMax))
    for i in range(2 , nclust+1) :
        listeDist = dicoidDist.values()
        maxi = max(listeDist)
        mi = list()
        idf = 0
        for clee in dicoidDist.keys() :
            if dicoidDist[clee] == maxi :
                idf = clee
        mi = dicoidEx[idf]
        dicoidDist[idf] = 0
        m.append(list(mi))
        for k in range(len(mat)) :
            if  outils.notIN(list(mat[k]) , m) :
                dicoidDist[k] = min([dicoidDist[k], fdist(pd.DataFrame(mat[k]),pd.DataFrame(mi))])         
    return m


# Cette Fonction permet de trouver pour
# chaque cluster la dimmenssion ou l'attribut le plus pertinant

def find_dimensions(L,dim,Mcurrent):
    x = {}
    Z = {}
    D = {}
    for i in L.keys() :
        x[i] = []
        for j in (dim) :
            somme = 0
            for exemple in L[i] :    
                somme += outils.distance_par_dim(Mcurrent[i],exemple,j)     
            x[i].append(somme/len(L[i]))
    for centre in range(len(Mcurrent)) :
        sommeY = 0
        Z[centre] = []
        for j in range(len(dim)) :
            sommeY += x[centre][j]
        Y = (sommeY/len(dim))
        sommeS = 0 
        for j in range(len(dim)) :
            sommeS += (x[centre][j]- Y)**2 
        sommeS = math.sqrt(sommeS)
        sigma = sommeS / (len(dim) - 1)   
        for z in range(len(dim)) :
            Z[centre].append((x[centre][z] -Y)/sigma)
    for center in Z.keys() :
        l = Z[center]
        for x in range(len(l)) :
                l[x] = abs(l[x])
        D[center] =dim[ l.index(max(l))]
    return D       




# Cette fonction permet d affecter tous les exemples aux centroide coresspondant

def assigne_point(base , D,Mcurrent) :
    C = {}
    mat = base.as_matrix(columns = None)
    for i in range (len(D)) :
        C[i] = []
    for exemple in  mat :
        liste = []
        for centre in range(len(Mcurrent)) :
            liste.append(outils.distance_par_dim(exemple , Mcurrent[centre] , D[centre]))
        
        C[liste.index(min(liste))].append (list(exemple))
    return C   




# Cette fonction permet d evaluer l inertie des cluster 
# elle permet aussi d obtenir le mauvais centroide

def evaluate_cluster(Clust,dim, Mcurrent) :
    dicoDistMoy = {}
    BadClust =[]
    for i in Clust.keys() :
        if len(Clust[i])>0 :
            dicoDistMoy[i] = []
            liste = []
            for exemple in Clust[i]:
                liste.append (outils.distance_par_dim(exemple , Mcurrent[i] , dim[i]))
            s = np.array(liste).sum()
            dicoDistMoy[i].append(s/len(liste)) 
    w = {}
    for c in dicoDistMoy.keys():
        somme = 0
        for dist in dicoDistMoy[c]:
            somme += dist
        w[c] = (somme/len(dim))
    somme = 0
    maxi = max(w.values())
    for c in w.keys() :
        if w[c] == maxi :
            indice = c
    BadClust = Mcurrent[indice]
    for i in dicoDistMoy.keys() :
        somme+= (len(dicoDistMoy[i]) * w[i])
    return somme/len(Clust) ,BadClust   



# Algorithe du Proclus

def Proclust (nclust ,base,fdist , seuil, iterMax ):
    
    # Declaration des variables
    mat = base.as_matrix()
    dim = outils.partiesliste(range(len(mat[0])),len(mat[0])) # Ensembles des dimmesions possibles
    BestObjectiv = 9999          # Critere d arret
    sigma = list()
    Affect = {}
    D = {}                       # Affection des dimmenssios aux clusters
    C = {}                       # Affection des exemples aux centroides 
    Mcurrent = []                # Ensembles des centroides a l iteration courentes 
    AffectBest = {}				 # Dictionnaire d affectation a la convergence 
    dimBest = {}				 # Affectation des dimention pour chaque cluster a la convergence
    niter = 0
        #######################################################################################
        #################################                      ################################
        ################################ Phase d initialisation ###############################
        #################################                      ################################
        #######################################################################################
    
    Mset= Greedy(base , 10,fdist)
    for j in range(nclust) :
        Mcurrent.append(Mset[j])
    Mbest = Mcurrent
        #######################################################################################
        #################################                     #################################
        ################################ Phase des traitements ################################
        #################################                     #################################
        #######################################################################################     
      
    while BestObjectiv > seuil and niter < iterMax  :
        # Pour chaque centre calculer la distence au centre le plus proche
        #######################################################################################
        for mi in Mcurrent : 
            mini = 900000000
            for i in range(len(Mcurrent)) :
                if mini > fdist(pd.DataFrame(Mcurrent[i]),pd.DataFrame(mi)) and Mcurrent[i] != mi  :
                    mini = fdist(pd.DataFrame(Mcurrent[i]),pd.DataFrame(mi))
            sigma.append(mini)
        #######################################################################################
                
            
        # Affectation des exemples suivant les distences calculees a l etape precedente 
        #######################################################################################
        dicoAffect = {}
        for centre in range(len(Mcurrent)) :
            dicoAffect[centre] = []
            for j, x in base.iterrows() :
                exemple = list(x)
                if fdist((pd.DataFrame(Mcurrent[centre])),pd.DataFrame(list(exemple))) < sigma[centre] :
                    dicoAffect[centre].append(exemple)
        #######################################################################################
        
        
        D = find_dimensions(dicoAffect,dim,Mcurrent)   
        C = assigne_point(base , D,Mcurrent)
        #######################################################################################
        #################################                  ####################################
        ################################ Phase d evaluation ###################################
        #################################                  ####################################
        #######################################################################################
        val ,badClust = evaluate_cluster(C,D, Mcurrent)
        
        Mtrans= Mcurrent
        Mcurrent = []
        for clust in Mtrans :
            if  outils.notIN(list(badClust),[list(clust)]):
                Mcurrent.append(clust)
        coorx = niter % len(Mset) 
        while not  outils.notIN(list(Mset[coorx]),Mcurrent) or not  outils.notIN(list(Mset[coorx]), [list(badClust)]):
            coorx = (coorx +1) % len(Mset)
        m1 = Mset[coorx]
        
        if val < BestObjectiv :
            Mbest =[]
            dimBest = D
            BestObjectiv = val
            for ex in Mcurrent :
                Mbest.append(ex)
            AffectBest = C
            Mbest.append(m1)
        Mcurrent.append(m1)
        niter += 1 
    listeEx = []

    for cle in AffectBest.keys() :
        listeEx = (pd.DataFrame(AffectBest[cle]).T).as_matrix()
        for i in range(len(listeEx)) :
             Mbest[cle][i] = np.mean(listeEx[i])
        
        # Creation d un dictionnaire contenant les centroides et les exemple correspondant 
        # les valeur seront les identifiant des exemple pour plus de confords visuel
    for cle in AffectBest.keys() : 
        Affect[cle] =[]
        for val in AffectBest[cle] :
            for j, x in base.iterrows() :
                exemple = []
                exemple.append(x[0])
                exemple.append(x[1])
                if not  outils.notIN(list(val) ,[list(exemple)]) :
                    Affect[cle].append(j)    
    output_proclus = o.output([Affect,Mbest,dimBest,BestObjectiv],"proclus") 
    output_proclus.construct_clusters()
    return output_proclus  

