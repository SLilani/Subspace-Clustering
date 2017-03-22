import numpy as np
import pandas as pd
from datetime import datetime as dt
import matplotlib.pyplot as plt
#get_ipython().magic(u'matplotlib inline')
import math
import random
import matplotlib
import colorsys



# Fonction de normalisation des donnees 
def normalisation(dff) :
    df = dff.copy()
    for i in df.columns :
        mi = min(df[i])
        ma = max(df[i])
        df[i] =  (df[i]- mi) / (ma - mi) 
    return df

# fonction de caclcul de deux points donnees 
# prend une fonction de calcul de distance en parametre 
def distance(fdist,x1,x2):
    return fdist(x1,x2)


# fonction de caclcul de distance euclidienne entre deux points 
def euclidienne(x1,x2):
    return math.sqrt(pow(x2.values-x1.values,2).sum())


# fonction de caclcul de distance de mannathan entre deux points 
def mannathan(x1,x2):
    return abs(x2.values-x1.values).sum()

# Cette Fonction permet de calculer la distence entre deux exemples 
# Suivant chaque dimension
# Prend en parametre deux exemples x1 et x2 ainsi qu une liste representant l ensemble des dimenssions possibles
def distance_par_dim(x1 , x2 , dim) :
    somme = 0
    for d in dim :
        somme  += pow(x2[d]-x1[d],2)
    return math.sqrt(somme)

# fonction de calcul d'un centroide 
def centroide(df) :
    return pd.DataFrame(np.mean(df)).T


# Fonctions utiles

# Fonctions qui renvoie les combinaisons possible entre les valeurs de deux listes
def combliste(l1,l2):
    res = []
    for e1 in l1 :
        for e2 in l2 :
            res.append(e1+e2)        
    return res

# Fonctions qui renvoie les combinaisons possible entre les valeurs de plusieurs listes
def comblistes(ensListes):
    res = ensListes[0]
    for i in range(len(ensListes) - 1) :
        res = combliste(res,ensListes[i + 1])
    resultat = []
    for e in res :
        e = np.array(e)
        resultat.append(e)
    for i in range(len(resultat)) :
        resultat[i] = resultat[i].reshape(len(resultat[i])/2,2)
    return resultat

# Fonctions qui renvoie les listes des intervalles pour chaque attribut (dimension) 
def getIntervalles(array_Bornes) :
    res = []
    for i in range(len(array_Bornes[0])) :
        colonne = list(array_Bornes[0:,i])
        col = []
        for v in range(len(colonne) - 1) :
            col.append([colonne[v],colonne[v+1]])
        res.append(col)
    return res
 
# Fonction qui renvoie l'ensemble des partie d'une liste    
def partiesliste(seq,dim):
    p = []
    i, imax = 0, 2**len(seq)-1
    while i <= imax:
        s = []
        j, jmax = 0, len(seq)-1
        while j <= jmax:
            if (i>>j)&1 == 1:
                s.append(seq[j])
            j += 1
        p.append(s)
        i += 1 
    del p[0]
    resultat = []
    for i in range(dim) :
        for e in p :
            if(len(e) == i +1 ) :
                resultat.append(e)
    
    return resultat 

# Fonction qui prend en parametere une liste
# Renvoie True si la liste passee en parametre contient des doublants

def ifDoublant(liste) : 
    for i in range(len(liste)-1) :
        for j in range(i+1,len(liste)):
            if not notIN((liste[i]),[liste[j]]) :
                return True
    return False

# Fonction notIN prend en argument deux liste 
# Rend une valeur booleenne 
# False si la premiere liste est incluse dans la deuxieme
def notIN (l1 , l2 ) :
    for liste in l2 :
        if l1 == liste :
            return False 
    return True


# Foction qui prend en parametre une chaine de caractere 
# qui est le chemin d un fichier csv ou se trouve les donnees
# et retourne un dataframe
def read_file (chemin) :
    data =  pd.read_csv(chemin)
    data = data.as_matrix()
    data = pd.DataFrame(data)
    return normalisation(data)
