
# coding: utf-8

# In[2]:

import numpy as np
import pandas as pd
from datetime import datetime as dt
import matplotlib.pyplot as plt
#get_ipython().magic(u'matplotlib inline')
import math
import random
import matplotlib
import colorsys


# In[3]:

# Fonction de normalisation des données 
def normalisation(dff) :
    df = dff.copy()
    for i in df.columns :
        mi = min(df[i])
        ma = max(df[i])
        df[i] =  (df[i]- mi) / (ma - mi) 
    return df

def read_file (chemin) :
    data =  pd.read_csv(chemin)
    return normalisation(data)
    
    
def createGaussianDataFrame(center,sigma,nb_points):
    return pd.DataFrame(np.random.multivariate_normal(center,sigma,nb_points))


def createXOR(nb_points,var) :
    G1 = createGaussianDataFrame(np.array([1,1]),np.array([[var,0],[0,var]]),nb_points)
    G2 = createGaussianDataFrame(np.array([0,0]),np.array([[var,0],[0,var]]),nb_points)
    G3 = createGaussianDataFrame(np.array([0,1]),np.array([[var,0],[0,var]]),nb_points)
    G4 = createGaussianDataFrame(np.array([1,0]),np.array([[var,0],[0,var]]),nb_points)
    return G1.append(G2.append(G3.append(G4,ignore_index=True),ignore_index=True),ignore_index = True)

def createParal(nb_points,var1,var2) :
    G1 = createGaussianDataFrame(np.array([1,0.6]),np.array([[var1,var2],[var2,var1]]),nb_points)
    G2 = createGaussianDataFrame(np.array([1,0.4]),np.array([[var1,var2],[var2,var1]]),nb_points)
    return G1.append(G2,ignore_index = True)

def createData(nb_points,var) :
    G1 = createGaussianDataFrame(np.array([1,1]),np.array([[var,0],[0,var]]),nb_points)
    G2 = createGaussianDataFrame(np.array([0,0]),np.array([[var,0],[0,var]]),nb_points)
    return G1.append(G2,ignore_index=True)


# In[4]:

# fonction de caclcul de deux points données 
# prend une fonction de calcul de distance en paramétre 
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


# In[5]:

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

# Fonction notIN prend en argument deux liste 
# Rend une valeur booleenne 
# False si la premiere liste est incluse dans la deuxieme
def notIN (l1 , l2 ) :
    for liste in l2 :
        if l1 == liste :
            return False 
    return True


# In[6]:

# Fonction qui prend en parametere une liste
# Renvoie True si la liste passée en parametre contient des doublants

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


# In[7]:

def AffichagesClusters(centroides,dictAffectation,base):
    
    M_data2D = base.as_matrix()
    colonne_X= M_data2D[0:,0]
    colonne_Y= M_data2D[0:,1]
    #Nombre de couluers
    colorNbr = len(centroides)
    #Choix aléatoire des couleurs
    colorNames = list(matplotlib.colors.cnames.keys())
    colors = [i for i in range(len(colorNames))]
    random.shuffle(colors)
    colors =colors[0:colorNbr] 
    
    
    colorMap = [] 
    for i in range(colorNbr) :
        colorMap.append(colorNames[colors[i]])
    colorMap = np.array(colorMap)
    
    categories = np.zeros(len(base))
    categories = np.array(categories)
    for key,values in dictAffectation.items() :
        for indice in values :
            categories[indice] = int(key)
    categories = categories.astype(int)
    
    plt.scatter(colonne_X,colonne_Y,s=100,c=colorMap[categories.astype(int)])
    
    #Affichage des centroides finaux en noir 
    M_data2D = centroides.as_matrix()
    colonne_X= M_data2D[0:,0]
    colonne_Y= M_data2D[0:,1]
    plt.scatter(colonne_X,colonne_Y,color='black')
    plt.show()


# In[8]:

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
        if notIN(list(mat[i]) , m) :
            dicoidEx[i] = mat[i]
    for i in range(len(mat)) :
        if notIN(list(mat[i]),m) :
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
            if notIN(list(mat[k]) , m) :
                dicoidDist[k] = min([dicoidDist[k], fdist(pd.DataFrame(mat[k]),pd.DataFrame(mi))])
            
    return m


# In[9]:

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
                somme += distance_par_dim(Mcurrent[i],exemple,j)     
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


# In[10]:

# Cette fonction permet d affecter tous les exemples aux centroide coresspondant

def assigne_point(base , D,Mcurrent) :
    C = {}
    mat = base.as_matrix(columns = None)
    for i in range (len(D)) :
        C[i] = []
    for exemple in  mat :
        liste = []
        for centre in range(len(Mcurrent)) :
            liste.append(distance_par_dim(exemple , Mcurrent[centre] , D[centre]))
        
        C[liste.index(min(liste))].append (list(exemple))
    return C   


# In[11]:

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
                liste.append (distance_par_dim(exemple , Mcurrent[i] , dim[i]))
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


# In[12]:

# Algorithe du Proclus

def Proclust (nclust ,base,fdist , seuil, iterMax ):
    
    # Declaration des variables
    mat = base.as_matrix()
    dim = partiesliste(range(len(mat[0])),len(mat[0])) # Ensembles des dimmesions possibles
    BestObjectiv = 9999          # Critere d arret
    sigma = list()
    Affect = {}
    D = {}                       # Affection des dimmenssios aux clusters
    C = {}                       # Affection des exemples aux centroides 
    Mcurrent = []                # Ensembles des centroides a l iteration courantes 
    AffectBest = {}
    dimBest = {}
        #######################################################################################
        #################################                      ################################
        ################################ Phase d initialisation ###############################
        #################################                      ################################
        #######################################################################################
    
    niter = 0
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
        # Pour chaque centre calculer la dustence au centre le plus proche
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
            if notIN(list(badClust),[list(clust)]):
                Mcurrent.append(clust)
        coorx = niter % len(Mset) 
        while not notIN(list(Mset[coorx]),Mcurrent) or not notIN(list(Mset[coorx]), [list(badClust)]):
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
                if not notIN(list(val) ,[list(exemple)]) :
                    Affect[cle].append(j)    
    return AffectBest,dimBest,Mbest,Affect, BestObjectiv


# In[13]:

def testVerticalProc():
    positive_pointsV = np.random.multivariate_normal(np.array([0,0]),np.array([[0,1],[0.005,0]]),50)
    negative_pointsV = np.random.multivariate_normal(np.array([1,0]),np.array([[0,1],[0.005,0]]),50)
    pointsV =pd.DataFrame(np.concatenate((positive_pointsV, negative_pointsV), axis=0))
    points = normalisation(pointsV)

    C,D,M,A,sete = Proclust (2 ,pointsV,euclidienne , 0.01,100)
    print("Ensemble des centroides :" , M)
    print("\n")
    print("Ensemble des dimentions par cluster :" ,D)
    print("\n")
    print("Affectation des points  0 :" , A[0])
    print("\n")
    print("Affectation des points  1 :" , A[1])
    AffichagesClusters(pd.DataFrame(M), A,pointsV))


# In[39]:

def testVerticalCenterProc():
    positive_pointsV = np.random.multivariate_normal(np.array([0,0]),np.array([[0,1],[0.005,0]]),50)
    negative_pointsV = np.random.multivariate_normal(np.array([1,0]),np.array([[0.01,0],[0,0.06]]),50)
    pointsV =pd.DataFrame(np.concatenate((positive_pointsV, negative_pointsV), axis=0))
    points = normalisation(pointsV)

    C,D,M,A,sete = Proclust (2 ,pointsV,euclidienne , 0.01,100)
    print("Ensemble des centroides :" , M)
    print("\n")
    print("Ensemble des dimentions par cluster :" ,D)
    print("\n")
    print("Affectation des points  0 :" , A[0])
    print("\n")
    print("Affectation des points  1 :" , A[1])
    AffichagesClusters(pd.DataFrame(M), A,pointsV)



# In[38]:

def testHorizontalProc():
    positive_pointsH = np.random.multivariate_normal(np.array([0,1]),np.array([[1,0],[0,0.005]]),50)
    negative_pointsH = np.random.multivariate_normal(np.array([0,0]),np.array([[1,0],[0,0.005]]),50)
    pointsH =pd.DataFrame(np.concatenate((positive_pointsH, negative_pointsH), axis=0))
    points = normalisation(pointsH)

    C,D,M,A,sete = Proclust (2 ,pointsH,euclidienne , 0.01,45)
    AffichagesClusters(pd.DataFrame(M), A,pointsH)
    print("Ensemble des centroides :" , M)
    print("\n")
    print("Ensemble des dimentions par cluster :" ,D)
    print("\n")
    print("Affectation des points  0 :" , A[0])
    print("\n")
    print("Affectation des points  1 :" , A[1])


# In[45]:

def testHorizontalCenterProc():
    positive_pointsH =  np.random.multivariate_normal(np.array([0,1]),np.array([[1,0],[0,0.005]]),50)
    negative_pointsH =  np.random.multivariate_normal(np.array([1,0]),np.array([[0.06,0],[0,0.01]]),50)
    pointsH =pd.DataFrame(np.concatenate((positive_pointsH, negative_pointsH), axis=0))
    points = normalisation(pointsH)

    C,D,M,A,sete = Proclust (2 ,pointsH,euclidienne , 0.001,60)
    AffichagesClusters(pd.DataFrame(M), A,pointsH)
    print("Ensemble des centroides :" , M)
    print("\n")
    print("Ensemble des dimentions par cluster :" ,D)
    print("\n")
    print("Affectation des points  0 :" , A[0])
    print("\n")
    print("Affectation des points  1 :" , A[1])


# In[15]:

def testCrosseProc():
    positive_pointsP = np.random.multivariate_normal(np.array([0,1]),np.array([[0,1],[0.08,0]]),50)
    negative_pointsP = np.random.multivariate_normal(np.array([0,0]),np.array([[1,0],[0,0.08]]),50)
    pointsP =pd.DataFrame(np.concatenate((positive_pointsP, negative_pointsP), axis=0))
    pointsP = normalisation(pointsP)

    C,D,M,A,sete = Proclust (2 ,pointsP,euclidienne , 0.01,45)
    AffichagesClusters(pd.DataFrame(M), A,pointsP)
    print("Ensemble des centroides :" , M)
    print("\n")
    print("Ensemble des dimentions par cluster :" ,D)
    print("\n")
    print("Affectation des points  0 :" , A[0])
    print("\n")
    print("Affectation des points  1 :" , A[1])



# In[16]:

def simpleTestProc():
    point =  createData(50,0.02)
    C,D,M,A,sete = Proclust (2 ,pd.DataFrame(point),euclidienne , 0.04,100)
    AffichagesClusters(pd.DataFrame(M), A,pd.DataFrame(point))
    print("Ensemble des centroides :" , M)
    print("\n")
    print("Ensemble des dimentions par cluster :" ,D)
    print("\n")
    print("Affectation des points  0 :" , A[0])
    print("\n")
    print("Affectation des points  1 :" , A[1])



# In[17]:

def testXorProc():
    pointXor = createXOR(25,0.009)
    C,D,M,A,sete = Proclust (4 ,pointXor,euclidienne , 0.001,200)
    AffichagesClusters(pd.DataFrame(M), A,pointXor)
    print("Ensemble des centroides :" , M)
    print("\n")
    print("Ensemble des dimentions par cluster :" ,D)
    print("\n")
    print("Affectation des points  0 :" , A[0])
    print("\n")
    print("Affectation des points  1 :" , A[1])
    print("\n")
    print("Affectation des points  2 :" , A[2])
    print("\n")
    print("Affectation des points  3 :" , A[3])




# In[20]:

def read_file (chemin) :
    data =  pd.read_csv(chemin)
    data = data.as_matrix()
    data = pd.DataFrame(data)
    return normalisation(data)


# In[25]:

def apply_proc(base , parametre) :
    C,D,M,A,sete = Proclust(parametre[0] , base, parametre[1],parametre[2],parametre[3])
    print("Valeur optimale trouvée : ",sete)
    print("\n")
    print("Ensemble des centroides : " , M)
    print("\n")
    print("Ensemble des dimentions par cluster :" ,D)
    print("\n")
    
    for cle in A.keys() :
        print("Affectation des points , ",cle," :" , A[cle])
        print("\n")
    AffichagesClusters(pd.DataFrame(M), A,base)            

