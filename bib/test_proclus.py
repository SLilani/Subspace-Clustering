
# coding: utf-8

# In[14]:

import outils as outils
import proclus as proclus
import dataplayer as dt
import pandas as pd
import matplotlib.pyplot as plt
import Evaluateur_centroide as ec
import Evaluateur_inertie as ei


# In[10]:

def apply_proc(base , parametre) :
    A,M,D,sete = proclus.Proclust(parametre[0] , base, parametre[1],parametre[2],parametre[3])
    print ("Valeur optimale trouvée : ",sete)
    print ("\n")
    print ("Ensemble des centroides : " , M)
    print ("\n")
    print ("Ensemble des dimentions par cluster :" ,D)
    print ("\n")
    
    for cle in A.keys() :
        print ("Affectation des points , ",cle," :" , A[cle])
        print ("\n")
    plt.figure()
    dt.affichagesClusters(pd.DataFrame(M), A,base)    
    plt.show()
    return pd.DataFrame(M),A

# In[17]:

parametres = [2,outils.euclidienne,0.01,40]
data,c = dt.create_gauss_vertical(50)
centroide_trouve ,dict_aff= apply_proc(data , parametres)
e1 = ec.Evaluateur_centroide(c,centroide_trouve) 
e1.Evaluate()
e1.__str__()
e = ei.Evaluateur_inertie(data,dict_aff,centroide_trouve)
e.Evaluate(outils.euclidienne)
e.__str__()

# In[18]:

parametres = [2,outils.euclidienne,0.01,40]
data,c = dt.create_gauss_vertical_cent(50)
centroide_trouve = apply_proc(data , parametres)
e1 = ec.Evaluateur_centroide(c,centroide_trouve) 
e1.Evaluate()
e1.__str__()


# In[19]:

parametres = [2,outils.euclidienne,0.01,40]
data,c = dt.create_gauss_horizontal(50)
centroide_trouve = apply_proc(data , parametres)
e1 = ec.Evaluateur_centroide(c,centroide_trouve) 
e1.Evaluate()
e1.__str__()


# In[ ]:

parametres = [2,outils.euclidienne,0.01,40]
data ,c= dt. create_gauss_horizontal_cent(50)
centroide_trouve = apply_proc(data , parametres)
e1 = ec.Evaluateur_centroide(c,centroide_trouve) 
e1.Evaluate()
e1.__str__()


# In[ ]:

parametres = [2,outils.euclidienne,0.01,40]
data,c = dt. create_gauss_cross(50)
centroide_trouve = apply_proc(data , parametres)
e1 = ec.Evaluateur_centroide(c,centroide_trouve) 
e1.Evaluate()
e1.__str__()


# In[ ]:

parametres = [2,outils.euclidienne,0.01,40]
data,c = dt.create_gauss_ronde(50,0.01)
centroide_trouve = apply_proc(data , parametres)
e1 = ec.Evaluateur_centroide(c,centroide_trouve) 
e1.Evaluate()
e1.__str__()


# In[ ]:

parametres = [4,outils.euclidienne,0.01,40]
data,c = dt. create_xor(50,0.01)
centroide_trouve = apply_proc(data , parametres)
e1 = ec.Evaluateur_centroide(c,centroide_trouve) 
e1.Evaluate()
e1.__str__()


# In[9]:

data = outils.read_file("donne_mod.csv")
parametres = [2,outils.euclidienne,0.08,40]
apply_proc(data , parametres)

