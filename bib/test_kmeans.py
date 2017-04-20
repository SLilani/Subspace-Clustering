# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 21:55:40 2017

@author: MDK
"""



# coding: utf-8

# In[3]:

import dataplayer as dtp
import kmoyennes as kmoyennes 
import Evaluateur_inertie as ei
import Evaluateur_centroide as ec
import outils as tools

# In[4]:

# Fonction de teste 
# Prend en paramétre base : Dataframe, nb_intervalles : nombre d'intervalles pour la discrétisation des dimensions
# et le taux : pour la densité des cellules

def test_kmeans(base,k,fdist,epsilon,iter_max,centroides_attendus) :
    kmeans_test = kmoyennes.kmoyennes(k,base,fdist,epsilon,iter_max)
    centroides_trouvees,aff = kmeans_test.run()
    kmeans_test.plot_clusters()
    kmeans_test.__str__()
    e = ei.Evaluateur_inertie(base,aff,centroides_trouvees)
    e.Evaluate(tools.euclidienne)
    e1 = ec.Evaluateur_centroide(centroides_trouvees,centroides_attendus)
    e1.Evaluate()
    e1.__str__()
    e.__str__()
    

# In[5]:

##########################################################################################################################
################################################### TESTS CLIQUE #########################################################
##########################################################################################################################

k = 3
fdist = tools.euclidienne
iter_max = 100
epsilon = 0.01
# In[6]:

################################################ TEST XOR ################################################################
nb_points = 1
var = 0.005
        
data_test_one, centroides = dtp.create_xor(nb_points,var)
test_kmeans(data_test_one,k,fdist,epsilon,iter_max,centroides)


# In[7]:

####################################### TEST GAUSSIENNES VERTICALES PARALLELES ###########################################
nb_points = 100       
data_test_two,centroides = dtp.create_gauss_vertical(nb_points)
test_kmeans(data_test_two,k,fdist,epsilon,iter_max,centroides)


# In[8]:

######################################### TEST GAUSSIENNES HORIZONTALES PARALLELES #######################################
nb_points =100     
data_test_two,centroides = dtp.create_gauss_horizontal(nb_points)
test_kmeans(data_test_two,k,fdist,epsilon,iter_max,centroides)


# In[9]:

################################################ TEST GAUSSIENNES HORIZONTALES centre ############################################
nb_points =100     
data_test_two ,centroides= dtp.create_gauss_horizontal_cent(nb_points)
test_kmeans(data_test_two,k,fdist,epsilon,iter_max,centroides)


# In[10]:

################################################ TEST GAUSSIENNES HORIZONTALES centre ############################################
nb_points =100     
data_test_two ,centroides= dtp.create_gauss_vertical_cent(nb_points)
test_kmeans(data_test_two,k,fdist,epsilon,iter_max,centroides)


# In[12]:

################################################ TEST GAUSSIENNES HORIZONTALES centre ############################################
nb_points =100     
data_test_two ,centroides= dtp.create_gauss_cross(nb_points)
test_kmeans(data_test_two,k,fdist,epsilon,iter_max,centroides)
    