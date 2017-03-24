
# coding: utf-8

# In[3]:

import bib.dataplayer as dtp
import bib.clique as clique
import matplotlib.pyplot as plt


# In[4]:

# Fonction de teste 
# Prend en paramétre base : Dataframe, nb_intervalles : nombre d'intervalles pour la discrétisation des dimensions
# et le taux : pour la densité des cellules

def test_clique(base,nb_intervalles,taux) :
    clique_test = clique.clique(base,nb_intervalles,taux)
    l_affectations = clique_test.run()
    plt.figure()
    print("- Les clusters pour la dimension  0 sont : ")
    dtp.affichagesClusters([],l_affectations['0'],base)
    plt.show()
    plt.figure()
    print("- Les clusters pour la dimension  1 sont : ")
    dtp.affichagesClusters([],l_affectations['1'],base)
    plt.show()
    plt.figure()
    print("- Les clusters pour la dimension  01 sont : ")
    dtp.affichagesClusters([],l_affectations['01'],base)
    plt.show()


# In[5]:

##########################################################################################################################
################################################### TESTS CLIQUE #########################################################
##########################################################################################################################


# In[6]:

################################################ TEST XOR ################################################################
nb_points = 100
var = 0.005
        
data_test_one = dtp.create_xor(nb_points,var)
test_clique(data_test_one,3,0.1)


# In[7]:

####################################### TEST GAUSSIENNES VERTICALES PARALLELES ###########################################
nb_points = 100       
data_test_two = dtp.create_gauss_vertical(nb_points)
test_clique(data_test_two,3,0.01)


# In[8]:

######################################### TEST GAUSSIENNES HORIZONTALES PARALLELES #######################################
nb_points =100     
data_test_two = dtp.create_gauss_horizontal(nb_points)
test_clique(data_test_two,3,0.01)


# In[9]:

################################################ TEST GAUSSIENNES HORIZONTALES centre ############################################
nb_points =100     
data_test_two = dtp.create_gauss_horizontal_cent(nb_points)
test_clique(data_test_two,3,0.1)


# In[10]:

################################################ TEST GAUSSIENNES HORIZONTALES centre ############################################
nb_points =100     
data_test_two = dtp.create_gauss_vertical_cent(nb_points)
test_clique(data_test_two,3,0.01)


# In[12]:

################################################ TEST GAUSSIENNES HORIZONTALES centre ############################################
nb_points =100     
data_test_two = dtp.create_gauss_cross(nb_points)
test_clique(data_test_two,4,0.04)

