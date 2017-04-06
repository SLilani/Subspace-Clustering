

# coding: utf-8

# In[3]:

import dataplayer as dtp
import clique as clique
import Evaluateur_inertie as ei
import outils as tools

# In[4]:

# Fonction de teste 
# Prend en paramétre base : Dataframe, nb_intervalles : nombre d'intervalles pour la discrétisation des dimensions
# et le taux : pour la densité des cellules

def test_clique(base,nb_intervalles,taux) :
    clique_test = clique.clique(base,nb_intervalles,taux)
    clique_test.run()
    clique_test.affichage_clique()
    aff,centro = clique_test.output_for_eval_inertie()
    e = ei.Evaluateur_inertie(base,aff,centro)
    e.Evaluate(tools.euclidienne)
    e.__str__()
    

# In[5]:

##########################################################################################################################
################################################### TESTS CLIQUE #########################################################
##########################################################################################################################


# In[6]:

################################################ TEST XOR ################################################################
nb_points = 10
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
test_clique(data_test_two,3,0.1)


# In[9]:

################################################ TEST GAUSSIENNES HORIZONTALES centre ############################################
nb_points =100     
data_test_two = dtp.create_gauss_horizontal_cent(nb_points)
test_clique(data_test_two,3,0.01)


# In[10]:

################################################ TEST GAUSSIENNES HORIZONTALES centre ############################################
nb_points =100     
data_test_two = dtp.create_gauss_vertical_cent(nb_points)
test_clique(data_test_two,3,0.01)


# In[12]:

################################################ TEST GAUSSIENNES HORIZONTALES centre ############################################
nb_points =100     
data_test_two = dtp.create_gauss_cross(nb_points)
test_clique(data_test_two,10,0.001)
    