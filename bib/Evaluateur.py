# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 19:45:47 2017

@author: MDK
"""

class Evaluateur :
     
    def __init__(self,base) :
        """ base : DataFrame d'exemples
        """
        self.base = base
        self.results = {}
       
    def Evaluate(self,results)  :
         self.results = results
     
    def set_results(self,results) :
        self.results = results
    
    def store_results(self,filename) :
        raise NotImplementedError("Please Implement this method")
    
    def __str__(self) :
        raise NotImplementedError("Please Implement this method")