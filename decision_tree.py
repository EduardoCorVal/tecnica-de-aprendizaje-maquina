"""
Decision Tree Classifier
"""

import numpy as np
import pandas as pd

class Node():
    """Representa un nodo de decisión en un árbol de decisión"""
    
    def __init__(self, feature_index = None, threshold = None, left = None, right = None, info_gain = None, value = None):
        '''Constructor de la clase'''
        
        # Para el decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        
        # Para el leaf node
        self.value = value
        
class DecisionTreeClassifier():
    """Es el arbol de decisión"""
    def __init__(self, min_samples_split = 2, max_depth = 2):
        '''Constructor de la clase'''
        
        # Inicialización de la raiz del arbol
        self.root = None
        pass