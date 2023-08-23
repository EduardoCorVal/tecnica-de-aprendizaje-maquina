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
        
        # Condiciones de parada
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        
    def construir_arbol(self, dataset, curr_depth = 0):
        ''' Función recursiva para construir el arbol '''
        
        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X) # (filas, columnas)
        
        # Dividir hasta qye se cumplan las condicoones
        if curr_depth <= self.max_depth and num_samples >= self.min_samples_split:
            # Encontrar el mejor split
            # best_split = self.get
            pass
        pass
            
    def split(self, dataset, feature_index, threshold):
        ''' Función para dividir la data '''
        
        dataset_left = np.array([row for row in dataset if row[feature_index] <= threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index] > threshold])
        return dataset_left, dataset_right
    
    def calculo_ganancia(self, parent, l_child, r_child):
        ''' Función para calcular la ganancia de información '''
        
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        ganacia = self.entropia(parent) - (weight_l*self.entropia(l_child) + weight_r*self.entropia(r_child))
        return ganacia  
            
    def entropia(self, y):
        ''' Función para calcular la entropia '''
        
        class_labels = np.unique(y)
        entropia = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropia += -p_cls * np.log2(p_cls)
        return entropia
            
    def calcular_leaft_value(self, in_y):
        ''' Función para calcular el leaf node '''
        
        in_y = list(in_y)
        return max(in_y, key=in_y.count)
    
    def prediccion(self, in_x):
        ''' Función para predecir un nuevo data set '''
        
        predicciones = [self.hacer_prediccion(x, self.root) for x in in_x]
        return predicciones
    
    def hacer_prediccion(self, x, tree):
        ''' Funcion para predecir un solo data point '''
        
        if tree.value != None:
            return tree.value
        
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self.hacer_prediccion(x, tree.left)
        else:
            return self.hacer_prediccion(x, tree.right)