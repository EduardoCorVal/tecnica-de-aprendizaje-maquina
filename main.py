"""
Archivo principal

Aquí se ejecuta el programa principal. Se importa el archivo 
'decision_tree_classifier.py' para poder utilizar la clase
"""

__author__ = "Eduardo Joel Cortez Valente"

import numpy as np
import pandas as pd
import random as rand
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from decision_tree_classifier import DecisionTreeClassifier


def precisionGrafica(val_scores: list, test_scores: list, dataset_name: str):
    plt.figure(figsize=(10, 6)) 
    plt.plot(range(1, 11), val_scores, marker='o', label='Validation Score')
    plt.plot(range(1, 11), test_scores, marker='o', label='Test Score')
    plt.title(f'Validation and Test Scores for {dataset_name}')
    plt.xlabel('Case')
    plt.ylabel('Accuracy Score')
    plt.xticks(range(1, 11))
    plt.legend()
    plt.tight_layout()
    # Guardar la figura en un archivo de imagen
    # plt.savefig(f'{dataset_name}_scores_graph.png')
    plt.show()
    
def conf_matrix(all_confusion_matrices: list):
    '''Función para graficar las matrices de confusión'''
    
    # Convertir la lista en un array NumPy
    all_confusion_matrices = np.array(all_confusion_matrices)

    # Calcular el número de filas y columnas necesarias para las subtramas
    num_rows = 3
    num_cols = (all_confusion_matrices.shape[0] + num_rows - 1) // num_rows

    # Crear una figura con subplots para cada matriz de confusión
    plt.figure(figsize=(12, 8))

    for i in range(all_confusion_matrices.shape[0]):
        plt.subplot(num_rows, num_cols, i + 1)  # Organizar las subtramas en filas y columnas
        plt.imshow(all_confusion_matrices[i], interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - Validation - Case {i + 1}')
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['Class 0', 'Class 1'])
        plt.yticks(tick_marks, ['Class 0', 'Class 1'])
        plt.xlabel('Predicted')
        plt.ylabel('True')

    plt.tight_layout()  # Ajustar el diseño de los subplots
    # Guardar la figura en un archivo de imagen
    # plt.savefig(f'{dataset_name}_confusion_matrix.png')
    plt.show()
    
def recallGrafica(recall_scores: list, dataset_name: str):
    '''Función para graficar los resultados de recall'''
    
    plt.figure(figsize=(10, 6)) 
    plt.plot(range(1, 11), recall_scores, marker='o', label='Recall Score')
    plt.title(f'Recall Score for {dataset_name}')
    plt.xlabel('Case')
    plt.ylabel('Recall Score')
    plt.xticks(range(1, 11))
    plt.legend()
    plt.tight_layout()
    # Guardar la figura en un archivo de imagen
    # plt.savefig(f'{dataset_name}_recall_graph.png')
    plt.show()

def conf_modelo(data, samples_split, depth=2, flag='csv', dataset_name="Not specified"):
    '''Función para obtener los resultados de la validación cruzada'''
    
    # Asteriscos de separacion
    # asterisks = ''.join(['*' for _ in range(80)])

    # Obtener los datos
    # print(f"La información a utlizar sera: \n")
    # print(data)
    # print(f"{asterisks}\n")
    validation_scores = []
    test_scores = []
    all_confusion_matrices = []
    recall_scores = []

    # Split del test de entrenamiento
    if flag == 'csv':
        X = data.iloc[:, :-1].values
        Y = data.iloc[:, -1].values.reshape(-1,1)

    # División original en entrenamiento, validación y prueba
    X_train_initial, X_test, Y_train_initial, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    # Nueva división en entrenamiento y validación
    for x in range(1, 11):
        # print(f"Para el caso {x}:\n")
        # rand.seed(x)
        random_state = rand.randint(1, 100)
        
        X_train, X_validation, Y_train, Y_validation = train_test_split(X_train_initial, Y_train_initial, test_size=0.2, random_state=random_state)

        classifier = DecisionTreeClassifier(min_samples_split=samples_split, max_depth=depth)
        classifier.fit(X_train, Y_train)

        Y_pred_validation = classifier.prediccion(X_validation)
        score_validation = accuracy_score(Y_validation, Y_pred_validation)
        # print(f"El resultado del accuracy en el conjunto de validación es de: {score_validation}\n")
        # print(f"{asterisks}\n")
        validation_scores.append(score_validation)

        # Ajustar el modelo nuevamente con el conjunto de entrenamiento completo
        classifier.fit(X_train_initial, Y_train_initial)

        Y_pred = classifier.prediccion(X_test)
        score = accuracy_score(Y_test, Y_pred)
        # print(f"El resultado del accuracy en el conjunto de prueba es de: {score}\n")
        # print(f"{asterisks}\n")
        test_scores.append(score)
        
        cm_validation = confusion_matrix(Y_validation, Y_pred_validation)
        all_confusion_matrices.append(cm_validation)
        
        recall = recall_score(Y_validation, Y_pred_validation, average='macro')
        recall_scores.append(recall)

    # Graficar los resultados  
    precisionGrafica(validation_scores, test_scores, dataset_name)
    
    # Mostrar las matrices de confusión
    conf_matrix(all_confusion_matrices)
    
    # Graficar los resultados de recall
    recallGrafica(recall_scores, dataset_name)
    

if __name__ == "__main__":
    # Probando con 'iris.csv'
    data_1 = pd.read_csv("iris.csv")
    conf_modelo(data_1, samples_split=3, depth=3, dataset_name="iris.csv")

    # Probando con 'digits.csv'
    # data_2 = pd.read_csv("digits.csv")
    # conf_modelo(data_2, samples_split=2, depth=8, dataset_name="digits.csv")