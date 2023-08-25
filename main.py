"""
Archivo principal

Aquí se ejecuta el programa principal. Se importa el archivo 
'decision_tree_classifier.py' para poder utilizar la clase
"""

__author__ = "Eduardo Joel Cortez Valente"

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from decision_tree_classifier import DecisionTreeClassifier

def conf_modelo(data, samples_split, depth=2, flag='csv'):
    # Asteriscos de separacion
    asterisks = ''.join(['*' for _ in range(70)])

    # Obtener los datos
    print(f"La información a utlizar sera: \n")
    print(data)
    print(f"{asterisks}\n")

    # Split del test de entrenamiento
    if flag == 'csv':
        X = data.iloc[:, :-1].values
        Y = data.iloc[:, -1].values.reshape(-1,1)

    # División original en entrenamiento, validación y prueba
    X_train_initial, X_test, Y_train_initial, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    # Nueva división en entrenamiento y validación
    X_train, X_validation, Y_train, Y_validation = train_test_split(X_train_initial, Y_train_initial, test_size=0.2, random_state=13)

    classifier = DecisionTreeClassifier(min_samples_split=samples_split, max_depth=depth)
    classifier.fit(X_train, Y_train)

    Y_pred_validation = classifier.prediccion(X_validation)
    score_validation = accuracy_score(Y_validation, Y_pred_validation)
    print(f"El resultado del accuracy en el conjunto de validación es de: {score_validation}\n")
    print(f"{asterisks}\n")

    # Ajustar el modelo nuevamente con el conjunto de entrenamiento completo
    classifier.fit(X_train_initial, Y_train_initial)

    Y_pred = classifier.prediccion(X_test)
    score = accuracy_score(Y_test, Y_pred)
    print(f"El resultado del accuracy en el conjunto de prueba es de: {score}\n")
    print(f"{asterisks}\n")


# Probando con 'iris.csv'
data_1 = pd.read_csv("iris.csv")
conf_modelo(data_1, samples_split=3, depth=3)

# Probando con 'digits.csv'
data_2 = pd.read_csv("digits.csv")
conf_modelo(data_2, samples_split=2, depth=8)