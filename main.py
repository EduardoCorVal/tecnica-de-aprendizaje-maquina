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

    # Se dividen los datos en conjuntos de: 
    #     * Entrenamiento (X_train, Y_train)
    #     * Prueba (X_test, Y_Test)
    # Se reserva el 20% de los datos para pruebas. Utilizo un random state de 13 para garantizar la reproducibilidad
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=13)

    # Entrenamiento del modelo
    classifier = DecisionTreeClassifier(min_samples_split=samples_split, max_depth=depth)
    classifier.fit(X_train,Y_train)

    # Predicción del modelo
    Y_pred = classifier.prediccion(X_test)

    # Evaluación del modelo
    score = accuracy_score(Y_test, Y_pred)
    print(f"El resultado del accuracy del modelo generado es de: {score}\n")
    print(f"{asterisks}\n")


# Probando con 'iris.csv'
data_1 = pd.read_csv("iris.csv")
conf_modelo(data_1, samples_split=3, depth=3)

# Probando con 'digits.csv'
data_2 = pd.read_csv("digits.csv")
conf_modelo(data_2, samples_split=2, depth=8)