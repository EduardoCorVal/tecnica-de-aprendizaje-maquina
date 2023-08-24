import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from decision_tree import DecisionTreeClassifier

# Obtener la data
data = pd.read_csv("iris.csv")
print(data)

# Split del test de entrenamiento
X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values.reshape(-1,1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=41)

# Fit the model
classifier = DecisionTreeClassifier(min_samples_split=3, max_depth=3)
classifier.fit(X_train,Y_train)

# Probar el modelo
Y_pred = classifier.prediccion(X_test) 
score = accuracy_score(Y_test, Y_pred)
print(score)
