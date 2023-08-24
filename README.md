# tecnica-de-aprendizaje-maquina

**Alumno**: Eduardo Joel Cortez Valente

**Matricula**: A01746664

**Materia**: Inteligencia artificial avanzada para la ciencia de datos

**Clave de la materia**: TC3006C

**Fecha**: 28/08/2023

**Entregable**: Momento de Retroalimentación: Módulo 2 Implementación de una técnica de aprendizaje máquina sin el uso de un framework

## Set up

Para correr esta aplicación es necesario tener instaladas las librerias:
* pandas
* sklearn
* numpy

En caso de no querer instalar dichas librerias de forma manual, se puede descargar el entorno de Anaconda y correr el entorno virtual de conda.

## Arquitectura

La aplicación se divide en dos partes. Una es el archivo __decision_tree_classifier.py__ donde se encuentra la implementación del algoritmo de un árbol de decisión para clasificación; en forma de clase.

Dicha clase es usada en el archivo __main.py__ donde, en la función *conf_modelo(data)*, se organiza la data de tal manera que sea mas facil proporcionar los datos y visualizar el resultado.

### Nota‼️
Está función solo sirve con set de datos previamente limpiados, y donde la variable a predecir (tomar como Y dentro del modelo) se encuentra en la ultima columna; tomando los demas datos como los parametros X

## Instrucciones de uso
Se añaden los datos en forma de CSV en la parte inferior del programa. Los ejemplos tomados para la prueba y ejecución del presente proyecto son *digits.csv* y *iris.csv*

Para ejecutar el programa corre el siguiente comando en el directorio raíz:
```shell
python main.py
```