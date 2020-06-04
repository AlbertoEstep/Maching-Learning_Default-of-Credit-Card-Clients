# -*- coding: utf-8 -*-

# AA - Práctica 1 - UGR
# Authors: Alberto Estepa Fernández & Carlos Santiago Sánchez Muñoz
# Date: 24/06/2020

# Importamos las librerías necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def valores_perdidos(df):
    total = df.isnull().sum().sort_values(ascending = False)
    percent = (df.isnull().sum() / df.isnull().count()).sort_values(ascending = False)
    missing_data = pd.concat([total, percent], axis = 1, keys = ['Total', 'Porcentaje'])
    return missing_data

def estudio(df):
    print(df.head(5))
    # Estadísticas de los datos leídos
    print("El número de instancias del dataset: {}".format(df.shape[0]))
    print("El número de atributos del dataset: {}".format(df.shape[1]))
    # Valores perdidos
    missing_data = valores_perdidos(df)
    print('Valores perdidos:\n')
    print(missing_data[missing_data['Total'] > 0])

print("--------------------------------------------------------------\n")
print("----------              CLASIFICACIÓN              -----------\n")
print("--------------------------------------------------------------\n")

print("Leyendo los datos...", end=" ", flush=True)
df = pd.read_excel('datos/default of credit card clients.xls',
                   skiprows = 1, index_col = 'ID')
print("Lectura completada.\n")
print("El número de atributos actual es de: {}".format(df.shape[1]))

def graf_bar(df, header, size):
    col = np.array(df.iloc[:, df.columns.to_list().index(header)])
    min = np.amin(col)
    max = np.amax(col)
    long = (max - min)/size
    intervals = []
    num = []
    for i in range(0,size):
        num.append(len(col[(min + i*long <= col) & (col < min + (i+1)*long)]))
        intervals.append("[{},{})".format(int(min + i*long), int(min + (i+1)*long)))
    num[-1] += len(col[col==max])
    intervals[-1] = "[{},{}]".format(round(max-long,1), max)

    print("Mostrando gráfica de barras asociada...")
    plt.bar(intervals, num, align="center")
    plt.xlabel("Intervalo")
    plt.xticks(rotation=30)
    plt.ylabel("Núm. instancias")
    plt.title("Gráfica de barras de " + header)
    plt.gcf().canvas.set_window_title("Proyecto AA")
    plt.show()

graf_bar(df, 'BILL_AMT1', 10)

input("\n----------- Pulse 'Enter' para continuar --------------\n\n\n")

print("Estudiamos la variable EDUCATION\n")
for i in df['EDUCATION'].unique():
    print("\t {} instancias del valor {}".format(df['EDUCATION'].value_counts()[i],i))

input("\n----------- Pulse 'Enter' para continuar --------------\n\n\n")

print("Tratamos los outliers de EDUCATION\n")
df.replace({'EDUCATION': {0: 4, 5: 4, 6: 4}}, inplace = True)
for i in df['EDUCATION'].unique():
    print("\t {} instancias del valor {}".format(df['EDUCATION'].value_counts()[i],i))

input("\n----------- Pulse 'Enter' para continuar --------------\n\n\n")

print("Estudiamos la variable MARRIAGE\n")
for i in df['MARRIAGE'].unique():
    print("\t {} instancias del valor {}".format(df['MARRIAGE'].value_counts()[i],i))

input("\n----------- Pulse 'Enter' para continuar --------------\n\n\n")

print("Tratamos los outliers de MARRIAGE\n")
df.replace({'MARRIAGE': {0: 3}}, inplace = True)
for i in df['MARRIAGE'].unique():
    print("\t {} instancias del valor {}".format(df['MARRIAGE'].value_counts()[i],i))

input("\n----------- Pulse 'Enter' para continuar --------------\n\n\n")

print("Binarizamos las variables necesarias\n")
df = pd.get_dummies(df, columns=['SEX', 'EDUCATION', 'MARRIAGE'])
df.rename(columns={'SEX_1':'Hombre',
                    'SEX_2':'Mujer',
                    'EDUCATION_1':'Educacion_postgrado',
                    'EDUCATION_2':'Educacion_universidad',
                    'EDUCATION_3':'Educacion_secundaria',
                    'EDUCATION_4':'Educacion_otros',
                    'MARRIAGE_1':'Casado',
                    'MARRIAGE_2':'Soltero',
                    'MARRIAGE_3':'Otro_estado_civil'},
               inplace=True)
df = pd.get_dummies(df, columns=['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6'])
print("El número de atributos actual es de: {}".format(df.shape[1]))

input("\n----------- Pulse 'Enter' para continuar --------------\n\n\n")

# Balanceo de las clases
for i in range (len(df['default payment next month'].unique())):
    total = df['default payment next month'].value_counts()[i]
    porcentaje = 100*total / df.shape[0]
    print("\t {} instancias de la clase {}, es decir, un {}% del total".format(total,i, porcentaje))

input("\n----------- Pulse 'Enter' para continuar --------------\n\n\n")

print("Matriz de correlacion\n")

def matriz_correlaciones(datos):
    f, ax = plt.subplots(figsize=(10, 8))
    corr = datos.corr(method = 'pearson')
    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
               square=True, ax=ax, cbar_kws={'label': 'Coeficiente Pearson'})
    f.suptitle('Matriz Correlaciones')
    plt.show()

matriz_correlaciones(df)

input("\n----------- Pulse 'Enter' para continuar --------------\n\n\n")

print("Matriz de correlacion de variables multicolineales\n")

# Seleccionamos las columnas que son multicolineales
def columnas_correladas(dataframe, coeficiente = 1):
    corr_matrix = dataframe.corr(method = 'pearson').abs() # Matriz de correlación en valor absoluto

    # Seleccionamos la matriz triangular superior de la matriz de correlación anterior
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Buscamos la columnas con una correlacion de 1 con alguna otra (cuando se da multicolinealidad)
    to_drop = [column for column in upper.columns if any(upper[column] >= coeficiente)]
    return to_drop
to_drop = columnas_correladas(dataframe = df, coeficiente = 1)

# Matriz de correlación en valor absoluto ampliada por columnas
def matriz_correlacion_ampliada(dataframe, columnas, k = 3):
# k: Número de variables a mostrar.
    # Matriz de correlación en valor absoluto
    corr_matrix = dataframe.corr(method = 'pearson').abs()
    for i in columnas:
        cols = corr_matrix.nlargest(k, i)[i].index
        cm = np.abs(np.corrcoef(dataframe[cols].values.T))
        ax = plt.axes()
        sns.set(font_scale = 1.25)
        hm = sns.heatmap(cm, cbar = True, annot = True, square = True,
                             fmt = '.2f', annot_kws = {'size': 10},
                             yticklabels = cols.values,
                             xticklabels = cols.values,
                             cbar_kws={'label': 'Coeficiente Pearson \
                                             en valor absoluto'})
        ax.set_title('M. de corr. ampliada de {}'.format(i))
        plt.show()

matriz_correlacion_ampliada(dataframe = df, k = 3, columnas = to_drop)

# Eliminamos la variable Hombre
df.drop(['Hombre'], axis=1, inplace = True)

input("\n----------- Pulse 'Enter' para continuar --------------\n\n\n")

print("El número de atributos actual es de: {}".format(df.shape[1]))

input("\n----------- Pulse 'Enter' para continuar --------------\n\n\n")

print("Separacion en train y test\n")

# Separamos el dataset original en atributos y etiquetas:
X = df.iloc[:,:-1] # Todas las columnas menos la última
y = df.iloc[:, -1] # Última columna

# Dividimos los conjuntos en test (20 %) y train (80 %)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

# Estadísticas de los datos leídos
n_train = X_train.shape[0]
n_test = X_test.shape[0]
porcentaje_train = 100*n_train/(n_test+n_train)
porcentaje_test = 100*n_test/(n_test+n_train)
print("El número de instancias de entrenamiento es de: {}".format(X_train.shape[0]))
print("El número de instancias de test es de: {}".format(X_test.shape[0]))
print("Porcentaje de train: {} y porcentaje de test: {}".format(
    porcentaje_train, porcentaje_test))

input("\n----------- Pulse 'Enter' para continuar --------------\n\n\n")

print("Preparamos el escalado de los datos\n")

preprocesado = [("escalado", MinMaxScaler())]

preprocesador = Pipeline(preprocesado)

datos_preprocesados = preprocesador.fit_transform(X_train)

input("\n----------- Pulse 'Enter' para continuar --------------\n\n\n")

print("Entrenando el modelo lineal.", end=" ", flush=True)
log = LogisticRegression(penalty='l2', # Regularización Ridge (L2)
                                    solver = 'lbfgs', # Algoritmo a utilizar en el problema de optimización, aunque es
                                                            # el dado por defecto
                                    max_iter = 1000)

log.fit(datos_preprocesados, y_train)
print(log.score(datos_preprocesados, y_train))
log.predict(X_test)
print(log.score(X_test, y_test))
print("Entrenamiento completado\n")

input("\n----------- Pulse 'Enter' para continuar --------------\n\n\n")

print("Entrenando el modelo RF.", end=" ", flush=True)
rf = RandomForestClassifier()

rf.fit(datos_preprocesados, y_train)
print(rf.score(datos_preprocesados, y_train))
rf.predict(X_test)
print(rf.score(X_test, y_test))
print("Entrenamiento completado\n")

input("\n----------- Pulse 'Enter' para continuar --------------\n\n\n")

print("Entrenando el modelo SVM.", end=" ", flush=True)
svc = SVC()
svc.fit(datos_preprocesados, y_train)
print(svc.score(datos_preprocesados, y_train))
svc.predict(X_test)
print(svc.score(X_test, y_test))
print("Entrenamiento completado\n")
