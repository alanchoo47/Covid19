"""
Que tal profesor, este es el avance del programa Gaussian Naive Bayes
No esta completanmente comentado, pero tiene varias cosas a mejorar y cosas que no funcionan, las cuales
me gustaria implementar.
¿Me podría mandar notas las cuales usted piensa que debería mejorar? Como claridad, codigo sucio, etc

""" 
import requests
import zipfile36 as z

import numpy as np 
from sklearn import metrics	
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from numpy import genfromtxt
from sklearn.metrics import confusion_matrix, classification_report

from time import time

def covid_09(xFile):
	"""
	Por el momento el programa abre 3 veces el mismo archivo para guardarlo en un ndarray cada uno. 
	Meta> 	Guardar los valores independientes en un ndarray (data) 
			Guardar lso valores dependientes en un ndarray (target)
	"""
	f = open(xFile)
	f.readline()
	f1 = open(xFile)
	f1.readline()
	f2 = open(xFile)
	f2.readline()

	#Se insertan los datos del archivo f2, csv, sin el header y sin valores Nan
	dataset = genfromtxt(f, delimiter=',',dtype=None,skip_header = 0)
	#print(dataset[])
	
	#Se insertan los datos del archivo f, csv, sin valores Nan, sin header y de las columnas del 0 al 8 (independientes)
	#con el numero maximo de renglones = al numero de renglones del archivo csv
	data = genfromtxt(f1, delimiter=',',dtype=None,skip_header = 0, usecols = (0,1,2,3,4,5,6,7,8), max_rows=len(dataset))
	
	#Se insertan los datos del archivo f1, csv, sin valores Nan, sin header y de la antepenultima columna(dependientes)
	#con el numero maximo de renglones = al numero de renglones del archivo csv
	target = genfromtxt(f2, delimiter=',',dtype=None,skip_header = 0, usecols=(-1), max_rows=len(dataset))
	
	#Valores independientes que caracterizan la informacion de los pacientes
	#print(data)
	#Valores dependientes que caracterizan el resultado de los pacientes
	#print(target)

	
	#Funcion que divide aleatoriamente nuestros valores para tener diversidad en las pruebas y que no lleven a resultados no creibles
	X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.20,random_state = 43)
	#print(len(X_train))
	#print(len(X_test))
	#print(len(y_train))
	#print(len(y_test))

	#Se le asignan el metodo GaussiaNB a la variable gnb 
	gnb = GaussianNB()
	"""Se le asigna el metodo fit con variables de entrenamiento, las cuales se les aplica la funcion de prediccion con 
	respecto al conjunto de datos, previamente seleccionados, para hacer pruebas 
	funcion fit> 		eliminar datos repetidos, eliminar ruido
	funcion predict> 
	"""	
	y_pred = gnb.fit(X_train, y_train).predict(X_test)
	
	print("y_pred")
	print(y_pred)

	print("y_pred_len")
	print(len(y_pred))
	#Esta funcion sirve para saber el numero de predicciones que se hicieron incorrectamente con nuestra y_test
	#print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))

	#Esta parte del codigo es para saber la precisión que tenemos al categorizar correctamente nuestras predicciones con el resultado en y (y_test)
	contador = 0
	for i in range(len(X_test)):
		#print(gnb.predict(([(data[i])])),[target[i]])
		if (gnb.predict(([(X_test[i])])) == [y_test[i]]).all():
			#hacerlo sobre el test 
			contador = contador+1

	print("Contador:  %s"  % contador)	
	precision = int(contador)/ len(X_test)
	print("Precision:  %.5f"  % precision)		
	
 
	#Esta parte nos muestra una tabla que nos dice la precision de cada valor resultante
	print(classification_report(y_test,y_pred))
	#Esta parte nos muestra una matrix que nos dice cuantos valores predijeron correctamente y cuantos incorrectamente
	print(confusion_matrix(y_test,y_pred))

	#print(gnb.predict([[1,5,1,	5,	35,	1,	97,	49]]))
	#Se cierran los archivos .csv
	f.close()
	f1.close()
	f2.close()

def covid_20(xFile):
	"""
	Por el momento el programa abre 3 veces el mismo archivo para guardarlo en un ndarray cada uno. 
	Meta> 	Guardar los valores independientes en un ndarray (data) 
			Guardar lso valores dependientes en un ndarray (target)
	"""
	f = open(xFile)
	f.readline()
	f1 = open(xFile)
	f1.readline()
	f2 = open(xFile)
	f2.readline()

	#Se insertan los datos del archivo f2, csv, sin el header y sin valores Nan
	dataset = genfromtxt(f, delimiter=',',dtype=None,skip_header = 0)
	#print(dataset[])
	
	#Se insertan los datos del archivo f, csv, sin valores Nan, sin header y de las columnas del 0 al 18 (independientes)
	#con el numero maximo de renglones = al numero de renglones del archivo csv

	data = genfromtxt(f1, delimiter=',',dtype=None,skip_header = 0, usecols = (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18), max_rows=len(dataset))
	
	#Se insertan los datos del archivo f1, csv, sin valores Nan, sin header y de la antepenultima columna(dependientes)
	#con el numero maximo de renglones = al numero de renglones del archivo csv
	target = genfromtxt(f2, delimiter=',',dtype=None,skip_header = 0, usecols=(-1), max_rows=len(dataset))
	
	#Valores independientes que caracterizan la informacion de los pacientes
	#print(data)
	#Valores dependientes que caracterizan el resultado de los pacientes
	#print(target)

	#Funcion que divide aleatoriamente nuestros valores para tener diversidad en las pruebas y que no lleven a resultados no creibles
	X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.20,random_state = 43)
	#print(len(X_train))
	#print(len(X_test))
	#print(len(y_train))
	#print(len(y_test))

	#Se le asignan el metodo GaussiaNB a la variable gnb 
	gnb = GaussianNB()
	"""Se le asigna el metodo fit con variables de entrenamiento, las cuales se les aplica la funcion de prediccion con 
	respecto al conjunto de datos, previamente seleccionados, para hacer pruebas 
	funcion fit> 		eliminar datos repetidos, eliminar ruido
	funcion predict> 
	"""	
	y_pred = gnb.fit(X_train, y_train).predict(X_test)
	
	print("y_pred_len")
	print(len(y_pred))
	
	print("y_pred")
	print(y_pred)

	#Esta funcion sirve para saber el numero de predicciones que se hicieron incorrectamente con nuestra y_test
	#print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))

	#Esta parte del codigo es para saber la precisión que tenemos al categorizar correctamente nuestras predicciones con el resultado en y (y_test)
	contador = 0
	for i in range(len(X_test)):
		#print(gnb.predict(([(data[i])])),[target[i]])
		if (gnb.predict(([(X_test[i])])) == [y_test[i]]).all():
			#hacerlo sobre el test 
			contador = contador+1

	print("Contador:  %s"  % contador)	
	precision = int(contador)/ len(X_test)
	print("Precision:  %.5f"  % precision)		
	
 
	#Esta parte nos muestra una tabla que nos dice la precision de cada valor resultante
	print(classification_report(y_test,y_pred))
	#Esta parte nos muestra una matrix que nos dice cuantos valores predijeron correctamente y cuantos incorrectamente
	print(confusion_matrix(y_test,y_pred))

	#print(gnb.predict([[1,5,1,	5,	35,	1,	97,	49]]))
	#Se cierran los archivos .csv
	f.close()
	f1.close()
	f2.close()

	#return precision
	#idea de return> agregarlos a un diccionario y ver que BD tiene la mejor prediccion mediante un sort()

def main():


	url = 'http://datosabiertos.salud.gob.mx/gobmx/salud/datos_abiertos/datos_abiertos_covid19.zip'
	#url = 'https://www.python.org/static/img/python-logo@2x.png'

	myfile = requests.get(url)
	open('C:/Users/eact/OneDrive/Escritorio/Alan/ServicioSocial_IA/COVID19/BaseDatos/Actual.zip', 'wb').write(myfile.content)


	#Extracting a zipfile
	with z.ZipFile('Actual.zip', 'r') as my_zip:
		print(my_zip.namelist())
		s = my_zip.namelist()
		print(s)
		my_zip.extractall()					#extracting all

	#========================================


	print("BD: 19_700_200912_covid19.csv")
		#print(s)
	covid_20("19_700_200912_covid19.csv")
		#covid_20(s)

	#print("BD: 19_700_200912_covid19_intubado_si_no.csv")
	#covid_20("19_700_200912_covid19_intubado_si_no.csv")

	#print("BD: 19_FULL_201101_covid19_intubado_si_no.csv")
	#covid_20("19_FULL_201101_covid19_intubado_si_no.csv")

	
	"""

	

	print("BD: 09_10000_0912_covid19.csv")
	covid_09("09_10000_0912_covid19.csv")
	count_elapsed_time(covid_09)
	print("BD: 09_FULL_0912_covid19.csv")
	covid_09("09_FULL_0912_covid19.csv")

	print("BD: 09_10000_0912_covid19.csv")
	covid_09("09_10000_0912_covid19.csv")
	print("BD: 09_FULL_0912_covid19.csv")
	covid_09("09_FULL_0912_covid19.csv")

	print("BD: 09_10000_0912_covid19.csv")
	covid_09("09_10000_0912_covid19.csv")
	print("BD: 09_FULL_0912_covid19.csv")
	covid_09("09_FULL_0912_covid19.csv")

	print("BD: 09_10000_0912_covid19.csv")
	covid_09("09_10000_0912_covid19.csv")
	print("BD: 09_FULL_0912_covid19.csv")
	covid_09("09_FULL_0912_covid19.csv")

	print("BD: 15_10000_0912_covid19.csv")
	covid_20("15_10000_0912_covid19.csv")
	print("BD: 15_10000_0912_covid19.csv")
	covid_20("15_10000_0912_covid19.csv")
	print("BD: 15_10000_0912_covid19.csv")
	covid_20("15_10000_0912_covid19.csv")
	print("BD: 15_10000_0912_covid19.csv")
	covid_20("15_10000_0912_covid19.csv")

	print(count_elapsed_time(covid_20))
	
	"""
	#print("BD: 15_10000_0912_covid19.csv")
	#covid_20("15_10000_0912_covid19.csv")
	
	#print("BD: 15_FULL_0912_covid19.csv")
	#covid_20("15_FULL_0912_covid19.csv")



main()

"""Verificar la precision del algoritmo
probarlo en la ultima base de datos (ayer) y probar con un 75/25()

contador para ver cuantos coinciden 

registros de distintos dias para ver el comportamiento del algoritmo
ya que, podemos ver el desempeño del algoritmo con respecto al tiempo

si probamos con los de mañana (mejora, empeora o nada)
si mejora es más completa y precisa, se quita la incertidumbre

¿Bayes es bueno para estos datos?

nos interesa la prediccion __ a cuantos casos les atinamos

programar o utilizar por libreia

**precision de manera manual_

--agregar columnas --- la precision aumenta
**Hacer más cosas de forma manual para tener más control 

**De momento con las columnas que tengo 

**Generar un documento para el historial, 

**Investigar que piden para el servicio social (documentos, reportes, etc.)
	**reporte final
			
*Entrenar con el 90/10__mejora

probar cada vez mas datos para ver que tan precisos podemos ser

**Lunes a la 10am

precision de columnas 1, checar en el de 15_100... csv


"""

#IDEAS
#INTERFAZ QUE permita la selección de columnas independientes y/o dependientes
#Interfaz que permita la entrada de datos a diferentes columnas o que seleccione una columna aleatoria 
#Interfaz que divida entre hombre,mujeres, otros y así nos de otras probabilidades
#Meter tiempo de ejecución
#Contar avance de datos que se van creando (avance actual- avance pasado)

#observar cno una base de dtaso pequeña mi test_split para ver que esta haciendo

