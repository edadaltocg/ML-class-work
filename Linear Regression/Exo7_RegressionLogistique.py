
"""
EXERCICE 7 : Regression logistique
"""

import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

#QUESTION) Il semble qu'un traitement (colonne Treatment) ait un effet sur une variable de "./RealMedicalData2.csv".
#          Utilisez 'sklearn.linear_model.LogisticRegression' pour trouver cette variable.





dataframe=pandas.read_csv("./RealMedicalData2.csv",sep=';',decimal=b',')

listColNames=list(dataframe.columns)


#1.2) extract X and Y as numpy arrays

XY=dataframe.values
ColNb_Y=listColNames.index('Treatment')


Y=XY[:,ColNb_Y].reshape((XY.shape[0],1))   #reshape is to make sure that Y is a column vector
X = np.delete(XY, ColNb_Y, 1)

X_scaled = preprocessing.scale(X)

listColNames.pop(ColNb_Y)     #to make it contains the column names of X only

#2) EXPLORE THE DATA

for Col in range(len(listColNames)):
  plt.plot(X[:,Col],Y[:],'.')
  plt.xlabel(listColNames[Col])
  plt.ylabel('Disease progression')
  plt.show()

#-> 'Biomarker 3' ressort clairement dans les figures

#3) PERFORM THE REGRESSION

from sklearn.linear_model import LogisticRegression


regressor=LogisticRegression(penalty='l1', C=10.0)

regressor.fit(X_scaled,Y)

print('Beta values')
for Col in range(len(listColNames)):
  print('-> '+listColNames[Col]+': '+str(regressor.coef_[0,Col]))

#-> 'Biomarker 3' ressort clairement avec la regression logistique