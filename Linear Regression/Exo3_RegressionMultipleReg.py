"""
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
EXERCICE 3 : REGRESSION LINEAIRE MULTIPLE AVEC REGULARISATION + VALIDATION CROISEE


Inspire de http://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_and_elasticnet.html
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score

# generation de donnee synthetiques...
np.random.seed(31)

#... definition de n et p
n = 75
p =  200   #remarque : n<200 => necessite de selectionner des variables

#... simulation de X
X = np.random.randn(n, p) #remarque : on ne tient pas en compte les $beta_0$

#... generation du vecteur beta dans lequel seules 10 valeurs sont non-nulles
beta = 3 * np.random.randn(p)
inds = np.arange(p)
np.random.shuffle(inds)
beta[inds[10:]] = 0 

#... simulation de y 
y = np.dot(X, beta) + (0.01 * np.random.normal(size=n))
# REMARQUE IMPORTANTE : y ne dÃ©pend que des variables i pour lesquelles beta[i] est non-nul

# ... coupe en deux les donnees en donnees d'apprentissage et donnes test
thresh=n // 2
X_train = X[thresh:]
y_train = y[thresh:]
X_test = X[:thresh]
y_test = y[:thresh]

# regression lineaire avec regularisation Lasso ...

#... regression
from sklearn.linear_model import Lasso

alpha = 0.5
lasso_regressor = Lasso(alpha=alpha)
lasso_regressor.fit(X_train, y_train)

y_pred_lasso = lasso_regressor.predict(X_test)
r2_score_lasso = r2_score(y_test, y_pred_lasso)

#... representation du resultat

fig = plt.figure()
plt.plot(y_test, y_pred_lasso, 'r.')
plt.xlabel('True value')
plt.ylabel('Predicted value')
plt.title('True vs predicted value (r2='+str(r2_score_lasso)+')')
plt.show()



fig = plt.figure()
plt.plot(lasso_regressor.coef_, 'r.')
plt.plot(beta, 'b.')
plt.legend(('Beta estime', 'Beta reel'), loc='lower right')
plt.title('Coefficients de beta')
plt.show()


#QUESTION 1:
#Jouez l'exemple avec n=100, n=75, n=50, n=25. Qu'en deduisez vous sur l'impact du nombre d'observations

#-> en dessous de 75 observations, on a du mal a avoir de bonnes predictions et estimer les bon beta. 

#QUESTION 2:
#On garde n=75. Utiliser la validation croisee de type K-folds pour trouver le meilleur coefficient alpha
#au sens de R2.
#AIDE : Pour comprendre l'utilisation de K-folds sous sklearn vous pouvez jouer l'exemple ci-dessous

from sklearn.model_selection import KFold

#data = np.random.randn(12,3)
#kf = KFold(n_splits=3)
#
#for train, test in kf.split(data):
#  print("+++++++++++++++++++++++++++++++++++++++")
#  print('train='+str(train)+':')
#  print(str(data[train,:]))
#  print('test='+str(test)+':')
#  print(str(data[test,:]))
#  print("+++++++++++++++++++++++++++++++++++++++")

#-> 1ere passe : estimtion de l'echelle du alpha optimal

kf = KFold(n_splits=8)
for alpha in [0.001,0.01,0.1,1.,10.]:
  sum_r2_scores=0.
  for train, test in kf.split(X):
    X_train=X[train]
    y_train=y[train]
    X_test=X[test]
    y_test=y[test]
  
    lasso_regressor = Lasso(alpha=alpha)
    lasso_regressor.fit(X_train, y_train)
  
    y_pred_lasso = lasso_regressor.predict(X_test)
    r2_score_lasso = r2_score(y_test, y_pred_lasso)
    #print(alpha,": ",r2_score_lasso)
    sum_r2_scores+=r2_score_lasso
  print(alpha," total: ",sum_r2_scores)

#-> meilleur score obtenu pour alpha=0.01

#-> 2eme passe : estimation plus fine du alpha optimal

kf = KFold(n_splits=8)
for alpha in [0.0025,0.005,0.01,0.02,0.04,0.08]:
  alpha=0.0025
  sum_r2_scores=0.
  for train, test in kf.split(X):
    X_train=X[train]
    y_train=y[train]
    X_test=X[test]
    y_test=y[test]
  
    lasso_regressor = Lasso(alpha=alpha)
    lasso_regressor.fit(X_train, y_train)
  
    y_pred_lasso = lasso_regressor.predict(X_test)
    r2_score_lasso = r2_score(y_test, y_pred_lasso)
    sum_r2_scores+=r2_score_lasso
  print(alpha," total: ",sum_r2_scores)

#-> meilleur score obtenu pour alpha=0.0025
#On pourra remarquer que l'on a pas suffisement regularise le probleme avec le LASSO 
#pour que les beta aient une interpretation claire. Les rÃ©sultats auront un R2 legerement
#inferieur mais tout a fait resonable avec par exemple alpha=0.5, et on retrouvera clairement
#les beta recherches.





#QUESTION 3:
#EVENTUELLEMENT : Comparez les resultats LASSO avec ceux d'ElasticNet
#Utilisez : from sklearn.linear_model import ElasticNet
#    puis : enet_regressor = ElasticNet(alpha=alpha, l1_ratio=0.7)

from sklearn.linear_model import ElasticNet

thresh=n // 2
X_train = X[thresh:]
y_train = y[thresh:]
X_test = X[:thresh]
y_test = y[:thresh]

alpha = 0.5

#++++

enet_regressor = ElasticNet(alpha=alpha, l1_ratio=0.1)
enet_regressor.fit(X_train, y_train)
y_pred_enet = enet_regressor.predict(X_test)
r2_score_enet = r2_score(y_test, y_pred_enet)

fig = plt.figure()
plt.plot(enet_regressor.coef_, 'r.')
plt.plot(beta, 'b.')
plt.legend(('Beta estime', 'Beta reel'), loc='lower right')
plt.title('Coefficients de beta (r2='+str(r2_score_enet)+')')
plt.show()
#++++

enet_regressor = ElasticNet(alpha=alpha, l1_ratio=0.25)
enet_regressor.fit(X_train, y_train)
y_pred_enet = enet_regressor.predict(X_test)
r2_score_enet = r2_score(y_test, y_pred_enet)

fig = plt.figure()
plt.plot(enet_regressor.coef_, 'r.')
plt.plot(beta, 'b.')
plt.legend(('Beta estime', 'Beta reel'), loc='lower right')
plt.title('Coefficients de beta (r2='+str(r2_score_enet)+')')
plt.show()

#++++

enet_regressor = ElasticNet(alpha=alpha, l1_ratio=0.5)
enet_regressor.fit(X_train, y_train)
y_pred_enet = enet_regressor.predict(X_test)
r2_score_enet = r2_score(y_test, y_pred_enet)

fig = plt.figure()
plt.plot(enet_regressor.coef_, 'r.')
plt.plot(beta, 'b.')
plt.legend(('Beta estime', 'Beta reel'), loc='lower right')
plt.title('Coefficients de beta (r2='+str(r2_score_enet)+')')
plt.show()


#++++

enet_regressor = ElasticNet(alpha=alpha, l1_ratio=0.75)
enet_regressor.fit(X_train, y_train)
y_pred_enet = enet_regressor.predict(X_test)
r2_score_enet = r2_score(y_test, y_pred_enet)

fig = plt.figure()
plt.plot(enet_regressor.coef_, 'r.')
plt.plot(beta, 'b.')
plt.legend(('Beta estime', 'Beta reel'), loc='lower right')
plt.title('Coefficients de beta (r2='+str(r2_score_enet)+')')
plt.show()




#++++

enet_regressor = ElasticNet(alpha=alpha, l1_ratio=0.99)
enet_regressor.fit(X_train, y_train)
y_pred_enet = enet_regressor.predict(X_test)
r2_score_enet = r2_score(y_test, y_pred_enet)

fig = plt.figure()
plt.plot(enet_regressor.coef_, 'r.')
plt.plot(beta, 'b.')
plt.legend(('Beta estime', 'Beta reel'), loc='lower right')
plt.title('Coefficients de beta (r2='+str(r2_score_enet)+')')
plt.show()


#-> Il est interessant de se souvenir que seules 10 valeurs de beta
#   ont ete utilisees pour simuler les donnees. On remarque alors que
#   les seuls resultats corrects sont ceux dans lesquels la regularisation
#   a force d'obtenir une selection de modele avec des beta>0  
#   parcimonieux (sparse), c'est a dire avec un poids fort sur la 
#   regularisation L1 d'elastic net et non la regularisation L2. En 
#   general, et surtout lorsque les observation sont bruitees ou peu
#   nombreuses, il est recommende d'utiliser des modeles de regularisation
#   qui sont pertinents par rapports aux donnees etudiees.

