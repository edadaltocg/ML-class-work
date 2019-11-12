
#exemple inspire de http://scikit-learn.org/stable/_downloads/plot_isotonic_regression.py

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

"""
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#PARTIE 1 : Utilisation de scikit-learn pour la regression lineaire
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""

#generation de donnees test
n = 100
x = np.arange(n)
y = np.random.randn(n)*30 + 50. * np.log(1 + np.arange(n))

# instanciation de sklearn.linear_model.LinearRegression
lr = LinearRegression()
lr.fit(x[:, np.newaxis], y)  # np.newaxis est utilise car x doit etre une matrice 2d avec 'LinearRegression'

# representation du resultat
fig = plt.figure()
plt.plot(x, y, 'r.')
plt.plot(x, lr.predict(x[:, np.newaxis]), 'b-')
plt.legend(('Data', 'Linear Fit'), loc='lower right')
plt.title('Linear regression')
plt.show()


#QUESTION 1.1 : 
#Bien comprendre le fonctionnement de lr, en particulier lr.fit et lr.predict

#-> lr est un objet qui permet d'effectuer la regression lineaire
#-> lr.fit permet d'apprendre les parametres du modele a partir de donnees d'apprentissage
#-> lr.predict permet de predire un 'y' partir d'un 'x' test

#QUESTION 1.2 :
#On s'interesse a x=105. En supposant que le model lineaire soit toujours 
#valide pour ce x, quelles valeur corresondante de y vous semble la plus 
#vraisemblable ? 

lr.predict([[105]])

#la valeur est 264.80754151
#On remarque que les valeurs donnees pour la prediction doivent etre dans un vecteur colonne, ici une matrice 1x1

"""
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#PARTIE 2 : impact et detection d'outliers
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""

#generation de donnees test
n = 10
x = np.arange(n)
y = 10. + 4.*x + np.random.randn(n)*3. 
y[9]=y[9]+20

# instanciation de sklearn.linear_model.LinearRegression
lr = LinearRegression()
lr.fit(x[:, np.newaxis], y)  # np.newaxis est utilise car x doit etre une matrice 2d avec 'LinearRegression'

# representation du resultat

print('b_0='+str(lr.intercept_)+' et b_1='+str(lr.coef_[0]))

fig = plt.figure()
plt.plot(x, y, 'r.')
plt.plot(x, lr.predict(x[:, np.newaxis]), 'b-')
plt.legend(('Data', 'Linear Fit'), loc='lower right')
plt.title('Linear regression')
plt.show()


#QUESTION 2.1 : 
#La ligne 'y[9]=y[9]+20' genere artificiellement une donnee aberrante.
#-> Tester l'impact de la donnee aberrante en estimant b_0, b_1 et s^2 
#   sur 5 jeux de donnees qui la contiennent cette donnee et 5 autres qui
#   ne la contiennent pas (simplement ne pas executer la ligne y[9]=y[9]+20).
#   On remarque que $\beta_0 = 10$, $\beta_1 = 4$ et $sigma=3$ dans les 
#   donnÃ©es simulees.

#sans donnee aberrante
for i in range(5):
  n = 10
  x = np.arange(n)
  y = 10. + 4.*x + np.random.randn(n)*3. 
  lr.fit(x[:, np.newaxis], y)  # np.newaxis est utilise car x doit etre une matrice 2d avec 'LinearRegression'
  print('b_0='+str(lr.intercept_)+' b_1='+str(lr.coef_[0])+' / les valeurs recherchees sont 10 et 4')
  s=np.std(y-lr.predict(x[:, np.newaxis]))
  print('Bruit estime='+str(s)+'    /  reel = 3 ')

#avec donnee aberrante
for i in range(5):
  n = 10
  x = np.arange(n)
  y = 10. + 4.*x + np.random.randn(n)*3. 
  y[9]=y[9]+20
  lr.fit(x[:, np.newaxis], y)  # np.newaxis est utilise car x doit etre une matrice 2d avec 'LinearRegression'
  print('b_0='+str(lr.intercept_)+' b_1='+str(lr.coef_[0])+' / les valeurs recherchees sont 10 et 4')
  s=np.std(y-lr.predict(x[:, np.newaxis]))
  print('Bruit estime='+str(s)+'    /  reel = 3 ')


#-> estimations correctes sans donnee aberrante mais biaisees avec la donnee aberrante.
#   On peut mesurer le biais en comparant la moyenne des parametres estimes aux vrais valeures des parametres
#   sur un grand nombre de repetitions

#QUESTION 2.2 : 
#2.2.a -> Pour chaque variable i, calculez les profils des rÃ©sidus 
#         $e_{(i)j}=y_j - \hat{y_{(i)j}}$ pour tous les j, ou   
#         \hat{y_{(i)j}} est l'estimation de y_j a partir d'un modele  
#         lineaire appris sans l'observation i.
#2.2.b -> En quoi le profil des e_{(i)j} est different pour i=9 que pour  
#         les autre i
#2.2.c -> Etendre ces calculs pour dÃ©finir la distance de Cook de chaque 
#         variable i
#
#AIDE : pour enlever un element 'i' de 'x' ou 'y', utiliser 
#       x_del_i=np.delete(x,i) et y_del_i=np.delete(y,i) 

#-> regeneration de donnees biaisees 
n = 10
x = np.arange(n)
y = 10. + 4.*x + np.random.randn(n)*3. 
y[9]=y[9]+20

#-> question 2.2.a
for i in range(n):
  x_del_i=np.delete(x,i)
  y_del_i=np.delete(y,i) 
  
  lr.fit(x_del_i[:, np.newaxis], y_del_i)
  
  print('variable supprimee='+str(i))
  print('residus ='+str(y-lr.predict(x[:, np.newaxis])))

#2.2.b -> l'estimation de j=9 est toujours la plus mauvaise. Elle est plus mauvaise quand i=9  
#         enleve de l'apprentissage que pour tous les autres i est, alors que toutes les 
#         autres predictions sont meilleures.
#         ... on peut clairement se douter que cette observation est un outlier (donnee aberrante). 

#-> question 2.2.c

lr.fit(x[:, np.newaxis], y)  # np.newaxis est utilise car x doit etre une matrice 2d avec 'LinearRegression'
y_pred=lr.predict(x[:, np.newaxis])

s2=np.sum((y-y_pred)*(y-y_pred))/(n-2)




for i in range(n):
  x_del_i=np.delete(x,i)
  y_del_i=np.delete(y,i) 
  
  lr.fit(x_del_i[:, np.newaxis], y_del_i)
  sum_squared_error=np.sum((y_pred-lr.predict(x[:, np.newaxis]))*(y_pred-lr.predict(x[:, np.newaxis])))
  
  print('D('+str(i)+')='+str(sum_squared_error/(2.*s2)))

#->la distance est clairement plus grande pour la valeur 9 que toutes les autres, qui sont stable. La donnee 
#  aberrante est encore retrouvee.





"""
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#PARTIE 3 : Vers la regression lineaire multiple et optimisation
#
#On considere que l'on connait les notes moyennes sur l'annee de n eleves 
#dans p matieres, ainsi que leur note a un concours en fin d'annee. On 
#se demande si on ne pourrait pas predire la note des etudiants au 
#concours en fonction de leur moyenne annuelle afin d'estimer leurs 
#chances au concours.
#
#On va resoudre le probleme a l'aide de la regression lineaire en 
#dimension p>1 sans utiliser scikit-learn. 
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""

#Question 1 :
# - A l'aide de la fonction 'SimulateObservations', simulez un jeu de donnees d'apprentissage [X_l,y_l] avec 30 observations et un jeu de test [X_t,y_t] avec 10 observations. Les observations seront en dimension p=10

def SimulateObservations(n_train,n_test,p):
  """
  n_train: number of training obserations to simulate
  n_test: number of test obserations to simulate
  p: dimension of the observations to simulate
  """
  
  ObsX_train=20.*np.random.rand(n_train,p)
  ObsX_tst=20.*np.random.rand(n_test,p)
  
  RefTheta=np.random.rand(p)**3
  RefTheta=RefTheta/RefTheta.sum()
  print("The thetas with which the values were simulated is: "+str(RefTheta))
  
  ObsY_train=np.dot(ObsX_train,RefTheta.reshape(p,1))+1.5*np.random.randn(n_train,1)
  ObsY_tst=np.dot(ObsX_tst,RefTheta.reshape(p,1))+1.5*np.random.randn(n_test,1)
  
  return [ObsX_train,ObsY_train,ObsX_tst,ObsY_tst,RefTheta]

p=10

[ObsX_train,ObsY_train,ObsX_tst,ObsY_tst,RefTheta]=SimulateObservations(30,10,p)




#Question 2 :
# - On considere un modele lineaire en dimension p>1 mettre en lien les x[i,:] et les y[i], c'est a dire que np.dot(x[i,:],theta_optimal) doit etre le plus proche possible de y[i] sur l'ensemble des observations i. Dans le modele lineaire multiple, theta_optimal est un vecteur de taille [p,1] qui pondere les differentes variables observees (ici les moyennes dans une matiere). Coder alors une fonction qui calcule la moyenne des differences au carre entre ces valeurs en fonction de theta.

def CptMSE(X,y_true,theta_test):
  y_pred=np.dot(X,theta_test)[:, np.newaxis]
  print(y_pred.shape)
  print(y_true.shape)
  MSE=np.mean(np.power(y_pred-y_true,2.))
  
  return MSE


theta_test=np.abs(np.random.randn(p))

MSE_test=CptMSE(ObsX_train,ObsY_train,theta_test)


#Question 3 -- option 1 :
# - On va maintenant chercher le theta_test qui minimise cette fonction (il correspondra a theta_optimal), et ainsi rÃ©soudre le probleme d'apprentissage de regression lineaire multiple. Utiliser pour cela la fonction minimize de scipy.optimize


global X
global y_true

def CptMyPbSpecificMSE(theta_test):
  """
  fonction optimisee par minimize
  """
  global X
  global y_true
  
  MSE_test=CptMSE(ObsX_train,ObsY_train,theta_test)
  
  return MSE_test


from scipy.optimize import minimize

theta_init=np.abs(np.random.randn(p))/10.


res = minimize(CptMyPbSpecificMSE, theta_init, method='Powell',options={'xtol': 10, 'disp': True})

print("Optimal theta found="+str(res.x))


#) ... validate the results

print('Results assessment on the training set')

MSE_init=CptMSE(ObsX_train,ObsY_train,theta_test)
print("MSE="+str(MSE_init)+" with theta init")


MSE_final=CptMSE(ObsX_train,ObsY_train,res.x)
print("MSE="+str(MSE_final)+" with the final theta")


MSE_ref=CptMSE(ObsX_train,ObsY_train,RefTheta)
print("MSE="+str(MSE_ref)+" with the reference theta")




print('Results assessment on the test set')

MSE_init=CptMSE(ObsX_tst,ObsY_tst,theta_test)
print("MSE="+str(MSE_init)+" with theta init")


MSE_final=CptMSE(ObsX_tst,ObsY_tst,res.x)
print("MSE="+str(MSE_final)+" with the final theta")


MSE_ref=CptMSE(ObsX_tst,ObsY_tst,RefTheta)
print("MSE="+str(MSE_ref)+" with the reference theta")




# -> il est interessant de remarquer qu'on a sur-appris : meilleur resultat avec 'MSE_final' que 'MSE_ref' sur 'training set'    /    meilleur resultat avec 'MSE_ref' que 'MSE_final' sur 'test set'







#Question 3 -- option 2 :
#De maniere alternative, le probleme peut etre resolu a l'aide d'une methode de descente de gradient codee a la main, dans laquelle les gradients seront calcules par differences finies.

def gradientApprox(fct_to_minimize,theta_loc,X_loc,Y_loc,epsilon=1e-5):
  
  fx=fct_to_minimize(X_loc,Y_loc,theta_loc)
  print(fx)
  ApproxGrad=np.zeros(np.size(theta_loc))
  veps=np.zeros(np.size(theta_loc))
  
  for i in range(np.size(theta_loc)):
    veps[:]=0.
    veps[i]+=epsilon
    ApproxGrad[i]=(fct_to_minimize(X_loc,Y_loc,theta_loc+veps)-fx)/epsilon
  return ApproxGrad


def GradientDescent(fct_to_minimize,theta_init,X_loc,Y_loc,alpha=0.01,N=100):
  """
  Remark: the multiplicatory coefficient of the gradients will be "alpha" divided by the norm of the first gradient 
  """
  
  #init
  l_thetas=[theta_init]
  theta_curr=theta_init.copy()

  #run the gradient descent
  n=0
  while n<N:
    #approximate the gradient of fct_to_minimize w.r.t. theta_curr
    g=gradientApprox(fct_to_minimize,theta_curr,X_loc,Y_loc)
    
    #set the multiplicatory coefficient of the gradients
    if n==0:
      NormFirstGrads=np.linalg.norm(g)
      coefMult=alpha/NormFirstGrads
      
    #update theta
    theta_curr=theta_curr-coefMult*g
    
    #save the current state and increment n
    l_thetas.append(theta_curr)
    n+=1

  return l_thetas


theta_init=np.ones(p)/p


l_thetas=GradientDescent(CptMSE,theta_init,ObsX_train,ObsY_train)


convergence_curve=[]
for i in range(len(l_thetas)):
  convergence_curve.append(np.linalg.norm(l_thetas[i]-RefTheta))


plt.plot(np.array(convergence_curve))
plt.show()

convergence_MSE_train=[]
for i in range(len(l_thetas)):
  convergence_MSE_train.append(CptMSE(ObsX_train,ObsY_train,l_thetas[i]))

plt.plot(np.array(convergence_MSE_train))
plt.title('Convergence of the MSE on the training set (value with the ref. theta='+str(CptMSE(ObsX_train,ObsY_train,RefTheta))+')')
plt.show()


#) ... validate the results


print('Results assessment on the training set')

MSE_init=CptMSE(ObsX_train,ObsY_train,theta_test)
print("MSE="+str(MSE_init)+" with theta init")


MSE_final=CptMSE(ObsX_train,ObsY_train,l_thetas[-1])
print("MSE="+str(MSE_final)+" with the final theta")


MSE_ref=CptMSE(ObsX_train,ObsY_train,RefTheta)
print("MSE="+str(MSE_ref)+" with the reference theta")




print('Results assessment on the test set')

MSE_init=CptMSE(ObsX_tst,ObsY_tst,theta_test)
print("MSE="+str(MSE_init)+" with theta init")


MSE_final=CptMSE(ObsX_tst,ObsY_tst,l_thetas[-1])
print("MSE="+str(MSE_final)+" with the final theta")


MSE_ref=CptMSE(ObsX_tst,ObsY_tst,RefTheta)
print("MSE="+str(MSE_ref)+" with the reference theta")





# -> il est interessant de remarquer qu'on a sur-appris : meilleur resultat avec 'MSE_final' que 'MSE_ref' sur 'training set'    /    meilleur resultat avec 'MSE_ref' que 'MSE_final' sur 'test set'










"""
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#PARTIE 4 : maximum de vraisemblance
# - Tirer 10 fois une piece a pile ou face et modeliser les resultats obtenus comme ceux
#d'une variable aleatoire X qui vaut X_i=0 si on a pile et X_i=1 si on a face.
# - Calculer le maximum de vraisemblance du parametre p d'un loi de Bernoulli qui modeliserait le probleme.
# - VÃ©rifier empiriquement comment Ã©volue ce maximum de vraisemblance si l'on effectue de plus en plus de tirages
# - Que se passe-t-il quand il y a trop de tirages ? ReprÃ©senter la log-vraisemblance plutot que la vraisemblance dans ce cas.
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""


NbPile=4.
NbFace=6.


PossibleValuesForP=np.linspace(0,1,100)

CorrespondingLikelihood=((PossibleValuesForP)**NbPile)*((1-PossibleValuesForP)**NbFace)

plt.plot(PossibleValuesForP,CorrespondingLikelihood)
plt.show()



#-> on peut prendre le max sur cette grille (methode grid search), ou bien on peut faire une descente de gradient comme dans la partie 3



NbPile=47.
NbFace=53.

CorrespondingLikelihood=((PossibleValuesForP)**NbPile)*((1-PossibleValuesForP)**NbFace)

plt.plot(PossibleValuesForP,CorrespondingLikelihood)
plt.show()



#-> on se rapproche peu a peu de p=0.5. Attention aux erreurs numÃ©riques quand le nombre de tirages devient trop grand. On utilisera plutot la log-vraisemblance dans ce cas.



NbPile=1005.
NbFace=995.


CorrespondingLikelihood=((PossibleValuesForP)**NbPile)*((1-PossibleValuesForP)**NbFace)

plt.plot(PossibleValuesForP,CorrespondingLikelihood)  #on ne voit rien
plt.show()



PossibleValuesForP=np.linspace(0.01,0.99,100)

CorrespondingLogLikelihood=NbPile*np.log(PossibleValuesForP)  +     NbFace*np.log(1-PossibleValuesForP)

plt.plot(PossibleValuesForP,CorrespondingLogLikelihood)  #on ne voit rien
plt.show()


