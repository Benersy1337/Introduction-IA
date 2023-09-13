# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


###Exploração dos dados
base_poker = pd.read_csv('poker-hand-testing.data')

base_poker.head(10)

base_poker.tail(5)

base_poker.describe()

base_poker[base_poker['HAND']==9]

# #Pré-processamento
# #Valores nulos

base_poker.isnull().sum()

# ##Divisão do dataset entre previsores e classe

X_poker = base_poker.iloc[:,0:11].values

y_poker = base_poker.iloc[:,10].values

# ###Escalonamento de valores
X_poker[:,0].min(), X_poker[:,1].min(), X_poker[:,2].min(),X_poker[:,3].min(),X_poker[:,4].min(),X_poker[:,5].min()
X_poker[:,6].min(),X_poker[:,7].min(),X_poker[:,8].min(),X_poker[:,9].min(),X_poker[:,10].min()

X_poker[:,0].max(), X_poker[:,1].max(), X_poker[:,2].max(),X_poker[:,3].max(),X_poker[:,4].max(),X_poker[:,5].max()
X_poker[:,6].max(),X_poker[:,7].max(),X_poker[:,8].max(),X_poker[:,9].max(),X_poker[:,10].max()

#normalização
from sklearn.preprocessing import MinMaxScaler
scaler_poker = MinMaxScaler()
X_poker = scaler_poker.fit_transform(X_poker)


# #Tratamento de valores categóricos
# #Label encoder
# #P M G GG XG
# #0 1 2 3  4

# from sklearn.preprocessing import LabelEncoder

# label_encoder = LabelEncoder()

# indices = [1,3,5,6,7,8,9,13]

# for i in indices:
#     X_census[:,i] = label_encoder.fit_transform(X_census[:,i])
    
    

#Divisão entre base de treinamento e testes

from sklearn.model_selection import train_test_split

X_poker_treinamento, X_poker_teste, y_poker_treinamento, y_poker_teste = train_test_split(X_poker,y_poker,test_size=0.25,random_state=0)
                                                                                              

#Salvar variáveis
import pickle

with open('poker.pkl', mode='wb') as f:
    pickle.dump([X_poker_treinamento, y_poker_treinamento, X_poker_teste,y_poker_teste],f)
    


# ###########NAIVE BAYES##############


import pickle
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB

with open('poker.pkl','rb') as f:
    X_poker_treinamento, y_poker_treinamento, X_poker_teste,y_poker_teste = pickle.load(f)

naive_poker = GaussianNB()

####Aprendizado (Treinamento)####
naive_poker.fit(X_poker_treinamento, y_poker_treinamento)

####Previsão###############
previsoes_poker = naive_poker.predict(X_poker_teste)

from sklearn.metrics import accuracy_score

accuracy_score(y_poker_teste, previsoes_poker)
#100%

from sklearn.metrics import confusion_matrix

confusion_matrix(y_poker_teste, previsoes_poker)

from yellowbrick.classifier import ConfusionMatrix
#pip install yellowbrick

cm = ConfusionMatrix(naive_poker)
cm.fit(X_poker_treinamento, y_poker_treinamento)
cm.score(X_poker_teste,y_poker_teste)

from sklearn.metrics import classification_report

classification_report(y_credit_teste, previsoes_credit)



# ####ARVORES DE DECISAO#########################


from sklearn.tree import DecisionTreeClassifier

with open('poker.pkl','rb') as f:
    X_poker_treinamento, y_poker_treinamento, X_poker_teste,y_poker_teste = pickle.load(f)

dt_poker = DecisionTreeClassifier(criterion='entropy', random_state=0)

####Aprendizado (Treinamento)####
dt_poker.fit(X_poker_treinamento, y_poker_treinamento)

####Previsão###############
previsoes_poker = dt_poker.predict(X_poker_teste)

from sklearn.metrics import accuracy_score

accuracy_score(y_poker_teste, previsoes_poker)
#100%

from sklearn.metrics import confusion_matrix

confusion_matrix(y_poker_teste, previsoes_poker)

from yellowbrick.classifier import ConfusionMatrix
#pip install yellowbrick

cm = ConfusionMatrix(dt_poker)
cm.fit(X_poker_treinamento, y_poker_treinamento)
cm.score(X_poker_teste,y_poker_teste)

from sklearn.metrics import classification_report

classification_report(y_poker_teste, previsoes_poker)

####Imprimir a arvore de decisão
from sklearn import tree
import matplotlib.pyplot as plt

previsores = ['S1','C1','S2','C2','S3','C3','S4','C4','S5','C5']
classes = ['0','1','2','3','4','5','6','7','8','9']
fig, axes = plt.subplots(nrows=1, ncols=1, figsize = (100,100))
tree.plot_tree(dt_poker, feature_names=previsores, class_names=classes, filled=True)
fig.savefig('arvore_poker.png')



# #####RANDOM FOREST############

from sklearn.ensemble import RandomForestClassifier

with open('poker.pkl','rb') as f:
    X_poker_treinamento, y_poker_treinamento, X_poker_teste,y_poker_teste = pickle.load(f)

rf_poker = RandomForestClassifier(n_estimators=40,criterion='entropy', random_state=0)

####Aprendizado (Treinamento)####
rf_poker.fit(X_poker_treinamento, y_poker_treinamento)

####Previsão###############
previsoes_poker = rf_poker.predict(X_poker_teste)

from sklearn.metrics import accuracy_score

accuracy_score(y_poker_teste, previsoes_poker)
#99%

from sklearn.metrics import confusion_matrix

confusion_matrix(y_poker_teste, previsoes_poker)

from yellowbrick.classifier import ConfusionMatrix
#pip install yellowbrick

cm = ConfusionMatrix(rf_poker)
cm.fit(X_poker_treinamento, y_poker_treinamento)
cm.score(X_poker_teste,y_poker_teste)
#99%

from sklearn.metrics import classification_report

classification_report(y_poker_teste, previsoes_poker)




# ####################################################################
# #
# # APRENDIZADO BASEADO EM INSTÂNCIAS - KNN
# #
# ####################################################################

from sklearn.neighbors import KNeighborsClassifier

import pickle

with open('poker.pkl','rb') as f:
    X_poker_treinamento, y_poker_treinamento, X_poker_teste,y_poker_teste = pickle.load(f)
    
knn_poker = KNeighborsClassifier(n_neighbors=5)
knn_poker.fit(X_poker_treinamento, y_poker_treinamento)

previsoes = knn_poker.predict(X_poker_teste)

from sklearn.metrics import accuracy_score

accuracy_score(y_poker_teste, previsoes)

# processamento demorado
#62%

from sklearn.metrics import confusion_matrix

confusion_matrix(y_poker_teste, previsoes)

from yellowbrick.classifier import ConfusionMatrix
#pip install yellowbrick

cm = ConfusionMatrix(knn_poker)
cm.fit(X_poker_treinamento, y_poker_treinamento)
cm.score(X_poker_teste,y_poker_teste)

# processamento demorado
#62%

from sklearn.metrics import classification_report

classification_report(y_poker_teste, previsoes)




# ################################################
# #
# # SVM
# #
# ##################################################

from sklearn.svm import SVC

import pickle

with open('poker.pkl','rb') as f:
    X_poker_treinamento, y_poker_treinamento, X_poker_teste,y_poker_teste = pickle.load(f)
    
svm_poker = SVC(kernel='rbf', C=4, random_state=1)
svm_poker.fit(X_poker_treinamento, y_poker_treinamento)

previsoes = svm_poker.predict(X_poker_teste)

from sklearn.metrics import accuracy_score

accuracy_score(y_poker_teste, previsoes)

#98,4%

from sklearn.metrics import confusion_matrix

confusion_matrix(y_poker_teste, previsoes)

from yellowbrick.classifier import ConfusionMatrix
#pip install yellowbrick

cm = ConfusionMatrix(svm_poker)
cm.fit(X_poker_treinamento, y_poker_treinamento)
cm.score(X_poker_teste,y_poker_teste)

from sklearn.metrics import classification_report

classification_report(y_poker_teste, previsoes)




# #######################################################################
# #
# # REDES NEURAIS
# #
# #######################################################################


from sklearn.neural_network import MLPClassifier

#Base credit data
import pickle
with open('poker.pkl','rb') as f:
    X_poker_treinamento, y_poker_treinamento, X_poker_teste,y_poker_teste = pickle.load(f)

rede_neural_poker = MLPClassifier(max_iter=1500, verbose=True, tol=0.0000001)
rede_neural_poker.fit(X_poker_treinamento, y_poker_treinamento)

previsoes = rede_neural_poker.predict(X_poker_teste)

from sklearn.metrics import accuracy_score

accuracy_score(y_poker_teste, previsoes)

#99,6%

from sklearn.metrics import confusion_matrix

confusion_matrix(y_poker_teste, previsoes)

from yellowbrick.classifier import ConfusionMatrix
#pip install yellowbrick

cm = ConfusionMatrix(rede_neural_poker)
cm.fit(X_poker_treinamento, y_poker_treinamento)
cm.score(X_poker_teste,y_poker_teste)

from sklearn.metrics import classification_report

classification_report(y_poker_teste, previsoes)





















