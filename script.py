# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

########################################################
#AULA 02
########################################################

###Exploração dos dados
poker_hand = pd.read_csv('poker-hand-testing.data')

# base_credit.head(10)

# base_credit.tail(5)

# base_credit.describe()

# base_credit[base_credit['income']>=45331]

# ###Visualizar os dados
# np.unique(base_credit['default'], return_counts=True)

# sns.countplot(x=base_credit['default'])

# plt.hist(x=base_credit['HAND'])

# plt.hist(x=base_credit['income'])

# plt.hist(x=base_credit['loan'])

# sns.pairplot(base_credit,vars=['age','income'],hue='default')

# sns.pairplot(base_credit,vars=['age','income','loan'],hue='default')

# ### Tratamento de dados inconsistentes
# base_credit.loc[base_credit['age']<0]

# #Apagar a coluna?
# base_credit2 = base_credit.drop('age',axis=1)

# #apagar somente as linhas com dados inconsistentes
# base_credit[base_credit['age']<0].index

# base_credit3 = base_credit.drop(base_credit[base_credit['age']<0].index)


# #Preencher manualmente com valores médios
# base_credit.mean()
# base_credit['age'][base_credit['age'] > 0].mean()
# base_credit.loc[base_credit['age'] < 0, 'age'] = 40.93

# base_credit.loc[base_credit['age']<0]

# ###Tratamento de valores nulos ou faltantes
# base_credit.isnull().sum()

# base_credit.loc[pd.isnull(base_credit['age']),'age'] = 40.93


# ##Divisão do dataset entre previsores e classe
# X_credit = base_credit.iloc[:,1:4].values
# y_credit = base_credit.iloc[:,4].values

# ###Escalonamento de valores
# X_credit[:,0].min(), X_credit[:,1].min(), X_credit[:,2].min()
# X_credit[:,0].max(), X_credit[:,1].max(), X_credit[:,2].max()

# #normalização
# from sklearn.preprocessing import MinMaxScaler
# scaler_credit = MinMaxScaler()
# X_credit = scaler_credit.fit_transform(X_credit)


# #### Base de dados do censo

# base_census = pd.read_csv('census.csv')

# #Visualização de dados
# sns.countplot(x=base_census['income'])

# plt.hist(x=base_census['age'])

# plt.hist(x=base_census['education-num'])

# plt.hist(x=base_census['hour-per-week'])

# import plotly.express as px

# grafico = px.treemap(base_census, path=['workclass'])
# grafico.write_html('grafico0.html')

# grafico = px.treemap(base_census, path=['workclass','age'])
# grafico.write_html('grafico1.html')


# #grafico de categorias paralelas

# grafico = px.parallel_categories(base_census, dimensions=['education','income'])
# grafico.write_html('grafico2.html')


# #Pré-processamento
# #Valores nulos
# base_census.isnull().sum()

# #Dividir previsores da classe
# X_census = base_census.iloc[:,0:14].values
# y_census = base_census.iloc[:,14].values

# #Tratamento de valores categóricos
# #Label encoder
# #P M G GG XG
# #0 1 2 3  4

# from sklearn.preprocessing import LabelEncoder

# label_encoder = LabelEncoder()

# indices = [1,3,5,6,7,8,9,13]

# for i in indices:
#     X_census[:,i] = label_encoder.fit_transform(X_census[:,i])
    
    
    
# ########################################################
# #AULA 03
# ########################################################

# ### OneHotEncoder
# #Carros

# #Gol Pálio Corsa
# # 0    1    2

# #Gol    1   0  0
# #Pálio  0   1  0
# #Corsa  0   0  1

# X_census = base_census.iloc[:,0:14].values

# len(np.unique(base_census['workclass']))


# indices = [1,3,5,6,7,8,9,13]

# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer

# onehotencoder_census = ColumnTransformer(transformers=[('OneHot',OneHotEncoder(),indices)],remainder='passthrough')

# X_census = onehotencoder_census.fit_transform(X_census).toarray()

# X_census.shape

# #Escalonamento dos valores
# from sklearn.preprocessing import StandardScaler
# scaler_sensus = StandardScaler()

# X_census = scaler_sensus.fit_transform(X_census)

# #Divisão entre base de treinamento e testes

# from sklearn.model_selection import train_test_split

# X_census_treinamento, X_census_teste, y_census_treinamento, y_census_teste = train_test_split(X_census,y_census,test_size=0.25,random_state=0)
                                                                                              
# X_credit_treinamento, X_credit_teste, y_credit_treinamento, y_credit_teste = train_test_split(X_credit,y_credit,test_size=0.25,random_state=0)

# #Salvar variáveis
# import pickle

# with open('census.pkl', mode='wb') as f:
#     pickle.dump([X_census_treinamento, y_census_treinamento, X_census_teste,y_census_teste],f)
    
# with open('credit.pkl', mode='wb') as f:
#     pickle.dump([X_credit_treinamento, y_credit_treinamento, X_credit_teste,y_credit_teste],f)


# ###########NAIVE BAYES##############


# import pickle
# import pandas as pd
# import numpy as np
# from sklearn.naive_bayes import GaussianNB

# with open('credit.pkl','rb') as f:
#     X_credit_treinamento, y_credit_treinamento, X_credit_teste,y_credit_teste = pickle.load(f)

# naive_credit = GaussianNB()

# ####Aprendizado (Treinamento)####
# naive_credit.fit(X_credit_treinamento, y_credit_treinamento)

# ####Previsão###############
# previsoes_credit = naive_credit.predict(X_credit_teste)

# from sklearn.metrics import accuracy_score

# accuracy_score(y_credit_teste, previsoes_credit)
# #93%

# from sklearn.metrics import confusion_matrix

# confusion_matrix(y_credit_teste, previsoes_credit)

# from yellowbrick.classifier import ConfusionMatrix
# #pip install yellowbrick

# cm = ConfusionMatrix(naive_credit)
# cm.fit(X_credit_treinamento, y_credit_treinamento)
# cm.score(X_credit_teste,y_credit_teste)

# from sklearn.metrics import classification_report

# classification_report(y_credit_teste, previsoes_credit)


# ## Base do census

# with open('census.pkl','rb') as f:
#     X_census_treinamento, y_census_treinamento, X_census_teste,y_census_teste = pickle.load(f)

# naive_census = GaussianNB()

# ####Aprendizado (Treinamento)####
# naive_census.fit(X_census_treinamento, y_census_treinamento)

# ####Previsão###############
# previsoes_census = naive_census.predict(X_census_teste)

# from sklearn.metrics import accuracy_score

# accuracy_score(y_census_teste, previsoes_census)
# #47,967%

# from sklearn.metrics import confusion_matrix

# confusion_matrix(y_census_teste, previsoes_census)

# from yellowbrick.classifier import ConfusionMatrix
# #pip install yellowbrick

# cm = ConfusionMatrix(naive_census)
# cm.fit(X_census_treinamento, y_census_treinamento)
# cm.score(X_census_teste,y_census_teste)

# from sklearn.metrics import classification_report

# classification_report(y_census_teste, previsoes_census)


# ####ARVORES DE DECISAO#########################
# from sklearn.tree import DecisionTreeClassifier

# with open('credit.pkl','rb') as f:
#     X_credit_treinamento, y_credit_treinamento, X_credit_teste,y_credit_teste = pickle.load(f)

# dt_credit = DecisionTreeClassifier(criterion='entropy', random_state=0)

# ####Aprendizado (Treinamento)####
# dt_credit.fit(X_credit_treinamento, y_credit_treinamento)

# ####Previsão###############
# previsoes_credit = dt_credit.predict(X_credit_teste)

# from sklearn.metrics import accuracy_score

# accuracy_score(y_credit_teste, previsoes_credit)
# #98.2%

# from sklearn.metrics import confusion_matrix

# confusion_matrix(y_credit_teste, previsoes_credit)

# from yellowbrick.classifier import ConfusionMatrix
# #pip install yellowbrick

# cm = ConfusionMatrix(dt_credit)
# cm.fit(X_credit_treinamento, y_credit_treinamento)
# cm.score(X_credit_teste,y_credit_teste)

# from sklearn.metrics import classification_report

# classification_report(y_credit_teste, previsoes_credit)

# ####Imprimir a arvore de decisão
# from sklearn import tree
# import matplotlib.pyplot as plt

# previsores = ['income','age','loan']
# classes = ['0','1']
# fig, axes = plt.subplots(nrows=1, ncols=1, figsize = (20,20))
# tree.plot_tree(dt_credit, feature_names=previsores, class_names=classes, filled=True)
# fig.savefig('arvore_credit.png')


# ## Base do census

# with open('census.pkl','rb') as f:
#     X_census_treinamento, y_census_treinamento, X_census_teste,y_census_teste = pickle.load(f)

# dt_census = DecisionTreeClassifier(criterion='entropy', random_state=0)


# ####Aprendizado (Treinamento)####
# dt_census.fit(X_census_treinamento, y_census_treinamento)

# ####Previsão###############
# previsoes_census = dt_census.predict(X_census_teste)

# from sklearn.metrics import accuracy_score

# accuracy_score(y_census_teste, previsoes_census)
# #81,64%

# from sklearn.metrics import confusion_matrix

# confusion_matrix(y_census_teste, previsoes_census)

# from yellowbrick.classifier import ConfusionMatrix
# #pip install yellowbrick

# cm = ConfusionMatrix(dt_census)
# cm.fit(X_census_treinamento, y_census_treinamento)
# cm.score(X_census_teste,y_census_teste)

# from sklearn.metrics import classification_report

# classification_report(y_census_teste, previsoes_census)



# #####RANDOM FOREST############

# from sklearn.ensemble import RandomForestClassifier

# with open('credit.pkl','rb') as f:
#     X_credit_treinamento, y_credit_treinamento, X_credit_teste,y_credit_teste = pickle.load(f)

# rf_credit = RandomForestClassifier(n_estimators=40,criterion='entropy', random_state=0)

# ####Aprendizado (Treinamento)####
# rf_credit.fit(X_credit_treinamento, y_credit_treinamento)

# ####Previsão###############
# previsoes_credit = rf_credit.predict(X_credit_teste)

# from sklearn.metrics import accuracy_score

# accuracy_score(y_credit_teste, previsoes_credit)
# #98.4%

# from sklearn.metrics import confusion_matrix

# confusion_matrix(y_credit_teste, previsoes_credit)

# from yellowbrick.classifier import ConfusionMatrix
# #pip install yellowbrick

# cm = ConfusionMatrix(rf_credit)
# cm.fit(X_credit_treinamento, y_credit_treinamento)
# cm.score(X_credit_teste,y_credit_teste)

# from sklearn.metrics import classification_report

# classification_report(y_credit_teste, previsoes_credit)


# ## Base do census

# with open('census.pkl','rb') as f:
#     X_census_treinamento, y_census_treinamento, X_census_teste,y_census_teste = pickle.load(f)

# rf_census = RandomForestClassifier(n_estimators=40,criterion='entropy', random_state=0)


# ####Aprendizado (Treinamento)####
# rf_census.fit(X_census_treinamento, y_census_treinamento)

# ####Previsão###############
# previsoes_census = rf_census.predict(X_census_teste)

# from sklearn.metrics import accuracy_score

# accuracy_score(y_census_teste, previsoes_census)
# #84,92%

# from sklearn.metrics import confusion_matrix

# confusion_matrix(y_census_teste, previsoes_census)

# from yellowbrick.classifier import ConfusionMatrix
# #pip install yellowbrick

# cm = ConfusionMatrix(rf_census)
# cm.fit(X_census_treinamento, y_census_treinamento)
# cm.score(X_census_teste,y_census_teste)

# from sklearn.metrics import classification_report

# classification_report(y_census_teste, previsoes_census)

# ###################################################################
# #
# #
# #  AULA 05
# #
# ####################################################################
# #
# # APRENDIZADO BASEADO EM INSTÂNCIAS - KNN
# #
# ####################################################################

# from sklearn.neighbors import KNeighborsClassifier

# ###Base credit
# import pickle
# with open('credit.pkl','rb') as f:
#     X_credit_treinamento, y_credit_treinamento, X_credit_teste,y_credit_teste = pickle.load(f)
    
# knn_credit = KNeighborsClassifier(n_neighbors=5)
# knn_credit.fit(X_credit_treinamento, y_credit_treinamento)

# previsoes = knn_credit.predict(X_credit_teste)

# from sklearn.metrics import accuracy_score

# accuracy_score(y_credit_teste, previsoes)

# #98,4%

# from sklearn.metrics import confusion_matrix

# confusion_matrix(y_credit_teste, previsoes)

# from yellowbrick.classifier import ConfusionMatrix
# #pip install yellowbrick

# cm = ConfusionMatrix(knn_credit)
# cm.fit(X_credit_treinamento, y_credit_treinamento)
# cm.score(X_credit_teste,y_credit_teste)

# from sklearn.metrics import classification_report

# classification_report(y_credit_teste, previsoes)

# ### Base census

# ###Base credit
# import pickle
# with open('census.pkl','rb') as f:
#     X_census_treinamento, y_census_treinamento, X_census_teste,y_census_teste = pickle.load(f)
    
# knn_credit = KNeighborsClassifier(n_neighbors=10)
# knn_credit.fit(X_census_treinamento, y_census_treinamento)

# previsoes = knn_credit.predict(X_census_teste)

# from sklearn.metrics import accuracy_score

# accuracy_score(y_census_teste, previsoes)

# #83,01

# from sklearn.metrics import confusion_matrix

# confusion_matrix(y_census_teste, previsoes)

# from yellowbrick.classifier import ConfusionMatrix
# #pip install yellowbrick

# cm = ConfusionMatrix(knn_credit)
# cm.fit(X_census_treinamento, y_census_treinamento)
# cm.score(X_census_teste,y_census_teste)

# from sklearn.metrics import classification_report

# classification_report(y_census_teste, previsoes)

# ################################################
# #
# # SVM
# #
# ##################################################

# from sklearn.svm import SVC

# ###Base credit
# import pickle
# with open('credit.pkl','rb') as f:
#     X_credit_treinamento, y_credit_treinamento, X_credit_teste,y_credit_teste = pickle.load(f)
    
# svm_credit = SVC(kernel='rbf', C=4, random_state=1)
# svm_credit.fit(X_credit_treinamento, y_credit_treinamento)

# previsoes = svm_credit.predict(X_credit_teste)

# from sklearn.metrics import accuracy_score

# accuracy_score(y_credit_teste, previsoes)

# #98,4%

# from sklearn.metrics import confusion_matrix

# confusion_matrix(y_credit_teste, previsoes)

# from yellowbrick.classifier import ConfusionMatrix
# #pip install yellowbrick

# cm = ConfusionMatrix(svm_credit)
# cm.fit(X_credit_treinamento, y_credit_treinamento)
# cm.score(X_credit_teste,y_credit_teste)

# from sklearn.metrics import classification_report

# classification_report(y_credit_teste, previsoes)

# ### Base census
# import pickle
# with open('census.pkl','rb') as f:
#     X_census_treinamento, y_census_treinamento, X_census_teste,y_census_teste = pickle.load(f)
    
# svm_credit = SVC(kernel='rbf', C=4)
# svm_credit.fit(X_census_treinamento, y_census_treinamento)

# previsoes = svm_credit.predict(X_census_teste)

# from sklearn.metrics import accuracy_score

# accuracy_score(y_census_teste, previsoes)

# #84,69

# from sklearn.metrics import confusion_matrix

# confusion_matrix(y_census_teste, previsoes)

# from yellowbrick.classifier import ConfusionMatrix
# #pip install yellowbrick

# cm = ConfusionMatrix(svm_credit)
# cm.fit(X_census_treinamento, y_census_treinamento)
# cm.score(X_census_teste,y_census_teste)

# from sklearn.metrics import classification_report

# classification_report(y_census_teste, previsoes)

# #######################################################################
# #
# # REDES NEURAIS
# #
# #######################################################################
# from sklearn.neural_network import MLPClassifier

# #Base credit data
# import pickle
# with open('credit.pkl','rb') as f:
#     X_credit_treinamento, y_credit_treinamento, X_credit_teste,y_credit_teste = pickle.load(f)

# rede_neural_credit = MLPClassifier(max_iter=1500, verbose=True, tol=0.0000001)
# rede_neural_credit.fit(X_credit_treinamento, y_credit_treinamento)

# previsoes = rede_neural_credit.predict(X_credit_teste)

# from sklearn.metrics import accuracy_score

# accuracy_score(y_credit_teste, previsoes)

# #99,6%

# from sklearn.metrics import confusion_matrix

# confusion_matrix(y_credit_teste, previsoes)

# from yellowbrick.classifier import ConfusionMatrix
# #pip install yellowbrick

# cm = ConfusionMatrix(rede_neural_credit)
# cm.fit(X_credit_treinamento, y_credit_treinamento)
# cm.score(X_credit_teste,y_credit_teste)

# from sklearn.metrics import classification_report

# classification_report(y_credit_teste, previsoes)


# ### Base census
# import pickle
# with open('census.pkl','rb') as f:
#     X_census_treinamento, y_census_treinamento, X_census_teste,y_census_teste = pickle.load(f)
    
# redes_neurais_credit = MLPClassifier(max_iter=1500, verbose=True, tol=0.00000000001)
# redes_neurais_credit.fit(X_census_treinamento, y_census_treinamento)


# from sklearn.metrics import accuracy_score

# accuracy_score(y_census_teste, previsoes)

# #82,72

# from sklearn.metrics import confusion_matrix

# confusion_matrix(y_census_teste, previsoes)

# from yellowbrick.classifier import ConfusionMatrix
# #pip install yellowbrick

# cm = ConfusionMatrix(redes_neurais_credit)
# cm.fit(X_census_treinamento, y_census_treinamento)
# cm.score(X_census_teste,y_census_teste)

# from sklearn.metrics import classification_report

# classification_report(y_census_teste, previsoes)




















