# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from pandas import DataFrame
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import max_error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

#input('\nPress the key to continue...\n')

df = pd.read_csv('test.csv')
df2 = pd.read_csv('train.csv')
zero = 0
base = ['NU_IDADE', 'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_REDACAO']
#print('\n Base Regressão Linear', base)
#print('\nDescritivo base de TREINO')
#print(df2[base].describe())

### Substituindo variaveis NULAS por ZERO no arquivo de teste ###
df.update(df['NU_NOTA_CN'].fillna(zero))
df.update(df['NU_NOTA_CH'].fillna(zero))
df.update(df['NU_NOTA_LC'].fillna(zero))
df.update(df['NU_NOTA_REDACAO'].fillna(zero))
#df_null = df[base]
#print('\n Variaveis nulas df')
#print(df_null.isnull().sum())

df2.update(df2['NU_NOTA_CN'].fillna(zero))
df2.update(df2['NU_NOTA_CH'].fillna(zero))
df2.update(df2['NU_NOTA_LC'].fillna(zero))
df2.update(df2['NU_NOTA_MT'].fillna(zero))
df2.update(df2['NU_NOTA_REDACAO'].fillna(zero))

### Deletando linhas com variaveis MISSING no arquivo de treino ###
#df2 = df2.dropna(subset=['NU_NOTA_CN'])
#df2 = df2.dropna(subset=['NU_NOTA_CH'])
#df2 = df2.dropna(subset=['NU_NOTA_LC'])
#df2 = df2.dropna(subset=['NU_NOTA_MT'])
#df2 = df2.dropna(subset=['NU_NOTA_REDACAO'])
#df2_null = df2[base+['NU_NOTA_MT']]
#print('\n Variaveis nulas df2')
#print(df2_null.isnull().sum())

### Deletando linhas com OUTLIERS no arquivo de treino ###
#df2 = df2.drop((df2.loc[df2['NU_NOTA_CN'] == 0]).index.tolist())
#df2 = df2.drop((df2.loc[df2['NU_NOTA_CH'] == 0]).index.tolist())
#df2 = df2.drop((df2.loc[df2['NU_NOTA_LC'] == 0]).index.tolist())
#df2 = df2.drop((df2.loc[df2['NU_NOTA_MT'] == 0]).index.tolist())
#df2 = df2.drop((df2.loc[df2['NU_NOTA_REDACAO'] == 0]).index.tolist())

#print('Notas CN ZERO: ', len(df2.loc[df2['NU_NOTA_CN'] == 0]))
#print('Notas CH ZERO: ', len(df2.loc[df2['NU_NOTA_CH'] == 0]))
#print('Notas LC ZERO: ', len(df2.loc[df2['NU_NOTA_LC'] == 0]))
#print('Notas MT ZERO: ', len(df2.loc[df2['NU_NOTA_MT'] == 0]))
#print('Notas RD ZERO: ', len(df2.loc[df2['NU_NOTA_REDACAO'] == 0]))

### Alterando variaveis ###
#questoes = {'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'H':8,'I':9,'J':10,'K':11,'L':12,'M':13,'N':14,'O':15,'P':16,'Q':17}
#df2['Q001'] = df2['Q001'].map(questoes)
#df2['Q002'] = df2['Q002'].map(questoes)
#df2['Q006'] = df2['Q006'].map(questoes)
#df2['Q024'] = df2['Q024'].map(questoes)
#df2['Q025'] = df2['Q025'].map(questoes)
#df2['Q026'] = df2['Q026'].map(questoes)
#df2['Q047'] = df2['Q047'].map(questoes)
#df['Q001'] = df['Q001'].map(questoes)
#df['Q002'] = df['Q002'].map(questoes)
#df['Q006'] = df['Q006'].map(questoes)
#df['Q024'] = df['Q024'].map(questoes)
#df['Q025'] = df['Q025'].map(questoes)
#df['Q026'] = df['Q026'].map(questoes)
#df['Q047'] = df['Q047'].map(questoes)

A = df[base]
X = df2[base]
Y = df2.NU_NOTA_MT

# Criando dados de treino e de teste
X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, Y, test_size = 0.30, random_state = 42)

#plt.scatter(df_train_x.NU_NOTA_REDACAO, df_train_y)
#plt.xlabel("Media nota da Redação")
#plt.ylabel("Nota Matemática")
#plt.title("Relação entre Redação e Nota de Matemática")
#plt.show()

regr = LinearRegression().fit(X_treino, Y_treino)
#print('\nCoeficientes de Treinamento\n')
#print('Score: ', regr.score(X_treino, Y_treino))
#print('Intercept_: ', regr.intercept_)
#print('Coeficiente: ', regr.coef_)
#print('Número de Coeficientes: ', len(regr.coef_))
#
#previsao = regr.predict(X_teste)
#
#print('\nMonitoramento de precisão do modelo')
#
#evs = explained_variance_score(Y_teste, previsao)
#print('\nEVS (Explained Variance Score): {0:.2f}'.format(evs), ' [Best value is 1.0]')
#
#mre = max_error(Y_teste, previsao)
#print('\nMRE (Max Residual Error): {0:.2f}'.format(mre), ' [Best value is 0.0]')
#
#mae = mean_absolute_error(Y_teste, previsao)
#print('\nMAE (Mean Absolute Error): {0:.2f}'.format(mae), ' [Best value is 0.0]')
#
#mse = mean_squared_error(Y_teste, previsao)
#print('\nMSE (Mean Squared Error): {0:.2f}'.format(mse), ' [Best value is 0.0]')

#nb_predict_train = regr.predict(df_train_x)
#print("\nExatidão TREINO (Accuracy): {0:.4f}".format(metrics.accuracy_score(df_train_y, nb_predict_train)))

#regr2 = LinearRegression()
#regr2.fit(df_train_x[['NU_NOTA_REDACAO']], df2.NU_NOTA_MT)
#mse2 = np.mean((df2.NU_NOTA_MT - regr2.predict(df_train_x[['NU_NOTA_REDACAO']])) ** 2)
#print('\nMSE 2 (Mean Squared Error): ', mse2)

df['NU_NOTA_MT'] = regr.predict(A)
df_answer = df[['NU_INSCRICAO', 'NU_NOTA_MT']]
df_answer.to_csv('answer.csv', index=False)

#input('\nPress the key to continue...\n')



