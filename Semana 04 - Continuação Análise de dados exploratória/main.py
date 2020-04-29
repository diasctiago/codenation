#!/usr/bin/env python
# coding: utf-8

# # Desafio 3
# 
# Neste desafio, iremos praticar nossos conhecimentos sobre distribuições de probabilidade. Para isso,
# dividiremos este desafio em duas partes:
#     
# 1. A primeira parte contará com 3 questões sobre um *data set* artificial com dados de uma amostra normal e
#     uma binomial.
# 2. A segunda parte será sobre a análise da distribuição de uma variável do _data set_ [Pulsar Star](https://archive.ics.uci.edu/ml/datasets/HTRU2), contendo 2 questões.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF


# In[2]:


#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# ## Parte 1

# ### _Setup_ da parte 1

# In[3]:


np.random.seed(42)
    
dataframe = pd.DataFrame({"normal": sct.norm.rvs(20, 4, size=10000),
                     "binomial": sct.binom.rvs(100, 0.2, size=10000)})


# ## Inicie sua análise a partir da parte 1 a partir daqui

# In[4]:


# Sua análise da parte 1 começa aqui.
dataframe.head(2)


# In[5]:


dataframe.describe()


# In[6]:


dataframe.info()


# In[7]:


#Questão 1
df = (dataframe.describe())
q1_norm = (df.loc['25%'].values)[0]
q2_norm = (df.loc['50%'].values)[0]
q3_norm = (df.loc['75%'].values)[0]
q1_binom = (df.loc['25%'].values)[1]
q2_binom = (df.loc['50%'].values)[1]
q3_binom = (df.loc['75%'].values)[1]
q1_norm_binon = float("{0:.3f}".format(q1_norm - q1_binom))
q2_norm_binon = float("{0:.3f}".format(q2_norm - q2_binom))
q3_norm_binon = float("{0:.3f}".format(q3_norm - q3_binom))
dif = (q1_norm_binon, q2_norm_binon, q3_norm_binon)
dif


# In[21]:


# Questão 1 (Performance)
q_norm = np.quantile(dataframe['normal'], [0.25, 0.50, 0.75]) 
q_binom = np.quantile(dataframe['binomial'], [0.25, 0.50, 0.75]) 
dif_norm_binom = np.around(q_norm-q_binom, 3)
tuple(dif_norm_binom)


# In[187]:


#Questao 2
media = dataframe['normal'].mean()
desvio = dataframe['normal'].std()
dif2 = ECDF(dataframe['normal'])(media + desvio) - ECDF(dataframe['normal'])(media - desvio)
print(dif2)


# In[24]:


# Questão 2 (Performance)
media = dataframe['normal'].mean()
desvio = dataframe['normal'].std()
probabilidade = ECDF(dataframe['normal'])
dif_intervalo = probabilidade(media + desvio) - probabilidade(media - desvio)
dif_intervalo


# In[28]:


#Questao 3
m_norm = dataframe['normal'].mean()
v_norm = dataframe['normal'].var()
m_binom = dataframe['binomial'].mean()
v_binom = dataframe['binomial'].var()
m = float("{0:.3f}".format(m_binom - m_norm))
v = float("{0:.3f}".format(v_binom - v_norm))
dif3 = (m, v)
dif3


# In[46]:


# Questão 3 (Performance)
norm = dataframe.normal.agg(['mean','var'])
binom = dataframe.binomial.agg(['mean','var'])
dif_med_var = np.around(binom-norm, 3)
tuple(dif_med_var)


# ## Questão 1
# 
# Qual a diferença entre os quartis (Q1, Q2 e Q3) das variáveis `normal` e `binomial` de `dataframe`? Responda como uma tupla de três elementos arredondados para três casas decimais.
# 
# Em outra palavras, sejam `q1_norm`, `q2_norm` e `q3_norm` os quantis da variável `normal` e `q1_binom`, `q2_binom` e `q3_binom` os quantis da variável `binom`, qual a diferença `(q1_norm - q1 binom, q2_norm - q2_binom, q3_norm - q3_binom)`?

# In[22]:


def q1():
    q_norm = np.quantile(dataframe['normal'], [0.25, 0.50, 0.75]) 
    q_binom = np.quantile(dataframe['binomial'], [0.25, 0.50, 0.75]) 
    dif_norm_binom = np.around(q_norm-q_binom, 3)
    return tuple(dif_norm_binom)


# Para refletir:
# 
# * Você esperava valores dessa magnitude?
# 
# * Você é capaz de explicar como distribuições aparentemente tão diferentes (discreta e contínua, por exemplo) conseguem dar esses valores?

# ## Questão 2
# 
# Considere o intervalo $[\bar{x} - s, \bar{x} + s]$, onde $\bar{x}$ é a média amostral e $s$ é o desvio padrão. Qual a probabilidade nesse intervalo, calculada pela função de distribuição acumulada empírica (CDF empírica) da variável `normal`? Responda como uma único escalar arredondado para três casas decimais.

# In[191]:


def q2():
    media = dataframe['normal'].mean()
    desvio = dataframe['normal'].std()
    probabilidade = ECDF(dataframe['normal'])
    dif_intervalo = probabilidade(media + desvio) - probabilidade(media - desvio)
    return float(dif_intervalo)


# Para refletir:
# 
# * Esse valor se aproxima do esperado teórico?
# * Experimente também para os intervalos $[\bar{x} - 2s, \bar{x} + 2s]$ e $[\bar{x} - 3s, \bar{x} + 3s]$.

# ## Questão 3
# 
# Qual é a diferença entre as médias e as variâncias das variáveis `binomial` e `normal`? Responda como uma tupla de dois elementos arredondados para três casas decimais.
# 
# Em outras palavras, sejam `m_binom` e `v_binom` a média e a variância da variável `binomial`, e `m_norm` e `v_norm` a média e a variância da variável `normal`. Quais as diferenças `(m_binom - m_norm, v_binom - v_norm)`?

# In[192]:


def q3():
    norm = dataframe.normal.agg(['mean','var'])
    binom = dataframe.binomial.agg(['mean','var'])
    dif_med_var = np.around(binom-norm, 3)
    return tuple(dif_med_var)


# Para refletir:
# 
# * Você esperava valore dessa magnitude?
# * Qual o efeito de aumentar ou diminuir $n$ (atualmente 100) na distribuição da variável `binomial`?

# ## Parte 2

# ### _Setup_ da parte 2

# In[49]:


stars = pd.read_csv("pulsar_stars.csv")

stars.rename({old_name: new_name
              for (old_name, new_name)
              in zip(stars.columns,
                     ["mean_profile", "sd_profile", "kurt_profile", "skew_profile", "mean_curve", "sd_curve", "kurt_curve", "skew_curve", "target"])
             },
             axis=1, inplace=True)

stars.loc[:, "target"] = stars.target.astype(bool)


# ## Inicie sua análise da parte 2 a partir daqui

# In[194]:


# Sua análise da parte 2 começa aqui.
stars.head(2)


# In[50]:


#Questão 4

df4 = stars.query('target == False').mean_profile
false_pulsar_mean_profile_standardized = (df4 - df4.mean()) / df4.std()
print(tuple(np.around(ECDF(false_pulsar_mean_profile_standardized)(sct.norm.ppf([0.80, 0.90, 0.95])).round(3),decimals=3)))


# In[61]:


#Questão 4 (Performance)
df4 = stars.query('target == False').mean_profile
false_pulsar_mean_profile_standardized = (df4 - df4.mean()) / df4.std()
cdf_emp = ECDF(false_pulsar_mean_profile_standardized)
quantis_dist_norm = sct.norm.ppf([0.80, 0.90, 0.95])
probabilidade_quantis = np.around(cdf_emp(quantis_dist_norm), 3)
tuple(probabilidade_quantis)


# In[243]:


#Questão 5

Q1 = np.quantile(false_pulsar_mean_profile_standardized, .25) - sct.norm.ppf(.25)
Q2 = np.quantile(false_pulsar_mean_profile_standardized, .50) - sct.norm.ppf(.50)
Q3 = np.quantile(false_pulsar_mean_profile_standardized, .75) - sct.norm.ppf(.75)
dif5 = np.around((Q1, Q2, Q3),decimals=3)
print(dif5)


# In[56]:


#Questão 5 (Performance)
quartis = np.quantile(false_pulsar_mean_profile_standardized, [.25,.50,.75])
dist_norm = sct.norm.ppf([.25,.50,.75])
dif_quartis_dist = np.around(quartis - dist_norm, 3)
tuple(dif_quartis_dist)


# ## Questão 4
# 
# Considerando a variável `mean_profile` de `stars`:
# 
# 1. Filtre apenas os valores de `mean_profile` onde `target == 0` (ou seja, onde a estrela não é um pulsar).
# 2. Padronize a variável `mean_profile` filtrada anteriormente para ter média 0 e variância 1.
# 
# Chamaremos a variável resultante de `false_pulsar_mean_profile_standardized`.
# 
# Encontre os quantis teóricos para uma distribuição normal de média 0 e variância 1 para 0.80, 0.90 e 0.95 através da função `norm.ppf()` disponível em `scipy.stats`.
# 
# Quais as probabilidade associadas a esses quantis utilizando a CDF empírica da variável `false_pulsar_mean_profile_standardized`? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[197]:


def q4():
    df4 = stars.query('target == False').mean_profile
    false_pulsar_mean_profile_standardized = (df4 - df4.mean()) / df4.std()
    cdf_emp = ECDF(false_pulsar_mean_profile_standardized)
    quantis_dist_norm = sct.norm.ppf([0.80, 0.90, 0.95])
    probabilidade_quantis = np.around(cdf_emp(quantis_dist_norm), 3)
    return tuple(probabilidade_quantis)


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?

# ## Questão 5
# 
# Qual a diferença entre os quantis Q1, Q2 e Q3 de `false_pulsar_mean_profile_standardized` e os mesmos quantis teóricos de uma distribuição normal de média 0 e variância 1? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[198]:


def q5():
    quartis = np.quantile(false_pulsar_mean_profile_standardized, [.25,.50,.75])
    dist_norm = sct.norm.ppf([.25,.50,.75])
    dif_quartis_dist = np.around(quartis - dist_norm, 3)
    return tuple(dif_quartis_dist)


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?
# * Curiosidade: alguns testes de hipóteses sobre normalidade dos dados utilizam essa mesma abordagem.

# In[ ]:




