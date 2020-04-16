#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[14]:


import pandas as pd
import numpy as np


# In[15]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[16]:


print(black_friday.head(1))


# In[17]:


print(black_friday.info())


# In[18]:


print(black_friday.describe())


# In[19]:


# 1 - Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

print(black_friday.shape)


# In[20]:


# 2 - Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.
print(len(black_friday.query('Age == "26-35" & Gender == "F"')))


# In[21]:


# 3 - Quantos usuários únicos há no dataset? Responda como um único escalar.
print(len(black_friday.groupby('User_ID')['User_ID']))


# In[22]:


# 4 - Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.
print(black_friday.dtypes)
print(len(black_friday.dtypes.value_counts()))


# In[23]:


# 5 - Qual porcentagem dos registros possui ao menos um valor null ? Responda como um único escalar entre 0 e 1.
print((black_friday.shape[0] - black_friday.dropna().shape[0]) / black_friday.shape[0])


# In[24]:


# 6 - Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.
print(black_friday.isnull().sum().max())


# In[25]:


# 7 - Qual o valor mais frequente (sem contar nulls) em Product_Category_3? Responda como um único escalar.
print(int(black_friday['Product_Category_3'].mode()))


# In[26]:


# 8 - Qual a nova média da variável (coluna) Purchase após sua normalização? Responda como um único escalar.
normalized = black_friday[['Purchase']]
normalized_df = (normalized - normalized.min()) / (normalized.max() - normalized.min())
print(float(normalized_df.mean()))


# In[27]:


# 9 - Quantas ocorrências entre -1 e 1 inclusive existem da variáel Purchase após sua padronização? 
# Responda como um único escalar.
normalized = black_friday[['Purchase']]
normalized_std = (normalized - normalized.mean()) / normalized.std()
print(len(normalized_std.query('Purchase >= -1 & Purchase <= 1')))


# In[28]:


# 10 - Podemos afirmar que se uma observação é null em Product_Category_2 ela também o é em Product_Category_3? 
# Responda com um bool (True, False).
print(len(black_friday.isna().query('Product_Category_2 == True & Product_Category_3 == False')) == False)


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[29]:


def q1():
    q1 = black_friday.shape
    return q1


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[30]:


def q2():
    q2 = len(black_friday.query('Age == "26-35" & Gender == "F"'))
    return int(q2)


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[31]:


def q3():
    q3 = len(black_friday.groupby('User_ID')['User_ID'])
    return int(q3)


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[32]:


def q4():
    q4 = len(black_friday.dtypes.value_counts())
    return int(q4)


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[33]:


def q5():
    q5 = (black_friday.shape[0] - black_friday.dropna().shape[0]) / black_friday.shape[0]
    return float(q5) 


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[34]:


def q6():
    q6 = black_friday.isnull().sum().max()
    return int(q6)


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[35]:


def q7():
    q7 = black_friday['Product_Category_3'].mode()
    return int(q7)


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[36]:


def q8():
    normalized = black_friday[['Purchase']]
    normalized_df = (normalized - normalized.min()) / (normalized.max() - normalized.min())
    q8 = normalized_df.mean()
    return float(q8)


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[37]:


def q9():
    normalized = black_friday[['Purchase']]
    normalized_std = (normalized - normalized.mean()) / normalized.std()
    q9 = len(normalized_std.query('Purchase >= -1 & Purchase <= 1'))
    return int(q9)


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[37]:


def q10():
    q10 = len(black_friday.isna().query('Product_Category_2 == True & Product_Category_3 == False')) == False
    return q10

