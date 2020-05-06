#!/usr/bin/env python
# coding: utf-8

# # Desafio 4
# 
# Neste desafio, vamos praticar um pouco sobre testes de hipóteses. Utilizaremos o _data set_ [2016 Olympics in Rio de Janeiro](https://www.kaggle.com/rio2016/olympic-games/), que contém dados sobre os atletas das Olimpíadas de 2016 no Rio de Janeiro.
# 
# Esse _data set_ conta com informações gerais sobre 11538 atletas como nome, nacionalidade, altura, peso e esporte praticado. Estaremos especialmente interessados nas variáveis numéricas altura (`height`) e peso (`weight`). As análises feitas aqui são parte de uma Análise Exploratória de Dados (EDA).
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import statsmodels.api as sm
import seaborn as sns


# In[3]:


# %matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[4]:


athletes = pd.read_csv("athletes.csv")


# In[5]:


def get_sample(df, col_name, n=100, seed=42):
    """Get a sample from a column of a dataframe.
    
    It drops any numpy.nan entries before sampling. The sampling
    is performed without replacement.
    
    Example of numpydoc for those who haven't seen yet.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Source dataframe.
    col_name : str
        Name of the column to be sampled.
    n : int
        Sample size. Default is 100.
    seed : int
        Random seed. Default is 42.
    
    Returns
    -------
    pandas.Series
        Sample of size n from dataframe's column.
    """
    np.random.seed(seed)
    
    random_idx = np.random.choice(df[col_name].dropna().index, size=n, replace=False)
    
    return df.loc[random_idx, col_name]


# ## Inicia sua análise a partir daqui

# In[6]:


# Sua análise começa aqui.
athletes.shape


# In[7]:


athletes.head(2)


# In[8]:


athletes.info()


# In[9]:


athletes.isna().sum()


# In[10]:


athletes.describe()


# In[11]:


# Selecionando a amostra de 3000 observações da coluna height

height_3000 = get_sample(athletes, 'height', n=3000, seed=42)


# In[12]:


# sm.qqplot(height_3000, fit=True, line="45");


# In[13]:


# sns.distplot(height_3000, bins=25);


# In[14]:


sct.shapiro(height_3000)


# In[15]:


shapiro_stat, shapiro_pvalue = sct.shapiro(height_3000)


# In[16]:


if shapiro_pvalue > 0.05:
    print('Com 95% de confiança, os dados são similares a uma distribição normal')
else:
    print('Com 95% de confiança, os dados NÃO são similares a uma distribição normal')


# In[17]:


sct.jarque_bera(height_3000)


# In[18]:


jarque_stat, jarque_pvalue = sct.jarque_bera(height_3000)


# In[19]:


if jarque_pvalue > 0.05:
    print('Com 95% de confiança, os dados são similares a uma distribição normal')
else:
    print('Com 95% de confiança, os dados NÃO são similares a uma distribição normal')


# In[20]:


# Selecionando a amostra de 3000 observações da coluna weight

weight_3000 = get_sample(athletes, 'weight', n=3000, seed=42)


# In[21]:


# sm.qqplot(weight_3000, fit=True, line="45");


# In[22]:


# sns.distplot(weight_3000, bins=25);


# In[23]:


sct.normaltest(weight_3000)


# In[24]:


normal_stat, normal_pvalue = sct.normaltest(weight_3000)


# In[25]:


if normal_pvalue > 0.05:
    print('Com 95% de confiança, os dados são similares a uma distribição normal')
else:
    print('Com 95% de confiança, os dados NÃO são similares a uma distribição normal')


# In[26]:


# transformação logarítmica da amostra de 3000 observações da coluna weight

weight_3000_log = np.log(weight_3000)


# In[27]:


# sm.qqplot(weight_3000_log, fit=True, line="45");


# In[28]:


# sns.distplot(weight_3000_log, bins=25);


# In[29]:


sct.normaltest(weight_3000_log)


# In[30]:


normal_stat_log, normal_pvalue_log = sct.normaltest(weight_3000_log)


# In[31]:


if normal_pvalue_log > 0.05:
    print('Com 95% de confiança, os dados são similares a uma distribição normal')
else:
    print('Com 95% de confiança, os dados NÃO são similares a uma distribição normal')


# In[32]:


# Filtrando todos atletas brasileiros, norte-americanos e canadenses as observações da coluna height

height_bra = athletes.query('nationality in "BRA"')['height'].dropna()
height_can = athletes.query('nationality in "CAN"')['height'].dropna()
height_usa = athletes.query('nationality in "USA"')['height'].dropna()


# In[33]:


"""
bra = sm.qqplot(height_bra, fit=True, line="45")
can = sm.qqplot(height_can, fit=True, line="45")
usa = sm.qqplot(height_usa, fit=True, line="45")
"""


# In[34]:


"""
sns.distplot(height_bra, bins=25, hist=False)
sns.distplot(height_can, bins=25, hist=False)
sns.distplot(height_usa, bins=25, hist=False)
plt.show()
"""


# In[35]:


print('A média de altura do Brasil:', height_bra.mean().round(3))
print('A média de altura do EUA:', height_usa.mean().round(3))
print('A média de altura do Canada:', height_can.mean().round(3))


# In[60]:


sct.ttest_ind(height_bra, height_usa, equal_var=False)


# In[61]:


ttest_stat1, ttest_pvalue1 = sct.ttest_ind(height_bra, height_usa, equal_var=False)


# In[62]:


if ttest_pvalue1 > 0.05:
    print('Com 95% de confiança, as médias das alturas de Brasil e EUA são estatisticamente iguais')
else:
    print('Com 95% de confiança, as médias das alturas de Brasil e EUA NÃO são estatisticamente iguais')


# In[63]:


sct.ttest_ind(height_bra, height_can, equal_var=False)


# In[64]:


ttest_stat2, ttest_pvalue2 = sct.ttest_ind(height_bra, height_can, equal_var=False)


# In[65]:


if ttest_pvalue2 > 0.05:
    print('Com 95% de confiança, as médias das alturas de Brasil e Canada são estatisticamente iguais')
else:
    print('Com 95% de confiança, as médias das alturas de Brasil e Canada NÃO são estatisticamente iguais')


# In[56]:


sct.ttest_ind(height_usa, height_can, equal_var=False)


# In[51]:


ttest_stat3, ttest_pvalue3 = sct.ttest_ind(height_usa, height_can)


# In[44]:


if ttest_pvalue3 > 0.05:
    print('Com 95% de confiança, as médias das alturas de EUA e Canada são estatisticamente iguais')
else:
    print('Com 95% de confiança, as médias das alturas de EUA e Canada NÃO são estatisticamente iguais')


# In[45]:


float(ttest_pvalue3.round(8))


# In[49]:





# In[ ]:





# ## Questão 1
# 
# Considerando uma amostra de tamanho 3000 da coluna `height` obtida com a função `get_sample()`, execute o teste de normalidade de Shapiro-Wilk com a função `scipy.stats.shapiro()`. Podemos afirmar que as alturas são normalmente distribuídas com base nesse teste (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).

# In[6]:


def q1():
    height_3000 = get_sample(athletes, 'height', n=3000, seed=42)
    shapiro_stat, shapiro_pvalue = sct.shapiro(height_3000)
    return bool(shapiro_pvalue > 0.05)


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Plote o qq-plot para essa variável e a analise.
# * Existe algum nível de significância razoável que nos dê outro resultado no teste? (Não faça isso na prática. Isso é chamado _p-value hacking_, e não é legal).

# ## Questão 2
# 
# Repita o mesmo procedimento acima, mas agora utilizando o teste de normalidade de Jarque-Bera através da função `scipy.stats.jarque_bera()`. Agora podemos afirmar que as alturas são normalmente distribuídas (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).

# In[7]:


def q2():
    height_3000 = get_sample(athletes, 'height', n=3000, seed=42)
    jarque_stat, jarque_pvalue = sct.jarque_bera(height_3000)
    return bool(jarque_pvalue > 0.05)


# __Para refletir__:
# 
# * Esse resultado faz sentido?

# ## Questão 3
# 
# Considerando agora uma amostra de tamanho 3000 da coluna `weight` obtida com a função `get_sample()`. Faça o teste de normalidade de D'Agostino-Pearson utilizando a função `scipy.stats.normaltest()`. Podemos afirmar que os pesos vêm de uma distribuição normal ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).

# In[8]:


def q3():
    weight_3000 = get_sample(athletes, 'weight', n=3000, seed=42)
    normal_stat, normal_pvalue = sct.normaltest(weight_3000)
    return bool(normal_pvalue > 0.05)


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Um _box plot_ também poderia ajudar a entender a resposta.

# ## Questão 4
# 
# Realize uma transformação logarítmica em na amostra de `weight` da questão 3 e repita o mesmo procedimento. Podemos afirmar a normalidade da variável transformada ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).

# In[9]:


def q4():
    weight_3000 = get_sample(athletes, 'weight', n=3000, seed=42)
    weight_3000_log = np.log(weight_3000)
    normal_stat_log, normal_pvalue_log = sct.normaltest(weight_3000_log)
    return bool(normal_pvalue_log > 0.05)


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Você esperava um resultado diferente agora?

# > __Para as questão 5 6 e 7 a seguir considere todos testes efetuados ao nível de significância de 5%__.

# ## Questão 5
# 
# Obtenha todos atletas brasileiros, norte-americanos e canadenses em `DataFrame`s chamados `bra`, `usa` e `can`,respectivamente. Realize um teste de hipóteses para comparação das médias das alturas (`height`) para amostras independentes e variâncias diferentes com a função `scipy.stats.ttest_ind()` entre `bra` e `usa`. Podemos afirmar que as médias são estatisticamente iguais? Responda com um boolean (`True` ou `False`).

# In[10]:


def q5():
    height_bra = athletes.query('nationality in "BRA"')['height'].dropna()
    height_usa = athletes.query('nationality in "USA"')['height'].dropna()
    ttest_stat1, ttest_pvalue1 = sct.ttest_ind(height_bra, height_usa, equal_var=False)
    return bool(ttest_pvalue1 > 0.05)


# ## Questão 6
# 
# Repita o procedimento da questão 5, mas agora entre as alturas de `bra` e `can`. Podemos afimar agora que as médias são estatisticamente iguais? Reponda com um boolean (`True` ou `False`).

# In[11]:


def q6():
    height_bra = athletes.query('nationality in "BRA"')['height'].dropna()
    height_can = athletes.query('nationality in "CAN"')['height'].dropna()
    ttest_stat2, ttest_pvalue2 = sct.ttest_ind(height_bra, height_can, equal_var=False)
    return bool(ttest_pvalue2 > 0.05)


# ## Questão 7
# 
# Repita o procedimento da questão 6, mas agora entre as alturas de `usa` e `can`. Qual o valor do p-valor retornado? Responda como um único escalar arredondado para oito casas decimais.

# In[12]:


def q7():
    height_can = athletes.query('nationality in "CAN"')['height'].dropna()
    height_usa = athletes.query('nationality in "USA"')['height'].dropna()
    ttest_stat3, ttest_pvalue3 = sct.ttest_ind(height_usa, height_can, equal_var=False)
    return float(ttest_pvalue3.round(8))


# __Para refletir__:
# 
# * O resultado faz sentido?
# * Você consegue interpretar esse p-valor?
# * Você consegue chegar a esse valor de p-valor a partir da variável de estatística?
