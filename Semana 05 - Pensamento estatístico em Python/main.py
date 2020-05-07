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

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import statsmodels.api as sm
import seaborn as sns


# In[67]:


#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[2]:


athletes = pd.read_csv("athletes.csv")


# In[3]:


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

# In[70]:


# Sua análise começa aqui.
athletes.shape


# In[71]:


# Uma breve exibição dos dados
athletes.head(2)


# In[72]:


# Exibição de algumas informações do dataset como colunas e tipos de variáveis
athletes.info()


# In[73]:


# Levantamemento da quantidade de variáveis nulas
athletes.isna().sum()


# In[74]:


# Um breve resumo estatistico das colunas
athletes.describe()


# In[4]:


# Selecionando a amostra de 3000 observações da coluna height

height_3000 = get_sample(athletes, 'height', n=3000, seed=42)


# In[76]:


# QQPLOT da amostra de 3000 observações da coluna height

sm.qqplot(height_3000, fit=True, line="45");


# In[77]:


# Histograma da amostra de 3000 observações da coluna height

sns.distplot(height_3000, bins=25);


# In[78]:


# Aplicação do Teste de Shapiro-Wilk na amostra de 3000 observações da coluna height

sct.shapiro(height_3000)


# In[79]:


shapiro_stat, shapiro_pvalue = sct.shapiro(height_3000)


# In[80]:


# Resultado do Teste de Shapiro-Wilk na amostra de 3000 observações da coluna height
# Comparando o seu resuldado com o alpha é possivel saber se a amostra tem uma distribuição normal

if shapiro_pvalue > 0.05:
    print('Com 95% de confiança, os dados são similares a uma distribição normal')
else:
    print('Com 95% de confiança, os dados NÃO são similares a uma distribição normal')


# In[13]:


# Aplicação do Teste de Jarque–Bera tambem na amostra de 3000 observações da coluna height

sct.jarque_bera(height_3000)


# In[82]:


jarque_stat, jarque_pvalue = sct.jarque_bera(height_3000)


# In[83]:


# Resultado do Teste de Jarque–Bera na amostra de 3000 observações da coluna height
# Comparando o seu resuldado com o alpha é possivel saber se a amostra tem uma distribuição normal
# Tanto Jarque–Bera quanto Shapiro-Wilk, comparando o p-valor com alpha podemos chegar nesta conclusão

if jarque_pvalue > 0.05:
    print('Com 95% de confiança, os dados são similares a uma distribição normal')
else:
    print('Com 95% de confiança, os dados NÃO são similares a uma distribição normal')


# In[5]:


# Selecionando a amostra de 3000 observações da coluna weight

weight_3000 = get_sample(athletes, 'weight', n=3000, seed=42)


# In[85]:


# QQPLOT da amostra de 3000 observações da coluna weight

sm.qqplot(weight_3000, fit=True, line="45");


# In[86]:


# Histograma da amostra de 3000 observações da coluna weight

sns.distplot(weight_3000, bins=25);


# In[9]:


# Aplicação do Normal Test na amostra de 3000 observações da coluna weight

sct.normaltest(weight_3000)


# In[10]:


normaltest_weight_3000 = sct.normaltest(weight_3000)


# In[11]:


# Resultado do Normal Test na amostra de 3000 observações da coluna weight
# Comparando o seu resuldado com o alpha é possivel saber se a amostra tem uma distribuição normal

if normaltest_weight_3000.pvalue > 0.05:
    print('Com 95% de confiança, os dados são similares a uma distribição normal')
else:
    print('Com 95% de confiança, os dados NÃO são similares a uma distribição normal')


# In[15]:


# transformação logarítmica da amostra de 3000 observações da coluna weight

weight_3000_log = np.log(weight_3000)


# In[91]:


# QQPLOT da amostra de 3000 observações da coluna weight após a transformação logarítmica

sm.qqplot(weight_3000_log, fit=True, line="45");


# In[92]:


# Histograma da amostra de 3000 observações da coluna weight após a transformação logarítmica

sns.distplot(weight_3000_log, bins=25);


# In[93]:


# Aplicação do Normal Test na amostra de 3000 observações da coluna weight após a transformação logarítmica

sct.normaltest(weight_3000_log)


# In[16]:


normaltest_weight_3000_log = sct.normaltest(weight_3000_log)


# In[17]:


# Resultado do Normal Test na amostra de 3000 observações da coluna weight após a transformação logarítmica
# Comparando o seu resuldado com o alpha é possivel saber se a amostra tem uma distribuição normal
# Em alguns casos a transformação logarítmica pode levar a uma distribuição normal

if normaltest_weight_3000_log.pvalue > 0.05:
    print('Com 95% de confiança, os dados são similares a uma distribição normal')
else:
    print('Com 95% de confiança, os dados NÃO são similares a uma distribição normal')


# In[19]:


# Filtrando todos atletas brasileiros, norte-americanos e canadenses as observações da coluna height

height_bra = athletes.query('nationality in "BRA"')['height'].dropna()
height_can = athletes.query('nationality in "CAN"')['height'].dropna()
height_usa = athletes.query('nationality in "USA"')['height'].dropna()


# In[97]:


# QQPLOT do atletas brasileiros, norte-americanos e canadenses para as observações da coluna height

bra = sm.qqplot(height_bra, fit=True, line="45")
can = sm.qqplot(height_can, fit=True, line="45")
usa = sm.qqplot(height_usa, fit=True, line="45")


# In[98]:


# Histograma do atletas brasileiros, norte-americanos e canadenses para as observações da coluna height

sns.distplot(height_bra, bins=25, hist=False)
sns.distplot(height_can, bins=25, hist=False)
sns.distplot(height_usa, bins=25, hist=False)
plt.show()


# In[99]:


# Verificando a média da coluna height para cada pais

print('A média de altura do Brasil:', height_bra.mean().round(3))
print('A média de altura do EUA:', height_usa.mean().round(3))
print('A média de altura do Canada:', height_can.mean().round(3))


# In[21]:


# Aplicação do teste ttest_ind, que compara dois grupos ( BRASIL e EUA) para identificar semelhanças estatísticas
# Comparando o resultado com o alpha é possivel saber se a amostra tem uma semelhanças estatísticas

sct.ttest_ind(height_bra, height_usa, equal_var=False)


# In[23]:


test_bra_usa = sct.ttest_ind(height_bra, height_usa, equal_var=False)


# In[24]:


if test_bra_usa.pvalue > 0.05:
    print('Com 95% de confiança, as médias das alturas de Brasil e EUA são estatisticamente iguais')
else:
    print('Com 95% de confiança, as médias das alturas de Brasil e EUA NÃO são estatisticamente iguais')


# In[103]:


# Aplicação do teste ttest_ind, que compara dois grupos ( BRASIL e CANADA) para identificar semelhanças estatísticas
# Comparando o resultado com o alpha é possivel saber se a amostra tem uma semelhanças estatísticas

sct.ttest_ind(height_bra, height_can, equal_var=False)


# In[25]:


test_bra_can = sct.ttest_ind(height_bra, height_can, equal_var=False)


# In[26]:


if test_bra_can.pvalue > 0.05:
    print('Com 95% de confiança, as médias das alturas de Brasil e Canada são estatisticamente iguais')
else:
    print('Com 95% de confiança, as médias das alturas de Brasil e Canada NÃO são estatisticamente iguais')


# In[106]:


# Aplicação do teste ttest_ind, que compara dois grupos ( EUA e CANADA) para identificar semelhanças estatísticas 
# Comparando o resultado com o alpha é possivel saber se a amostra tem uma semelhanças estatisticas

sct.ttest_ind(height_usa, height_can, equal_var=False)


# In[27]:


test_usa_can = sct.ttest_ind(height_usa, height_can)


# In[28]:


if test_usa_can.pvalue > 0.05:
    print('Com 95% de confiança, as médias das alturas de EUA e Canada são estatisticamente iguais')
else:
    print('Com 95% de confiança, as médias das alturas de EUA e Canada NÃO são estatisticamente iguais')


# In[29]:


# Exibindo o p-valor do teste ttest_ind

float(test_usa_can.pvalue.round(8))


# ## Questão 1
# 
# Considerando uma amostra de tamanho 3000 da coluna `height` obtida com a função `get_sample()`, execute o teste de normalidade de Shapiro-Wilk com a função `scipy.stats.shapiro()`. Podemos afirmar que as alturas são normalmente distribuídas com base nesse teste (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).

# In[110]:


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

# In[111]:


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

# In[30]:


def q3():
    weight_3000 = get_sample(athletes, 'weight', n=3000, seed=42)
    normaltest_weight_3000 = sct.normaltest(weight_3000)
    return bool(normaltest_weight_3000.pvalue > 0.05)


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Um _box plot_ também poderia ajudar a entender a resposta.

# ## Questão 4
# 
# Realize uma transformação logarítmica em na amostra de `weight` da questão 3 e repita o mesmo procedimento. Podemos afirmar a normalidade da variável transformada ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).

# In[113]:


def q4():
    weight_3000 = get_sample(athletes, 'weight', n=3000, seed=42)
    weight_3000_log = np.log(weight_3000)
    normaltest_weight_3000_log = sct.normaltest(weight_3000_log)
    return bool(normaltest_weight_3000_log.pvalue > 0.05)


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Você esperava um resultado diferente agora?

# > __Para as questão 5 6 e 7 a seguir considere todos testes efetuados ao nível de significância de 5%__.

# ## Questão 5
# 
# Obtenha todos atletas brasileiros, norte-americanos e canadenses em `DataFrame`s chamados `bra`, `usa` e `can`,respectivamente. Realize um teste de hipóteses para comparação das médias das alturas (`height`) para amostras independentes e variâncias diferentes com a função `scipy.stats.ttest_ind()` entre `bra` e `usa`. Podemos afirmar que as médias são estatisticamente iguais? Responda com um boolean (`True` ou `False`).

# In[31]:


def q5():
    height_bra = athletes.query('nationality in "BRA"')['height'].dropna()
    height_usa = athletes.query('nationality in "USA"')['height'].dropna()
    test_bra_usa = sct.ttest_ind(height_bra, height_usa, equal_var=False)
    return bool(test_bra_usa.pvalue > 0.05)


# ## Questão 6
# 
# Repita o procedimento da questão 5, mas agora entre as alturas de `bra` e `can`. Podemos afimar agora que as médias são estatisticamente iguais? Reponda com um boolean (`True` ou `False`).

# In[32]:


def q6():
    height_bra = athletes.query('nationality in "BRA"')['height'].dropna()
    height_can = athletes.query('nationality in "CAN"')['height'].dropna()
    test_bra_can = sct.ttest_ind(height_bra, height_can, equal_var=False)
    return bool(test_bra_can.pvalue > 0.05)


# ## Questão 7
# 
# Repita o procedimento da questão 6, mas agora entre as alturas de `usa` e `can`. Qual o valor do p-valor retornado? Responda como um único escalar arredondado para oito casas decimais.

# In[33]:


def q7():
    height_can = athletes.query('nationality in "CAN"')['height'].dropna()
    height_usa = athletes.query('nationality in "USA"')['height'].dropna()
    test_usa_can = sct.ttest_ind(height_usa, height_can, equal_var=False)
    return float(test_usa_can.pvalue.round(8))


# __Para refletir__:
# 
# * O resultado faz sentido?
# * Você consegue interpretar esse p-valor?
# * Você consegue chegar a esse valor de p-valor a partir da variável de estatística?

# In[ ]:




