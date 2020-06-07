#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk
from sklearn.preprocessing import (KBinsDiscretizer, OneHotEncoder, StandardScaler)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from scipy.stats import zscore
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import (CountVectorizer, TfidfTransformer, TfidfVectorizer)
from sklearn.compose import ColumnTransformer


# In[2]:


# Algumas configurações para o matplotlib.
#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[5]:


countries = pd.read_csv("countries.csv")


# In[6]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head(5)


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# In[7]:


# Sua análise começa aqui.
# Ajustando as variáveis Country e Region possuem espaços a mais no começo e no final da string
countries["Country"] = countries["Country"].str.strip()
countries["Region"] = countries["Region"].str.strip()


# In[8]:


countries.head()


# In[9]:


# Verificar Nulos
countries.isna().sum() 


# In[10]:


'''
df['DataFrame Column'] = df['DataFrame Column'].astype(float)
 df['DataFrame Column'] = df['DataFrame Column']..str.replace(',','.'), alterar ',' por '.'
 df['DataFrame Column'] = pd.to_numeric(df['DataFrame Column'],errors='coerce'), If ‘coerce’, then invalid parsing will be set as NaN.
'''

countries['Pop_density'] = pd.to_numeric(countries['Pop_density'].str.replace(',','.'))
countries['Coastline_ratio'] = pd.to_numeric(countries['Coastline_ratio'].str.replace(',','.'))
countries['Net_migration'] = pd.to_numeric(countries['Net_migration'].str.replace(',','.'),errors='coerce')
countries['Infant_mortality'] = pd.to_numeric(countries['Infant_mortality'].str.replace(',','.'),errors='coerce')
countries['Literacy'] = pd.to_numeric(countries['Literacy'].str.replace(',','.'),errors='coerce')
countries['Phones_per_1000'] = pd.to_numeric(countries['Phones_per_1000'].str.replace(',','.'),errors='coerce')
countries['Arable'] = pd.to_numeric(countries['Arable'].str.replace(',','.'),errors='coerce')
countries['Crops'] = pd.to_numeric(countries['Crops'].str.replace(',','.'),errors='coerce')
countries['Other'] = pd.to_numeric(countries['Other'].str.replace(',','.'),errors='coerce')
countries['Climate'] = pd.to_numeric(countries['Climate'].str.replace(',','.'),errors='coerce')
countries['Birthrate'] = pd.to_numeric(countries['Birthrate'].str.replace(',','.'),errors='coerce')
countries['Deathrate'] = pd.to_numeric(countries['Deathrate'].str.replace(',','.'),errors='coerce')
countries['Agriculture'] = pd.to_numeric(countries['Agriculture'].str.replace(',','.'),errors='coerce')
countries['Industry'] = pd.to_numeric(countries['Industry'].str.replace(',','.'),errors='coerce')
countries['Service'] = pd.to_numeric(countries['Service'].str.replace(',','.'),errors='coerce')


# In[11]:


countries.info()


# In[12]:


countries.head()


# In[13]:


# Verificar Regiões do dataset
regions =  countries['Region'].unique()
sorted(list(regions))


# In[14]:


# Discretizando a variável Pop_density em 10 intervalos, quantos países se encontram acima do 90º percentil.

est = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
X = countries[['Pop_density']]
y = countries.Country
est_quantile_90 = est.fit_transform(X,y) >= 9.
int(np.sum(est_quantile_90))


# In[15]:


# Países se encontram acima do 90º percentil sem usar função KBinsDiscretizer
countries_density = countries[['Country', 'Pop_density']]
quantile_90 = countries['Pop_density'].quantile(.9)
countries_90 = countries_density.query('Pop_density > @quantile_90')
int(countries_90.shape[0])


# In[26]:


# Codificando as variáveis Region e Climate usando one-hot encoding
 
one_hot_encoder = OneHotEncoder(sparse=False, dtype=np.int)
region_climate = countries[['Region', 'Climate']].dropna()
region_climate_encoded = one_hot_encoder.fit_transform(region_climate)
int(region_climate_encoded.shape[1])


# In[27]:


dummies_encoded = pd.get_dummies(countries[['Region', 'Climate']].fillna('NaN'))
int(dummies_encoded.shape[1])


# In[17]:


# Realiznado um teste
test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[18]:


# Criando um pipeline
# 1 Preencha as variáveis do tipo int64 e float64 com suas respectivas medianas.
# 2 Padronize essas variáveis.
# Qual o valor da variável Arable após o pipeline

var_num = list(countries.select_dtypes(include='number').columns)
df_var_num = countries[var_num]
df_var_num['Arable'].unique()

# Função pipeline para substituir nulos pela média e padronizar valores
num_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                               ('scaler', StandardScaler())])
pipeline_var_num = num_pipeline.fit_transform(df_var_num)
pipeline_median = pd.DataFrame(pipeline_var_num, columns=df_var_num.columns)

# Função Pipeline no test_country
pipeline_test_country = num_pipeline.transform([test_country[2:]])
pipeline_position = pd.DataFrame(pipeline_test_country, columns=df_var_num.columns)
arable = pipeline_position['Arable']
float(arable.round(3))


# In[22]:


# Descubra o número de outliers da variável Net_migration segundo o método do boxplot
net_migration = countries['Net_migration'].dropna()
sns.boxplot(net_migration, palette='Set3',  orient='vertical');


# In[23]:


# Encontrar a faixa normal para encontrar o outliers

quantile1 = net_migration.quantile(0.25)
quantile3 = net_migration.quantile(0.75)
iqr = quantile3 - quantile1

non_outlier_interval_iqr = [quantile1 - 1.5 * iqr, quantile3 + 1.5 * iqr]

print(f"Faixa considerada \"normal\": {non_outlier_interval_iqr}")


# In[24]:


# Encontrando os outliers

outliers_abaixo = net_migration[(net_migration < non_outlier_interval_iqr[0])]
outliers_acima = net_migration[(net_migration > non_outlier_interval_iqr[1])]
outliers_iqr = (int(len(outliers_abaixo)), int(len(outliers_acima)), 'True')
outliers_iqr


# In[16]:


# Analisando o dataset fetch_20newsgroups
categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)


# In[20]:


# Contagem da palavra phone
vectorizer = CountVectorizer()
vectorizer_count = vectorizer.fit_transform(newsgroup.data)
words_vectorizer = vectorizer.vocabulary_
int(vectorizer_count[:, words_vectorizer['phone']].sum())


# In[33]:


# TF-IDF da palavra phone
tfidf = TfidfVectorizer().fit(newsgroup.data)
tfidf_count = tfidf.transform(newsgroup.data)
words_tfidf = tfidf.vocabulary_
float(tfidf_count[:, words_tfidf['phone']].sum().round(3))


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[28]:


def q1():
    regions =  countries['Region'].unique()
    return sorted(list(regions))


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[29]:


def q2():
    est = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
    X = countries[['Pop_density']]
    y = countries.Country
    est_quantile_90 = est.fit_transform(X,y) >= 9.
    return int(np.sum(est_quantile_90))


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[28]:


def q3():
    dummies_encoded = pd.get_dummies(countries[['Region', 'Climate']].fillna('NaN'))
    return int(dummies_encoded.shape[1])


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[31]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[32]:


def q4():
    # Criando um novo DF com as variáveis numéricas
    var_num = list(countries.select_dtypes(include='number').columns)
    df_var_num = countries[var_num]

    # Função pipeline para substituir nulos pela média e padronizar valores
    num_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                   ('scaler', StandardScaler())])
    # Aplicando pipeline no df_var_num
    pipeline_var_num = num_pipeline.fit_transform(df_var_num)
    pipeline_median = pd.DataFrame(pipeline_var_num, columns=df_var_num.columns)

    # Função Pipeline no test_country
    pipeline_test_country = num_pipeline.transform([test_country[2:]])
    pipeline_position = pd.DataFrame(pipeline_test_country, columns=df_var_num.columns)
    arable = pipeline_position['Arable']
    return float(arable.round(3))


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[3]:


def q5():
    # Criando uma série com a coluna Net_migration
    net_migration = countries['Net_migration'].dropna()
    
    # Encontrando os quartis e calculando o iqr
    quantile1 = net_migration.quantile(0.25)
    quantile3 = net_migration.quantile(0.75)
    iqr = quantile3 - quantile1
    
    # Encontrando o intervalo normal
    non_outlier_interval_iqr = [quantile1 - 1.5 * iqr, quantile3 + 1.5 * iqr]

    # Encontrando os outliers acima e abaixo do intervalo normal
    outliers_abaixo = net_migration[(net_migration < non_outlier_interval_iqr[0])]
    outliers_acima = net_migration[(net_migration > non_outlier_interval_iqr[1])]
    return (int(len(outliers_abaixo)), int(len(outliers_acima)), False)


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[21]:


def q6():
    vectorizer = CountVectorizer()
    vectorizer_count = vectorizer.fit_transform(newsgroup.data)
    words_vectorizer = vectorizer.vocabulary_
    return int(vectorizer_count[:, words_vectorizer['phone']].sum())


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[6]:


def q7():
    tfidf = TfidfVectorizer().fit(newsgroup.data)
    tfidf_count = tfidf.transform(newsgroup.data)
    words_tfidf = tfidf.vocabulary_
    return float(tfidf_count[:, words_tfidf['phone']].sum().round(3))


# In[ ]:




