#!/usr/bin/env python
# coding: utf-8

# # **Introdução**

# # 1. Importando Bibliotecas

# In[4]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn as sk


# # 2. Leitura e Pré-processamento de Dados

# ### 2.1 Dataframe recebe CSV

# In[25]:


df = pd.read_csv("datasets/cancer-death-rates-by-age.csv")


# In[26]:


df


# ### Criando um array com o nome de todas as entidades do Dataframe

# In[27]:


valores_unicos = df['Entity'].unique()
valores_unicos


# ### Separando as entidades por regiões

# In[28]:


#regions

array_regions = []

for palavra in valores_unicos:
    if 'Region' in palavra or 'WB' in palavra or 'WHO' in palavra:
        array_regions.append(palavra)

df_regions= pd.DataFrame(array_regions, columns=['Regions'])

print(df_regions)


# ### Retirando as entidades que não são países

# In[56]:


#retirando

# Obter os valores únicos da coluna 'Entity' que atendem ao critério
valores_unicos = df['Entity'].unique()
array_not_a_country = []

for palavra in valores_unicos:
    if 'G20' in palavra or 'WB' in palavra or 'WHO' in palavra or 'Income' in palavra:
        array_not_a_country.append(palavra)

df_not_a_country = pd.DataFrame(array_regions, columns=['Regions'])


# In[57]:


# Eliminar os valores presentes em df_regions do DataFrame df
df = df[~df['Entity'].isin(df_regions['Regions'])]

# Exibir o DataFrame após a eliminação
print("\nDataFrame após a eliminação:")
df


# ### Agrupamento dos países de acordo com a renda

# In[29]:


#países de acordo com a renda
array_country_income = []

for palavra in valores_unicos:
    if 'Income' in palavra:
        array_country_income.append(palavra)

df_country_income= pd.DataFrame(array_country_income, columns=['Country Income'])

print(df_country_income)


# ## 2.2 Limpeza e Transformação de Dados Necessários

# *Para facilitar a manipulação das colunas, abreviamos Deaths - Neoplasms - Sex: Both - Age para **DNS**
# 
# 
# 

# In[30]:


df = df.rename(columns={ "Deaths - Neoplasms - Sex: Both - Age: Under 5 (Rate)": "| DNS < 5 |",
                "Deaths - Neoplasms - Sex: Both - Age: Age-standardized (Rate)": "DNS Padronizado",
                "Deaths - Neoplasms - Sex: Both - Age: All Ages (Rate)": "| DNS all ages |",
                "Deaths - Neoplasms - Sex: Both - Age: 70+ years (Rate)" : "DNS > 70",
                "Deaths - Neoplasms - Sex: Both - Age: 5-14 years (Rate)" : "| 5 < DNS < 14 |",
                "Deaths - Neoplasms - Sex: Both - Age: 50-69 years (Rate)" : "50 < DNS < 69",
                "Deaths - Neoplasms - Sex: Both - Age: 15-49 years (Rate)" : "| 15 < DNS < 49 |" })


# *Alinhamento das colunas*
# 
# 

# In[31]:


df.head().style.set_table_styles([dict(selector='th', props=[('text-align', 'center')]),
                                    dict(selector='td', props=[('text-align', 'center')])])


# In[ ]:


#Verificando a existência de valores NaN  e se isso vai interferir
df.info()


# Não há valores NaN nas colunas DNS. Há apenas na coluna Code

# Lembrar:
# 
# 
# *   Verificar o que não é país dentro da coluna Entity (já foi feito)
# *   Alterar a ordem das colunas. Começar com < 5, e assim por diante
# 
# 

# ## 2.3 Medidas de Centralidade

# ### Medianas

# In[33]:


#medianas das taxas de mortalidade de acordo com o país e o grupo etário

colunas = df.columns[3:]

medians = {}

for coluna in colunas:
    medians[coluna] = df.groupby('Entity')[coluna].median().sort_values()

    # Exibir as medianas calculadas para cada coluna
for coluna, medianas in medians.items():
    print(f"Mediana da coluna '{coluna}':")
    print(medianas)
    print()


# In[48]:


#desvio padrão

std = {}
for coluna in colunas:
    std[coluna] = df.groupby('Entity')[coluna].std().sort_values(ascending=False)

    # Exibir as medianas calculadas para cada coluna
for coluna, stds in std.items():
    print(f"Mediana da coluna '{coluna}':")
    print(stds)
    print()


# ### Desvio Padrão

# In[49]:


# 10 Países com maiores taxas de mortalidade
for coluna, stds in std.items():  
  print(stds.tail(5))
  print()


# # 3. Exploração através de gráficos

# ## Gráficos de barra

# In[44]:


df_latam = df.loc[(df['Entity'] == 'Latin America & Caribbean')]


# ### Monaco

# #### Taxa de mortalidade ao longo dos anos em Monaco

# In[33]:


df_monaco = df.loc[(df['Entity']=='Monaco')]
sns.barplot(data=df_monaco, x='Year', y='DNS > 70').set(title="Relação das Mortes para maiores de 70 anos")
plt.xlabel('Ano')
plt.ylabel('Contagem de Mortes')

# Rotacionar os rótulos dos anos e diminuir a fonte
plt.xticks(rotation=45, fontsize=8)

plt.show()


# In[34]:


df_monaco = df.loc[(df['Entity']=='Monaco')]
sns.barplot(data=df_monaco, x='Year', y='| DNS < 5 |').set(title="Relação das Mortes para menores de 5 anos")
plt.xlabel('Ano')
plt.ylabel('Contagem de Mortes')

# Rotacionar os rótulos dos anos e diminuir a fonte
plt.xticks(rotation=45, fontsize=8)

plt.show()


# 

# ### South Sudan

# #### Taxa de mortalidade ao longo dos anos em South Sudan

# In[43]:


df_South_Sudan = df.loc[(df['Entity']=='South Sudan')]
sns.barplot(data=df_South_Sudan, x='Year', y='DNS > 70').set(title="Relação das Mortes para maiores de 70 anos")
plt.ylabel('Contagem de Mortes')
plt.xlabel('Ano')

# Rotacionar os rótulos dos anos e diminuir a fonte
plt.xticks(rotation=45, fontsize=8)

plt.show()


# ### Russia 

# #### Taxa de mortalidade ao longo nos anos em Russia

# In[36]:


df_Russia = df.loc[(df['Entity']=='Russia')]
sns.barplot(data=df_Russia, x='Year', y='DNS > 70').set(title="Relação das Mortes para maiores de 70 anos")
plt.xlabel('Ano')
plt.ylabel('Contagem de Mortes')

# Rotacionar os rótulos dos anos e diminuir a fonte
plt.xticks(rotation=45, fontsize=8)

plt.show()


# ## Boxplots

# ### Russia

# In[37]:


sns.boxplot(data=df_Russia, x='Entity', y='DNS > 70').set(title="Relação das Mortes para maiores de 70 anos")


# ### BRICS

# In[37]:


df_filtered_brics = df.loc[((df['Entity'] == 'Brazil') | (df['Entity'] == 'Russia' ) | (df['Entity'] == 'India') | (df['Entity'] == 'China') | (df['Entity'] == 'South Africa'))]
sns.boxplot(data=df_filtered_brics , x='Entity', y='DNS > 70')
plt.title('Taxa de mortalidade (pessoas com mais de 70 anos) dos países do BRICS',fontsize=10)


# In[38]:


df_filtered_brics = df.loc[((df['Entity'] == 'Brazil') | (df['Entity'] == 'Russia' ) | (df['Entity'] == 'India') | (df['Entity'] == 'China') | (df['Entity'] == 'South Africa'))]
sns.boxplot(data=df_filtered_brics , x='Entity', y='50 < DNS < 69')
plt.title('Taxa de mortalidade (pessoas entre 50 e 69 anos) dos países do BRICS',fontsize=10)


# In[39]:



df_filtered_brics = df.loc[((df['Entity'] == 'Brazil') | (df['Entity'] == 'Russia' ) | (df['Entity'] == 'India') | (df['Entity'] == 'China') | (df['Entity'] == 'South Africa'))]
sns.boxplot(data=df_filtered_brics , x='Entity', y='| 5 < DNS < 14 |')
plt.title('Taxa de mortalidade (pessoas entre 5 e 14 anos) dos países do BRICS',fontsize=10)


# In[40]:


df_filtered_brics = df.loc[((df['Entity'] == 'Brazil') | (df['Entity'] == 'Russia' ) | (df['Entity'] == 'India') | (df['Entity'] == 'China') | (df['Entity'] == 'South Africa'))]
sns.boxplot(data=df_filtered_brics , x='Entity', y='| DNS < 5 |')
plt.title('Taxa de mortalidade (crianças menores de 5 anos) dos países do BRICS',fontsize=10)


# # 4. Perguntas

# Rascunho:
# 
# 
# * Qual a faixa etária que mais morre nesses países? (responder através de gráfico de barras)
# * Qual o país mais constante? (responder através de gráficos do k-means)
# * Qual o país com maior queda de casos nos últimos 5 anos?
# * Em qual ano mais países tiveram altas em suas taxas?
# * Qual o país com o maior investimento público em saúde? 
# * E qual o país com o menor investimento?
# 
# 
# 
# 
# 

# ## Etapa de Clusterização

# In[41]:


from sklearn.cluster import KMeans


# In[42]:


kbrics = df_filtered_brics[['DNS > 70', '50 < DNS < 69']]

# Definir o número de clusters
k = 3

# Executar o algoritmo K-Means
kmeans = KMeans(n_clusters=k)
kmeans.fit(kbrics)
centroides = kmeans.cluster_centers_

# Adicionar as colunas de cluster ao DataFrame original
df_filtered_brics['Cluster'] = kmeans.labels_

# Visualizar os resultados do cluster
sns.scatterplot(data=df_filtered_brics, x='DNS > 70', y='50 < DNS < 69', hue='Cluster', palette='bright')

# Plotar os centroides
plt.scatter(centroides[:, 0], centroides[:, 1], marker='X', color='black', s=100)

# Configurações adicionais do gráfico
plt.xlabel('DNS > 70')
plt.ylabel('50 < DNS < 69')
plt.title('1.1 Gráfico de dispersão com centroides')

# Exibir o gráfico
plt.show()


# 

# # 5. Conclusões

# ## Ideias

# In[ ]:



''' Quero plotar mapa
# Dados
data = {
    'Country': ['Brazil', 'Argentina', 'Chile'],
    'Value': [10, 20, 30]
}

# GeoJSON com as geometrias dos países
geojson = 'caminho/para/arquivo.geojson'

# Plotagem do mapa de cloropleta
fig = px.choropleth_mapbox(data_frame=data, geojson=geojson, locations='Country', color='Value',
                           mapbox_style='carto-positron', center={'lat': 0, 'lon': 0}, zoom=2)

# Exibição do mapa
fig.show()
'''

