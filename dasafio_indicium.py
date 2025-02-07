"""
- DESAFIO DO PROCESSO SELETIVO LIGHTHOUSE - INDICIUM ACADEMY

- Ciência de Dados

- Candidato: Luiz Rodrigo Hamada
"""

"""
Modelo descrito de precificação de aluguéis temporários na cidade de New York utilizando técnicas de Machine Learning.
"""

# Importando bibliotecas

import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import joblib

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

"""
Realizando o carregamento de dados, leitura dos dados iniciais do arquivo 'dados' e verificando a existência de
valores nulos
"""

dados = pd.read_csv('teste_indicium_precificacao.csv')
df = pd.DataFrame(dados)
pd.set_option('display.max_columns', None)
df = df.dropna()
print(df.head())
"""
    id                                              nome  host_id  
0  2595                             Skylit Midtown Castle     2845   
2  3831                   Cozy Entire Floor of Brownstone     4869   
3  5022  Entire Apt: Spacious Studio/Loft by central park     7192   
4  5099         Large Cozy 1 BR Apartment In Midtown East     7322   
5  5121                                   BlissArtsSpace!     7356   

     host_name bairro_group              bairro  latitude  longitude  
0     Jennifer    Manhattan             Midtown  40.75362  -73.98377   
2  LisaRoxanne     Brooklyn        Clinton Hill  40.68514  -73.95976   
3        Laura    Manhattan         East Harlem  40.79851  -73.94399   
4        Chris    Manhattan         Murray Hill  40.74767  -73.97500   
5        Garon     Brooklyn  Bedford-Stuyvesant  40.68688  -73.95596   

         room_type  price  minimo_noites  numero_de_reviews ultima_review  
0  Entire home/apt    225              1                 45    2019-05-21   
2  Entire home/apt     89              1                270    2019-07-05   
3  Entire home/apt     80             10                  9    2018-11-19   
4  Entire home/apt    200              3                 74    2019-06-22   
5     Private room     60             45                 49    2017-10-05   

   reviews_por_mes  calculado_host_listings_count  disponibilidade_365  
0             0.38                              2                  355  
2             4.64                              1                  194  
3             0.10                              1                    0  
4             0.59                              1                  129  
5             0.40                              1                    0
"""

df.info()

"""
<class 'pandas.core.frame.DataFrame'>
Index: 38820 entries, 0 to 48851
Data columns (total 16 columns):
 #   Column                         Non-Null Count  Dtype  
---  ------                         --------------  -----  
 0   id                             38820 non-null  int64  
 1   nome                           38820 non-null  object 
 2   host_id                        38820 non-null  int64  
 3   host_name                      38820 non-null  object 
 4   bairro_group                   38820 non-null  object 
 5   bairro                         38820 non-null  object 
 6   latitude                       38820 non-null  float64
 7   longitude                      38820 non-null  float64
 8   room_type                      38820 non-null  object 
 9   price                          38820 non-null  int64  
 10  minimo_noites                  38820 non-null  int64  
 11  numero_de_reviews              38820 non-null  int64  
 12  ultima_review                  38820 non-null  object 
 13  reviews_por_mes                38820 non-null  float64
 14  calculado_host_listings_count  38820 non-null  int64  
 15  disponibilidade_365            38820 non-null  int64  
dtypes: float64(3), int64(7), object(6)
memory usage: 5.0+ MB
"""

print(df.isnull().values.any()) # False
print(df.isnull().sum())

"""
id                               0
nome                             0
host_id                          0
host_name                        0
bairro_group                     0
bairro                           0
latitude                         0
longitude                        0
room_type                        0
price                            0
minimo_noites                    0
numero_de_reviews                0
ultima_review                    0
reviews_por_mes                  0
calculado_host_listings_count    0
disponibilidade_365              0
dtype: int64
"""

# Identificando e removendo duplicatas, caso existam:
print(df.drop_duplicates(inplace=True)) # None

# Verificando uma estatística descritiva:
pd.set_option('display.max_columns', 20) # garantindo a análise de todas as colunas
print(df.describe())

"""
                id       host_id      latitude     longitude         price  
count  3.882000e+04  3.882000e+04  38820.000000  38820.000000  38820.000000   
mean   1.810127e+07  6.424747e+07     40.728131    -73.951148    142.332354   
std    1.069347e+07  7.589779e+07      0.054990      0.046693    196.997290   
min    2.595000e+03  2.438000e+03     40.506410    -74.244420      0.000000   
25%    8.722029e+06  7.032517e+06     40.688640    -73.982462     69.000000   
50%    1.887339e+07  2.837143e+07     40.721710    -73.954805    101.000000   
75%    2.756792e+07  1.019092e+08     40.762990    -73.935020    170.000000   
max    3.645581e+07  2.738417e+08     40.913060    -73.712990  10000.000000   

       minimo_noites  numero_de_reviews  reviews_por_mes  
count   38820.000000       38820.000000     38820.000000   
mean        5.869346          29.290778         1.373259   
std        17.389233          48.183410         1.680339   
min         1.000000           1.000000         0.010000   
25%         1.000000           3.000000         0.190000   
50%         2.000000           9.000000         0.720000   
75%         4.000000          33.000000         2.020000   
max      1250.000000         629.000000        58.500000   

       calculado_host_listings_count  disponibilidade_365  
count                   38820.000000         38820.000000  
mean                        5.166589           114.879856  
std                        26.303293           129.525398  
min                         1.000000             0.000000  
25%                         1.000000             0.000000  
50%                         1.000000            55.000000  
75%                         2.000000           229.000000  
max                       327.000000           365.000000  

Algumas observações a partir das informações estatísticas descritivas acima é que valores como a média de lat e long não 
possuem significado geográfico e nem estatístico. O id também pode ser desconsiderados do ponto de vista estatístico.
"""

# Analisando o preço do aluguel dos imóveis e sua localização geográfica

lat_long_price = df[['latitude', 'longitude', 'price']]

lat = df['latitude']
long = df['longitude']
price = df['price']
room_type = df['room_type']
bairro = df['bairro']
bairro_roup = df['bairro_group']
id = df['id']

title = "Projection prices of temporary rents in NYC"
fig = px.scatter_geo(df,
                     lat=lat, lon=long, title=title,
                     hover_data=['id', 'room_type'],
                     color=price,
                     color_continuous_scale='rainbow', # melhor discrepância na visualização
                     labels={'color': 'Price'},
                     projection='albers usa',
                     )
fig.show()


"""
Opções para personalização de projeção no mapa:

projection (str) – One of 'equirectangular', 'mercator', 'orthographic', 'natural earth', 'kavrayskiy7', 'miller', 
'robinson', 'eckert4', 'azimuthal equal area', 'azimuthal equidistant', 'conic equal area', 'conic conformal', 
'conic equidistant', 'gnomonic', 'stereographic', 'mollweide', 'hammer', 'transverse mercator', 'albers usa', 
'winkel tripel', 'aitoff', or 'sinusoidal'`Default depends on `scope.
"""

"""
Ao analisar a projeção acima, verifica-se que não há um padrão lógico entre a precificação do valor dos aluguéis
temporários e sua localização geográfica para um imóvel em particular, visto que, no mesmo quateirão, dois imóveis 
opostos verificou-se as deguintes discrepâncias:

lat 40.65724, long -73.9245 -> price = $ 7500
let 40.65742, long -73.92385 -> price = $ 85

Contudo, existem outras variáveis que é o tipo de acomodação, pois existem acomodações compartilhadas, apenas o quarto e 
o aluguel do imóvel inteiro. Um dos imóveis, o aluguel custa $ 7500 para um quarto, enquanto o outro imóvel o aluguel 
custa $ 85 para o apartamento inteiro. Podemos estar nos deparando com um dado espúrio dentro do dataset. 

lat 40.65724, long -73.9245 -> price = $ 7500 -> room_type: Private room
let 40.65742, long -73.92385 -> price = $ 85 -> Entire home/apt
"""

"""
Todas as variáveis são: 

id, nome, host_id, host_name, bairro_group, bairro, room_type, latitude, longitude, price, minimo_noites, 
numero_de_reviews, ultima_review, reviews_por_mes, calculado_host_listings_count, disponibilidade_365

Com base no objetivo, que é a precificação, infere-se neste momento que as variáveis julgadas importantes para a 
precificação do aluguel temporário são a localização do imóvel de uma forma geral (bairro_group, bairro, latitude, 
longitude) e o tipo de imóvel (room_type), pois pode ser que o aluguel seja de um imóvel por inteiro, como pode ser o 
aluguel de quarto privado ou quarto compartilhado. Desta forma, a relação entre essas variáveis pode auxiliar na 
previsão de precificação do valor do aluguel temporário. 

As variáveis id, id_host, nome, ultima_review (data),  podem não possui nenhum significado do ponto de visto 
estatístico, podendo serem removidas. 

As demais variáveis minimo_noites, reviews_por_mes, numero_de_reviews, calculado_host_listings_count, latitude, 
longitude e disponibilidade_365 passarão pelo processo de normalização.
"""


# Removendo dados, Transformando a variável room_type, bairro_group e bairro em variáveis numérica e normalizando
df = df.drop(columns=['id', 'host_id', 'nome', 'host_name', 'ultima_review'])

le_room_type = LabelEncoder()
le_bairro_group = LabelEncoder()
le_bairro = LabelEncoder()
df['room_type_numeric'] = le_room_type.fit_transform(df['room_type'])
df['bairro_group_numeric'] = le_bairro_group.fit_transform(df['bairro_group'])
df['bairro_numeric'] = le_bairro.fit_transform(df['bairro'])

# Deixando somente dados numéricos
df = df.drop(columns=['room_type', 'bairro_group', 'bairro'])

# Normalizando
df_to_norm = ['latitude', 'longitude', 'minimo_noites', 'numero_de_reviews', 'reviews_por_mes',
              'calculado_host_listings_count', 'disponibilidade_365']
scaler = MinMaxScaler()
df[df_to_norm] = scaler.fit_transform(df[df_to_norm])
print(df)

"""
        latitude  longitude  price  minimo_noites  numero_de_reviews  
0      0.607918   0.490469    225       0.000000           0.070064   
2      0.439518   0.535649     89       0.000000           0.428344   
3      0.718308   0.565324     80       0.007206           0.012739   
4      0.593287   0.506972    200       0.001601           0.116242   
5      0.443797   0.542800     60       0.035228           0.076433   
...         ...        ...    ...            ...                ...   
48781  0.675224   0.547128    129       0.000000           0.000000   
48789  0.601574   0.808818     45       0.000000           0.000000   
48798  0.087004   0.191314    235       0.000000           0.000000   
48804  0.741325   0.602939    100       0.000000           0.001592   
48851  0.471265   0.595394     30       0.000000           0.000000   

       reviews_por_mes  calculado_host_listings_count  disponibilidade_365  
0             0.006326                       0.003067             0.972603   
2             0.079159                       0.000000             0.531507   
3             0.001539                       0.000000             0.000000   
4             0.009916                       0.000000             0.353425   
5             0.006668                       0.000000             0.000000   
...                ...                            ...                  ...   
48781         0.016926                       0.000000             0.402740   
48789         0.016926                       0.015337             0.928767   
48798         0.016926                       0.000000             0.238356   
48804         0.034023                       0.000000             0.109589   
48851         0.016926                       0.000000             0.002740   

       room_type_numeric  bairro_group_numeric  bairro_numeric  
0                      0                     2             126  
2                      0                     1              41  
3                      0                     2              61  
4                      0                     2             136  
5                      1                     1              13  
...                  ...                   ...             ...  
48781                  1                     2             199  
48789                  1                     3              77  
48798                  1                     4              89  
48804                  0                     0             133  
48851                  1                     1              28  

[38820 rows x 11 columns]
"""

# Plotando todas as variáveis em função da variável price
plt.scatter(df['room_type_numeric'], df['price'])
plt.title(f'room_type_numeric x price')
plt.xlabel(f'room_type_numeric')
plt.ylabel('price ($)')
plt.show()

# for i in df:
#     plt.scatter(df[i], df['price'])
#     plt.title(f'{i} x price')
#     plt.xlabel(f'{i}')
#     plt.ylabel('price ($)')
#     plt.show()


# Identificando e removendo outliers - Z-score
zscores = stats.zscore(df)
anomaly = df[np.abs(zscores)>3]
df = df[zscores < 3]
print(f'Anomalies Identified:\n{anomaly}')

"""
        latitude  longitude  price  minimo_noites  numero_de_reviews  
2      0.439518   0.535649     89       0.000000           0.428344   
6      0.635633   0.488286     79       0.000801           0.683121   
10     0.625476   0.481249     85       0.000801           0.297771   
13     0.562867   0.450050    120       0.071257           0.041401   
15     0.455625   0.512899    215       0.000801           0.313694   
...         ...        ...    ...            ...                ...   
48400  0.248592   0.309787     65       0.000000           0.000000   
48525  0.249920   0.914758     45       0.001601           0.000000   
48798  0.087004   0.191314    235       0.000000           0.000000   
48798  0.087004   0.191314    235       0.000000           0.000000   
48798  0.087004   0.191314    235       0.000000           0.000000   

       reviews_por_mes  calculado_host_listings_count  disponibilidade_365  
2             0.079159                       0.000000             0.531507   
6             0.059155                       0.000000             0.602740   
10            0.025474                       0.000000             0.106849   
13            0.003590                       0.000000             0.000000   
15            0.029236                       0.000000             0.879452   
...                ...                            ...                  ...   
48400         0.016926                       0.000000             0.490411   
48525         0.016926                       0.003067             0.246575   
48798         0.016926                       0.000000             0.238356   
48798         0.016926                       0.000000             0.238356   
48798         0.016926                       0.000000             0.238356   

       room_type_numeric  bairro_group_numeric  bairro_numeric  
2                      0                     1              41  
6                      1                     2              94  
10                     1                     2              94  
13                     0                     2             207  
15                     0                     1              80  
...                  ...                   ...             ...  
48400                  1                     4             168  
48525                  1                     3              12  
48798                  1                     4              89  
48798                  1                     4              89  
48798                  1                     4              89  

[4008 rows x 11 columns]
"""

# Matriz de correlação
corr = df.corr(method='spearman')
sns.heatmap(corr, cmap='coolwarm', annot=True, fmt='.2f')
plt.title('Correlation matrix no z-score')
plt.xticks(rotation=15)
plt.show()

corr = anomaly.corr(method='spearman')
sns.heatmap(corr, cmap='coolwarm', annot=True, fmt='.2f')
plt.title('Correlation matrix z-score')
plt.xticks(rotation=15)
plt.show()

"""
Nota-se que há uma relação aceitável e questionável entre 'price' e 'room_typ_numeric' com o valor de -0,68 ~ -0,7. É 
possível notar que não houve alteração significativa com a matriz de correlação após a remoção dos outliers detectados. 
Serão utilizados, portanto, todos os dados no modelo preditivo.

Para fins de informação, após uma rápida consulta, um dos bairros mais caros para se morar em NYC são:
- SoHo: US$ 6.100 mensais (R$ 37 mil)
- Tribeca: US$ 8.295 mensais (R$ 51 mil)
- Outros bairros caros incluem Kingsbridge e Bronx, com média de aluguel de US$ 1.854 (R$ 10 mil).
"""


room_type_map = dict(zip(range(len(le_room_type.classes_)), le_room_type.classes_))
print(f'Número e classe associada ao room_type: {room_type_map}')

bairro_group_map = dict(zip(range(len(le_bairro_group.classes_)), le_bairro_group.classes_))
print(f'Número e classe associada ao bairro_group: {bairro_group_map}')

bairro_map = dict(zip(range(len(le_bairro.classes_)), le_bairro.classes_))
print(f'Número e classe associada ao bairro_group: {bairro_map}')

"""
De acordo com a matriz de correlação, identificamos somente uma variável que pode ser considerada 'significativa', 
a room_type_numeric (-0,68), mas ainda sim questionável. Desta forma, acredita-se que o modelo de regressão linear não 
seja a melhor opção para o desenvolvimento do modelo preditivo da precificação. Assim sendo, dada a relação não linear 
entre a variável dependente e as independentes, opta-se, portanto, pela utilização de árvore de decisão para elaboração 
do modelo.
"""
# Árvore de Decisão

# Separando as variáveis independentes da dependente
y = df.iloc[:,2].values
X = df.iloc[:,[0,1,3,4,5,6,7,8,9,10]].values

# df.info() # confirma a posição das variáveis acima

# Dividindo os dados em treino e teste

X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size=0.3, random_state=1)
# test_size indica a porção que eu quero utilizar para os meus dados de teste. Se eu colocar 0.30, ele usará 70% como
# treinamento. random_state é para repetir as amostras.

modelo = DecisionTreeClassifier(random_state=1) # reprodutividade / repetição do resultado
modelo.fit(X_treinamento, y_treinamento)

# Salvando o modelo conforme solicitado
joblib.dump(modelo,'modelo_decision_tree.pkl')

previsoes = modelo.predict(X_teste)

accuracy = accuracy_score(y_teste, previsoes)
precision = precision_score(y_teste, previsoes, average='weighted')
recall = recall_score(y_teste, previsoes, average='weighted')
f1 = f1_score(y_teste, previsoes, average='weighted')
print(f'Accuracy: {accuracy} / Precision: {precision} / Recall: {recall} / F1: {f1}')

# Fazendo as previsões

# X_treinamento_feat = df.iloc[:,[8,9,10]].values # somente com o tipo de quarto, bairro e bairro_group
X_treinamento_feat2 = df.iloc[:,[3,7,8,9,10]].values # tipo de quarto, bairro ,bairro_group, min_noite e disponibilidade
y_treinamento_feat = df.iloc[:,2].values # price

modelo = DecisionTreeClassifier()
modelo.fit(X_treinamento_feat2, y_treinamento_feat)


num_minimo_noites = int(input('Número mínimo de noites: '))
disponibilidade = int(input('Disponibilidade: '))
tipo_quarto = int(input('0: Apartamento Inteiro, 1: Quarto privado, 2: Quarto compartilhado: '))
grupo_bairro = int(input('0: Bronx, 1: Brooklyn, 2: Manhattan, 3: Queens, 4: Staten Island: '))
bairro_nyc = int(input('Digite o número do bairro desejado: '))


features = [[num_minimo_noites, disponibilidade, tipo_quarto, grupo_bairro, bairro_nyc]]
# features = [[tipo_quarto, grupo_bairro, bairro_nyc]]
predicted_value_rent = modelo.predict(features)

print(f"A previsão do valor do aluguel temporário será de $ {predicted_value_rent[0]:.2f}.")

"""
Vamos verificar se o número de noites e a disponibilidade ao longo do ano interferem no preço.

Escolhendo as opções aleatórias:
1 - Número mínimo de noites: 1
2 - Disponibilidade: 200
3 - 0: Apartamento Inteiro, 1: Quarto privado, 2: Quarto compartilhado: 0
4 - 0: Bronx, 1: Brooklyn, 2: Manhattan, 3: Queens, 4: Staten Island: 1
5 - Digite o número do bairro desejado: 25

R.: A previsão do valor do aluguel temporário é de $ 115.00.

Agora, considerando somente a localização do imóvel e tipo de acomodação:

R.: A previsão do valor do aluguel temporário será de $ 200.00.

Assim sendo, o número mínimo de noites e a disponibilidade ao longo do ano interferem no preço.
"""

"""
Um apartamento com as seguintes características:

{'id': 2595,
 'nome': 'Skylit Midtown Castle',
 'host_id': 2845,
 'host_name': 'Jennifer',
 'bairro_group': 'Manhattan',
 'bairro': 'Midtown',   ---------->  126: 'Midtown'
 'latitude': 40.75362,
 'longitude': -73.98377,
 'room_type': 'Entire home/apt',
 'minimo_noites': 1,
 'numero_de_reviews': 45,
 'ultima_review': '2019-05-21',
 'reviews_por_mes': 0.38,
 'calculado_host_listings_count': 2,
 'disponibilidade_365': 355}

Teria como sugestão de preço: $ 125.00

Número mínimo de noites: 1
Disponibilidade: 355
0: Apartamento Inteiro, 1: Quarto privado, 2: Quarto compartilhado: 0
0: Bronx, 1: Brooklyn, 2: Manhattan, 3: Queens, 4: Staten Island: 2
Digite o número do bairro desejado: 126
A previsão do valor do aluguel temporário será de $ 125.00.
"""

"""
Para a realização deste desafio, seguem abaixo as referências utilziadas para consultas:

- Netto, A.; Maciel, F. Python para Data Science e Machine Learfing Descomplicado. 1. ed. Rio de Janeiro: Editora Alta 
Books, 2021;

- Matthes, E. Curso Intensivo de Python - Uma Introdução Prática e Baseada em Projetos à Programação. 3. ed. São Paulo.
Editora Novatec, 2023;

- Amaral, F. Formação Completa Inteligência Artificial e Machine Learning. 2025. Curso online. Disponível em:
<https://www.udemy.com/course/inteligencia-artificial-e-machine-learning/?kw=intelig%C3%AAncia&src=sac&couponCode=KEEPLEARNINGBR>.
Acessado em 03 de fevereiro de 2025.

"""