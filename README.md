# LH_CD_LuizHamada
Modelo descritivo de precificação de aluguéis temporários na cidade de New York utilizando técnicas de Machine Learning.

# Apresentação
Este projeto teve como objetivo criar um modelo preditivo através de técnicas de machine learning com o intuito de realizar previsões de precificação de aluguéis temporários de um conjunto de dados de imóveis da cidade de Nova Iorque. O dataset intitulado 'teste_indicium_precificacao.csv' possui as seguintes informações:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 48894 entries, 0 to 48893
Data columns (total 16 columns):
     Column                         Non-Null Count  Dtype  
---  ------                         --------------  -----  
 0   id                             48894 non-null  int64  
 1   nome                           48878 non-null  object 
 2   host_id                        48894 non-null  int64  
 3   host_name                      48873 non-null  object 
 4   bairro_group                   48894 non-null  object 
 5   bairro                         48894 non-null  object 
 6   latitude                       48894 non-null  float64
 7   longitude                      48894 non-null  float64
 8   room_type                      48894 non-null  object 
 9   price                          48894 non-null  int64  
 10  minimo_noites                  48894 non-null  int64  
 11  numero_de_reviews              48894 non-null  int64  
 12  ultima_review                  38842 non-null  object 
 13  reviews_por_mes                38842 non-null  float64
 14  calculado_host_listings_count  48894 non-null  int64  
 15  disponibilidade_365            48894 non-null  int64  
dtypes: float64(3), int64(7), object(6)
memory usage: 6.0+ MB

# Instalação
Para que o modelo preditivo atinge todos os padrões satisfatórios, certifique-se de que tenha instalado em sua máquina python3.x e as seguintes bibliotecas: numpy, pandas, scipy, seaborn, matplotlib, sklearn.metrics (accuracy_score, precision_score, recall_score, f1_score), sklearn.preprocessing (LabelEncoder, MinMaxScaler), sklearn.tree (DecisionTreeClassifier), sklearn.model_selection (train_test_split).
