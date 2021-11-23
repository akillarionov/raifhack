#!/usr/bin/env python
# coding: utf-8

# ### Импорт библиотек

# In[1]:


import pandas as pd
import numpy as np
from datetime import datetime as dt


# In[2]:


# Visualization

import seaborn as sns
sns.set()

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import plotly.graph_objects as go


# In[3]:


#!conda install -c conda-forge lightgbm --yes
#!conda install -c conda-forge xgboost --yes
#!conda install -c conda-forge plotly --yes


# In[4]:


#pip install lightgbm
#pip install xgboost
#!pip install plotly


# In[5]:


# Feature Engineering

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans


# In[6]:


# ML

from sklearn.model_selection import train_test_split
from sklearn import metrics
import lightgbm as ltb
from lightgbm import LGBMRegressor
import xgboost as xg
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score


# In[7]:


import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows',None)
pd.set_option("display.float_format", "{:.3f}".format)


# In[8]:


RANDOM_SEED = 20210925
get_ipython().system('pip freeze > requirements.txt')


# ### Подготовка функций

# In[9]:


def hist_descr(col):
    title = col.split('_')[1] + "_" + col.split('_')[-1]
    
    fig, ax = plt.subplots(1, 1, figsize = (5, 3))
    ax.hist(df[col], bins = 50);
    ax.set_title(title)

    plt.tight_layout()
    plt.show()
    
    print(df[col].describe()[['mean', 'std', 'min', '50%', 'max']].round(2).reset_index().
            to_csv(header=None, index=None, sep='\t'))


# In[10]:


def correl(X):
    
    corr = df[X].corr()
    sns.set(font_scale=0.9)
    plt.subplots(figsize=(9, 6))
    plt.tight_layout()
    sns.heatmap(corr.abs(), square=True, vmin=0, vmax=1,
                annot=True, fmt=".1f", linewidths=0.1, cmap="icefire");


# ### Загрузка датасетов

# In[11]:


df_train = pd.read_csv('/.../data/train.csv')
df_test = pd.read_csv('/.../data/test.csv')

print(df_train.shape, df_test.shape)


# In[12]:


df_test['target_price'] = 0
df_train['target_price'] = df_train.per_square_meter_price
df_train.drop('per_square_meter_price', axis=1, inplace=True)

# Разметим трейн и тест

df_train['sample'] = 1
df_test['sample'] = 0


# In[13]:


# Объединим данные в единый датасет

df = df_train.append(df_test, sort=False).reset_index(drop=True)
del df_train, df_test
df.shape


# In[14]:


# Здесь рассмотрим только данные по price_type == 1
# Как и в валидационной выборке

display(df[df['sample'] == 0].price_type.value_counts())
display(df[df['sample'] == 1].price_type.value_counts())


# In[15]:


df = df[df.price_type == 1].reset_index()
df.drop('price_type', axis=1, inplace=True)
df.shape


# ### Обработка пропусков

# In[16]:


#df.isna().sum()


# In[17]:


# Отдельно обработаем одну запись - заменим город и население

#df[df.osm_city_nearest_population.isna()]

df.loc[df['osm_city_nearest_name'] == '饶河县', 'osm_city_nearest_name'] = 'КНР'
a = df[df.region == 'Приморский край']['osm_city_nearest_population'].median()
df.loc[df.osm_city_nearest_population.isna(), 'osm_city_nearest_population'] = a


# In[18]:


# Обозначаем какие записи изначально имели пропуски

df['na_house_1000'] = pd.isna(df['reform_house_population_1000']).astype('float64')
df['na_house_500'] = pd.isna(df['reform_house_population_500']).astype('float64')
df['na_floor_1000'] = pd.isna(df['reform_mean_floor_count_1000']).astype('float64')
df['na_floor_500'] = pd.isna(df['reform_mean_floor_count_500']).astype('float64')
df['na_yearb_1000'] = pd.isna(df['reform_mean_year_building_1000']).astype('float64')
df['na_yearb_500'] = pd.isna(df['reform_mean_year_building_500']).astype('float64')


# In[19]:


isna_cols = ['reform_house_population_1000', 'reform_house_population_500', 
             'reform_mean_floor_count_1000', 'reform_mean_floor_count_500', 
             'reform_mean_year_building_1000', 'reform_mean_year_building_500']

# Используем моду только по трейн выборке, - так что лика по дате нет
for c in isna_cols:
    print(c , '\t', df[df['sample'] == 1][c].mode()[0].round(0))


# **Далее каскад мероприятий по замене отсутствующих значений наиболее частым значением в train**

# In[20]:


# Замена всех пропущенных значений (на основе данных train)

df.reform_house_population_1000 = df.apply(lambda row: 2113
                                           if pd.isna(row.reform_house_population_1000) 
                                           else row.reform_house_population_1000, axis=1)


# In[21]:


df.reform_house_population_500 = df.apply(lambda row: 486
                                          if pd.isna(row.reform_house_population_500) 
                                          else row.reform_house_population_500, axis=1)


# In[22]:


df.reform_mean_floor_count_1000 = df.apply(lambda row: 7
                                           if pd.isna(row.reform_mean_floor_count_1000) 
                                           else row.reform_mean_floor_count_1000, axis=1)


# In[23]:


df.reform_mean_floor_count_500 = df.apply(lambda row: 5
                                          if pd.isna(row.reform_mean_floor_count_500) 
                                          else row.reform_mean_floor_count_500, axis=1)


# In[24]:


df.reform_mean_year_building_1000 = df.apply(lambda row: 1968
                                             if pd.isna(row.reform_mean_year_building_1000) 
                                             else row.reform_mean_year_building_1000, axis=1)


# In[25]:


df.reform_mean_year_building_500 = df.apply(lambda row: 1967
                                            if pd.isna(row.reform_mean_year_building_500) 
                                            else row.reform_mean_year_building_500, axis=1)


# **Street**

# In[26]:


print('Города с пропусками улиц в train:')
trc = set(df[(df.street.isna()) & (df['sample'] == 1)]['osm_city_nearest_name'])
display(trc)


# In[27]:


print('Города с пропусками улиц в test:')
tec = set(df[(df.street.isna()) & (df['sample'] == 0)]['osm_city_nearest_name'])
display(tec)


# **Пересечений данных train и test по условию быть не должно, поэтому заполним пропуски модой по отдельности.**

# In[28]:


for city in trc:
    
    a = df[(df['sample'] == 1) & (df['osm_city_nearest_name'] == city)]['street'].mode()[0]
    
    df.loc[(df['sample'] == 1) & (df['street'].isna()) & 
           (df['osm_city_nearest_name'] == city), 'street'] = a


# In[29]:


for city in tec:
    
    a = df[(df['sample'] == 0) & (df['osm_city_nearest_name'] == city)]['street'].mode()[0]
    
    df.loc[(df['sample'] == 0) & (df['street'].isna()) & 
           (df['osm_city_nearest_name'] == city), 'street'] = a


# #### Floor

# In[30]:


# https://github.com/BatyaZhizni/Raifhack-DS/

# почистим признак floor
df['floor'] = df['floor'].mask(df['floor'] == '-1.0', -1)               .mask(df['floor'] == '-2.0', -2)               .mask(df['floor'] == '-3.0', -3)               .mask(df['floor'] == 'подвал, 1', 1)               .mask(df['floor'] == 'подвал', -1)               .mask(df['floor'] == 'цоколь, 1', 1)               .mask(df['floor'] == '1,2,антресоль', 1)               .mask(df['floor'] == 'цоколь', 0)               .mask(df['floor'] == 'тех.этаж (6)', 6)               .mask(df['floor'] == 'Подвал', -1)               .mask(df['floor'] == 'Цоколь', 0)               .mask(df['floor'] == 'фактически на уровне 1 этажа', 1)               .mask(df['floor'] == '1,2,3', 1)               .mask(df['floor'] == '1, подвал', 1)               .mask(df['floor'] == '1,2,3,4', 1)               .mask(df['floor'] == '1,2', 1)               .mask(df['floor'] == '1,2,3,4,5', 1)               .mask(df['floor'] == '5, мансарда', 5)               .mask(df['floor'] == '1-й, подвал', 1)               .mask(df['floor'] == '1, подвал, антресоль', 1)               .mask(df['floor'] == 'мезонин', 2)               .mask(df['floor'] == 'подвал, 1-3', 1)               .mask(df['floor'] == '1 (Цокольный этаж)', 0)               .mask(df['floor'] == '3, Мансарда (4 эт)', 3)               .mask(df['floor'] == 'подвал,1', 1)               .mask(df['floor'] == '1, антресоль', 1)               .mask(df['floor'] == '1-3', 1)               .mask(df['floor'] == 'мансарда (4эт)', 4)               .mask(df['floor'] == '1, 2.', 1)               .mask(df['floor'] == 'подвал , 1 ', 1)               .mask(df['floor'] == '1, 2', 1)               .mask(df['floor'] == 'подвал, 1,2,3', 1)               .mask(df['floor'] == '1 + подвал (без отделки)', 1)               .mask(df['floor'] == 'мансарда', 3)               .mask(df['floor'] == '2,3', 2)               .mask(df['floor'] == '4, 5', 4)               .mask(df['floor'] == '1-й, 2-й', 1)               .mask(df['floor'] == '1 этаж, подвал', 1)               .mask(df['floor'] == '1, цоколь', 1)               .mask(df['floor'] == 'подвал, 1-7, техэтаж', 1)               .mask(df['floor'] == '3 (антресоль)', 3)               .mask(df['floor'] == '1, 2, 3', 1)               .mask(df['floor'] == 'Цоколь, 1,2(мансарда)', 1)               .mask(df['floor'] == 'подвал, 3. 4 этаж', 3)               .mask(df['floor'] == 'подвал, 1-4 этаж', 1)               .mask(df['floor'] == 'подва, 1.2 этаж', 1)               .mask(df['floor'] == '2, 3', 2)               .mask(df['floor'] == '7,8', 7)               .mask(df['floor'] == '1 этаж', 1)               .mask(df['floor'] == '1-й', 1)               .mask(df['floor'] == '3 этаж', 3)               .mask(df['floor'] == '4 этаж', 4)               .mask(df['floor'] == '5 этаж', 5)               .mask(df['floor'] == 'подвал,1,2,3,4,5', 1)               .mask(df['floor'] == 'подвал, цоколь, 1 этаж', 1)               .mask(df['floor'] == '3, мансарда', 3)               .mask(df['floor'] == 'цоколь, 1, 2,3,4,5,6', 1)               .mask(df['floor'] == ' 1, 2, Антресоль', 1)               .mask(df['floor'] == '3 этаж, мансарда (4 этаж)', 3)               .mask(df['floor'] == 'цокольный', 0)               .mask(df['floor'] == '1,2 ', 1)               .mask(df['floor'] == '3,4', 3)               .mask(df['floor'] == 'подвал, 1 и 4 этаж', 1)               .mask(df['floor'] == '5(мансарда)', 5)               .mask(df['floor'] == 'технический этаж,5,6', 5)               .mask(df['floor'] == ' 1-2, подвальный', 1)               .mask(df['floor'] == '1, 2, 3, мансардный', 1)               .mask(df['floor'] == 'подвал, 1, 2, 3', 1)               .mask(df['floor'] == '1,2,3, антресоль, технический этаж', 1)               .mask(df['floor'] == '3, 4', 3)               .mask(df['floor'] == '1-3 этажи, цоколь (188,4 кв.м), подвал (104 кв.м)', 1)               .mask(df['floor'] == '1,2,3,4, подвал', 1)               .mask(df['floor'] == '2-й', 2)               .mask(df['floor'] == '1, 2 этаж', 1)               .mask(df['floor'] == 'подвал, 1, 2', 1)               .mask(df['floor'] == '1-7', 1)               .mask(df['floor'] == '1 (по док-м цоколь)', 1)               .mask(df['floor'] == '1,2,подвал ', 1)               .mask(df['floor'] == 'подвал, 2', 2)               .mask(df['floor'] == 'подвал,1,2,3', 1)               .mask(df['floor'] == '1,2,3 этаж, подвал ', 1)               .mask(df['floor'] == '1,2,3 этаж, подвал', 1)               .mask(df['floor'] == '2, 3, 4, тех.этаж', 2)               .mask(df['floor'] == 'цокольный, 1,2', 1)               .mask(df['floor'] == 'Техническое подполье', -1)               .mask(df['floor'] == '1.2', 1)               .astype(float)


# In[31]:


print('train distribution')
print(df[df['sample'] == 1]['floor'].value_counts(normalize=True, dropna=False).round(4)[:2])

print('\ntest distribution')
print(df[df['sample'] == 0]['floor'].value_counts(normalize=True, dropna=False).round(4)[:2])


# In[32]:


# Наиболее частое после пропусков значение = 1

df.loc[df['floor'].isna(), 'floor'] = 1


# ### Проверка автокорреляции признаков OSM
# #### И удаление наиболее взаимосвязанных

# In[33]:


# Перенос текстового признака в конец таблицы
df['city_nearest_name'] = df['osm_city_nearest_name']
df.drop('osm_city_nearest_name', axis=1, inplace=True)


# In[34]:


X1 = list(df.columns[6:14])
X1


# In[35]:


#correl(X1)


# In[36]:


X1.remove('osm_building_points_in_0.005')
X1.remove('osm_building_points_in_0.0075')
X1.remove('osm_building_points_in_0.01')

X1.remove('osm_amenity_points_in_0.005')
X1.remove('osm_amenity_points_in_0.0075')
X1.remove('osm_amenity_points_in_0.01')


# In[37]:


X2 = list(df.columns[14:25])
X2


# In[38]:


#correl(X2)


# In[39]:


X2.remove('osm_crossing_points_in_0.005')
X2.remove('osm_crossing_points_in_0.0075')
X2.remove('osm_crossing_points_in_0.01')

X2.remove('osm_catering_points_in_0.005')
X2.remove('osm_catering_points_in_0.0075')
X2.remove('osm_catering_points_in_0.01')


# In[40]:


X3 = list(df.columns[25:36])
X3


# In[41]:


#correl(X3)


# In[42]:


X3.remove('osm_culture_points_in_0.005')
X3.remove('osm_culture_points_in_0.0075')
X3.remove('osm_culture_points_in_0.01')

X3.remove('osm_finance_points_in_0.005')
X3.remove('osm_finance_points_in_0.0075')
X3.remove('osm_finance_points_in_0.01')

X3.remove('osm_healthcare_points_in_0.0075')
X3.remove('osm_healthcare_points_in_0.01')


# In[43]:


X4 = list(df.columns[36:45])
X4


# In[44]:


#correl(X4)


# In[45]:


X4.remove('osm_historic_points_in_0.0075')
X4.remove('osm_historic_points_in_0.01')

X4.remove('osm_hotels_points_in_0.0075')
X4.remove('osm_hotels_points_in_0.01')

X4.remove('osm_leisure_points_in_0.0075')
X4.remove('osm_leisure_points_in_0.01')


# In[46]:


X5 = list(df.columns[45:53])
X5


# In[47]:


#correl(X5)


# In[48]:


X5.remove('osm_offices_points_in_0.005')
X5.remove('osm_offices_points_in_0.0075')
X5.remove('osm_offices_points_in_0.01')

X5.remove('osm_shops_points_in_0.005')
X5.remove('osm_shops_points_in_0.0075')
X5.remove('osm_shops_points_in_0.01')


# In[49]:


X6 = list(df.columns[53:62])
X6


# In[50]:


#correl(X6)


# In[51]:


X6.remove('osm_train_stop_points_in_0.0075')
X6.remove('osm_train_stop_points_in_0.01')

X6.remove('osm_transport_stop_points_in_0.0075')
X6.remove('osm_transport_stop_points_in_0.01')


# In[52]:


X7 = list(df.columns[62:70])
X7


# In[53]:


#correl(X7)


# In[54]:


X7.remove('reform_count_of_houses_1000')
X7.remove('reform_house_population_1000')
X7.remove('reform_mean_floor_count_1000')
X7.remove('reform_mean_year_building_1000')


# In[55]:


X_osm = X1+X2+X3+X4+X5+X6+X7
#correl(X_osm)


# In[56]:


X_osm.remove('osm_amenity_points_in_0.001')
X_osm.remove('osm_catering_points_in_0.001')
X_osm.remove('osm_healthcare_points_in_0.005')
X_osm.remove('osm_hotels_points_in_0.005')

#X_osm.remove('osm_crossing_closest_dist')
#X_osm.remove('osm_catering_points_in_0.001')


# In[57]:


correl(X_osm)


# **На этом этапе логарифмируем числовые признаки и посмотрим, где это могло принести пользу**

# In[58]:


X_log_osm = []
for col in X_osm:
    title = 'log_' + col.split('_')[1] + col.split('_')[-2] + col.split('_')[-1]
    X_log_osm.append(title)
    df[title] = np.log(df[col] + 1)
    df[[col, title]].hist(figsize=(7, 4), bins=40);


# In[59]:


#correl(X_log_osm)


# In[60]:


X_log_osm.remove('log_citynearestpopulation')
X_log_osm.remove('log_shopsin0.001')
X_log_osm.remove('log_meanbuilding500')
X_log_osm.remove('log_counthouses500')


# In[61]:


correl(X_log_osm)


# #### Площадь недвижимости и целевая переменная - стоимость 1 кв. метра
# 
# Лошарифмируем для нормального распределения

# In[62]:


df['log_square'] = np.log(df['total_square'])
df[['total_square', 'log_square']].hist(figsize=(8, 3), bins=50);


# In[63]:


# для тестовой выборки - логарифмирование целевой переменной не важно, будет 0

df['log_target'] = np.log(np.where(df['sample'] == 1, df.target_price, 1))
df[['target_price', 'log_target']].hist(figsize=(8, 3), bins=50);


# ### Обработка даты

# In[64]:


# Приведём данные к общему формату
df['date'] = pd.to_datetime(df.date, format='%Y-%m-%d', errors='coerce')


# In[65]:


df['year'] = df.date.dt.year
df['month'] = df.date.dt.month
df['day'] = df.date.dt.day


# In[66]:


# Посмотрим на временные периоды в данных

df.groupby('sample')['month'].value_counts(dropna=False)


# In[67]:


daily = df[df['sample'] == 1].groupby(['day', 'month'], as_index=False)['city']     .agg({'city' : 'count'}).rename({'city' : 'total_number'}, axis=1)     .groupby('day', as_index=False)['total_number'].agg({'total_number' : 'mean'})     .rename({'total_number' : 'average_number'}, axis=1)

fig = plt.figure(figsize=(12, 4))
ax = sns.barplot(
                data=daily,
                x='day',
                y='average_number',
                palette="Blues_d")
fig.suptitle('Среднee число объявлений', fontsize=15)
sns.despine()
plt.show()


# #### Отображение даты и месяца на круг

# 12-й месяц не сильно отличается от 1-го, на практике это очень близкие значения. Так же и 31-е и 1-е числа очень близки. Но модель этого не знает, поэтому лучше будет отображать зацикленные данные в виде "циферблата". Для этого заменим показатели месяца/даты на функции синуса и косинуса.

# In[68]:


# Словарь с общим количеством дней в месяцах 2020 года

days_in_month = {1:31, 2:29, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 13:31}


# In[69]:


df['sin_day'] = np.sin(2*np.pi*df.day/df.month.map(days_in_month))
df['cos_day'] = np.cos(2*np.pi*df.day/df.month.map(days_in_month))

df.drop('day', axis=1, inplace=True)

df.sample(60).plot.scatter('sin_day', 'cos_day').set_aspect('equal');


# In[70]:


# За одно добавим признак времени года

seasons = {1: 'winter', 2: 'winter', 3: 'spring', 4: 'spring', 5: 'spring', 6: 'summer', 
           7: 'summer', 8: 'summer', 9: 'autumn', 10: 'autumn', 11: 'autumn', 12: 'winter'}

df['season'] = df.month.map(seasons)


# In[71]:


# Аналогично датам, обработаем данные по месяцам

df['sin_month'] = np.sin(2*np.pi*df.month/12)
df['cos_month'] = np.cos(2*np.pi*df.month/12)

df.drop('month', axis=1, inplace=True)

df.sample(60).plot.scatter('sin_month', 'cos_month').set_aspect('equal');


# Добавим дни недели и выходные

# In[72]:


#df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek + 1
#df['weekend'] = np.where(df['day_of_week'] >= 6, 1, 0)

#df['dow_sin'] = np.sin(2*np.pi*df.day_of_week/7)
#df['dow_cos'] = np.cos(2*np.pi*df.day_of_week/7)

#df.drop('day_of_week', axis=1, inplace=True)


# In[73]:


df.head(1)


# ### Отбор признаков для модели

# In[74]:


# price_type исключаем

X_check = X_log_osm + ['realty_type', 'log_square', 'log_target', 'sample'] +                       ['sin_day', 'cos_day', 'sin_month', 'cos_month'] +                       ['na_floor_1000', 'na_floor_500', 'na_house_1000', 
                       'na_house_500', 'na_yearb_1000', 'na_yearb_500']


# In[75]:


print(df[df['sample'] == 1][X_check].shape, df[df['sample'] == 0][X_check].shape)


# ### OneHotEncoder

# #### Попробуем сделать dummy для укрупненного признака по населению (10к)

# In[76]:


df['pop_thnd'] = round(df.osm_city_nearest_population / 10000, 0).astype('int64')
df['pop_thnd'].value_counts(dropna=False)[:5]


# In[77]:


OHE = OneHotEncoder(sparse=False)


# In[78]:


# Кодируем население по 10к

popul100k = OHE.fit_transform(df.pop_thnd.values.reshape(-1,1))
pop_tmp = pd.DataFrame(popul100k, columns=['pop_' + str(i) for i in range(len(popul100k[0]))])


# In[79]:


df_check = pd.concat([df[X_check], pop_tmp], axis=1)
df_check.shape


# In[80]:


# Кодируем время года

seas = OHE.fit_transform(df.season.values.reshape(-1,1))
seas_tmp = pd.DataFrame(seas, columns=['season_' + str(i) for i in range(len(seas[0]))])


# In[81]:


# Кодируем realty_type

realty_t = OHE.fit_transform(df.realty_type.values.reshape(-1,1))
rt_tmp = pd.DataFrame(realty_t, columns=['realty_' + str(i) for i in range(len(realty_t[0]))])


# In[82]:


# Кодируем названия областей

region_ohe = OHE.fit_transform(df.region.values.reshape(-1,1))
tmp_reg = pd.DataFrame(region_ohe, columns=['reg_' + str(i) for i in range(len(region_ohe[0]))])


# In[83]:


# Кодируем floor

floor_ohe = OHE.fit_transform(df.floor.values.reshape(-1,1))
tmp_floor = pd.DataFrame(floor_ohe, columns=['flo_' + str(i) for i in range(len(floor_ohe[0]))])


# In[84]:


df_check = pd.concat([df_check, seas_tmp, rt_tmp, tmp_reg, tmp_floor], axis=1)
df_check.shape


# ### Геоданные

# In[85]:


df['lat_rad'] = np.radians(df['lat'])
df['lng_rad'] = np.radians(df['lng'])


# In[86]:


plt.figure(figsize = (10,6))
sns.scatterplot(df['lat_rad'], df['lng_rad']);


# In[87]:


from sklearn.cluster import KMeans

kmeans = KMeans(10)
clusters = kmeans.fit_predict(df[['lat_rad','lng_rad']])
df['geo_cluster_KM'] = kmeans.predict(df[['lat_rad','lng_rad']])


# In[88]:


df.geo_cluster_KM.value_counts()


# In[89]:


# Кодируем clusters

clust_ohe = OHE.fit_transform(df.geo_cluster_KM.values.reshape(-1,1))
tmp_cls = pd.DataFrame(clust_ohe, columns=['clus_' + str(i) for i in range(len(clust_ohe[0]))])


# In[90]:


df_check = pd.concat([df_check, df[['lat_rad', 'lng_rad']], tmp_cls], axis=1)
df_check.shape


# ### Создание модели

# In[94]:


X = df_check[df_check['sample'] == 1].drop(['log_target', 'sample'], axis=1)
X.shape


# In[95]:


y = df_check[df_check['sample'] == 1]['log_target']
y.shape


# In[96]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_SEED)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# XGB

# In[115]:


params = {
    'max_depth': 10,
    'gamma': 0,
    'eta': .001,
    'subsample': 0.8,
    'colsample_bytree': 0.9,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 0
}


# In[116]:


matrix_train = xg.DMatrix(X_train, label=y_train)
matrix_test = xg.DMatrix(X_test, label=y_test)


# In[117]:


start_time = dt.now()
model = xg.train(params=params, dtrain=matrix_train, num_boost_round=10000,
                 early_stopping_rounds=30, evals=[(matrix_test, 'test')])


# In[118]:


print("Training time: %i seconds with %i training examples" %
      ((dt.now()-start_time).total_seconds(), len(y_train)))


# In[119]:


y_pred = model.predict(xg.DMatrix(X_test))


# In[120]:


print('MAE:', round(mean_absolute_error(y_test, y_pred), 3))  
print('MSE:', round(mean_squared_error(y_test, y_pred), 3))  
print('RMSE:', round(np.sqrt(mean_squared_error(y_test, y_pred)), 3))
print('VarScore:', round(explained_variance_score(y_test, y_pred), 3))


# In[ ]:


MAE: 0.22
MSE: 0.095
RMSE: 0.309
VarScore: 0.706


# ### Запуск на валидационной выборке и сабмишн

# In[121]:


test_sub = pd.read_csv('/.../data/test_submission.csv')
test_sub.head()


# In[122]:


len(test_sub)


# In[123]:


X_val = df_check[df_check['sample'] == 0].drop(['log_target', 'sample'], axis=1)
print(len(X_val))


# In[124]:


predict_sub = model.predict(xg.DMatrix(X_val))


# In[125]:


predict_sub = np.exp(predict_sub)
pd.Series(predict_sub).head()


# In[126]:


test_sub['per_square_meter_price'] = predict_sub
test_sub.to_csv('submission5.csv', index=False)
test_sub.head(10)


# In[ ]:




