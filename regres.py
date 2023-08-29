import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import warnings
from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMRegressor
from scipy.stats import uniform, randint as sp_randint
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae


# Definição dos atributos e leitura dos dados abalone
data = pd.read_csv('abalone_dataset.csv')

columns = ['sex', 'length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight', 'type'] 

data.columns = columns

data = data.drop(columns=['sex'])

# Definição das bases de aprendizagem X e teste Y matendo proporções originais dos tipos
def random_kfold_shuffle(data):
    type1 = data.query('type == 1')
    type1.reset_index(drop=True,inplace=True)

    type2 = data.query('type == 2')
    type2.reset_index(drop=True,inplace=True)

    type3 = data.query('type == 3')
    type3.reset_index(drop=True,inplace=True)

    test_1 = sorted(random.sample([i for i in type1.index.values],int(type1.shape[0]*.3)+1))
    x = type1.iloc[[i for i in type1.index if i not in test_1],:]
    y = type1.iloc[test_1,:]

    test_2 = sorted(random.sample([i for i in type2.index.values],int(type2.shape[0]*.3)+1))
    x_2 = type2.iloc[[i for i in type2.index if i not in test_2],:]
    y_2 = type2.iloc[test_2,:]

    test_3 = sorted(random.sample([i for i in type3.index.values],int(type3.shape[0]*.3)+1))
    x_3 = type3.iloc[[i for i in type3.index if i not in test_3],:]
    y_3 = type3.iloc[test_3,:]

    x = pd.concat([x, x_2, x_3], ignore_index=True)
    y = pd.concat([y, y_2, y_3], ignore_index=True)

    x_train = x.drop(columns='type')  # Remove a coluna de rótulo 'type'
    y_train = y['type']  # Extrai a coluna 'type' como rótulo

    y_2_train = y_2['type']  # Extrai a coluna 'type' do segundo conjunto
    y_3_train = y_3['type']  # Extrai a coluna 'type' do terceiro conjunto

    X_test = pd.concat([y, y_2, y_3], ignore_index=True).drop(columns='type')  # Conjunto de teste sem a coluna de rótulo
    y_test = pd.concat([y['type'], y_2['type'], y_3['type']], ignore_index=True)  # Rótulos de teste

    X_train = pd.concat([x_train, x_2.drop(columns='type'), x_3.drop(columns='type')], ignore_index=True)  # Concatenar conjuntos de treinamento
    y_train = pd.concat([y_train, y_2_train, y_3_train], ignore_index=True)  # Concatenar rótulos de treinamento

    return X_train, X_test, y_train, y_test

# Extrair conjuntos de treinamento e teste
x_train, x_test, y_train, y_test = random_kfold_shuffle(data)

def runGridSearchToEvaluateBestModelParams():
    global best_params
    lgb_model = LGBMRegressor(subsample=0.9)

    params = {'learning_rate': uniform(0, 1),
              'n_estimators': sp_randint(200, 1500),
              'num_leaves': sp_randint(20, 200),
              'max_depth': sp_randint(2, 15),
              'min_child_weight': uniform(0, 2),
              'colsample_bytree': uniform(0, 1),
              }
    lgb_random = RandomizedSearchCV(lgb_model, param_distributions=params, n_iter=10, cv=3, random_state=42, 
                                scoring='neg_root_mean_squared_error', verbose=10, return_train_score=True)
    lgb_random = lgb_random.fit(x_train, y_train)  # Correção: usar x_train e y_train em vez de X_train e y_train

    best_params = lgb_random.best_params_
    print("\n\n\n")
    print("Aooo...best params...:", best_params)
    
# ...

# Chamar a função para avaliar melhores parâmetros
runGridSearchToEvaluateBestModelParams()

# ...
