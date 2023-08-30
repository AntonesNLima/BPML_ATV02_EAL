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


#Definição dos atributos e leitura dos dados abalone
data = pd.read_csv('abalone_dataset.csv')

columns = ['sex', 'length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight', 'type'] 

data.columns = columns

data = data.drop(columns=['sex'])

#Definição das bases de aprendizagem treino e teste Y matendo proporções originais dos tipos
def random_kfold_shuffle(data):
    type1 = data.query('type == 1')
    type1.reset_index(drop=True,inplace=True)

    type2 = data.query('type == 2')
    type2.reset_index(drop=True,inplace=True)

    type3 = data.query('type == 3')
    type3.reset_index(drop=True,inplace=True)

    test_1 = sorted(random.sample([i for i in type1.index.values],int(type1.shape[0]*.3)+1))
    train = type1.iloc[[i for i in type1.index if i not in test_1],:]
    teste = type1.iloc[test_1,:]

    test_2 = sorted(random.sample([i for i in type2.index.values],int(type2.shape[0]*.3)+1))
    train_2 = type2.iloc[[i for i in type2.index if i not in test_2],:]
    teste_2 = type2.iloc[test_2,:]

    test_3 = sorted(random.sample([i for i in type3.index.values],int(type3.shape[0]*.3)+1))
    train_3 = type3.iloc[[i for i in type3.index if i not in test_3],:]
    teste_3 = type3.iloc[test_3,:]

    train = pd.concat([train, train_2, train_3], ignore_index=True)
    teste = pd.concat([teste, teste_2, teste_3], ignore_index=True)

    x_train = train.drop('type', axis=1)
    y_train = train['type']

    x_test = teste.drop('type', axis=1)
    y_test = teste['type']
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = random_kfold_shuffle(data)

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
    lgb_random = lgb_random.fit(x_train, y_train)

    best_params = lgb_random.best_params_
    print("\n\n\n")
    print("Aooo...best params...:", best_params)

runGridSearchToEvaluateBestModelParams()

def runRegressionWithBestParams():
    global y_pred
    global best_params
    model = LGBMRegressor(**best_params, subsample=0.9, random_state=42, n_jobs=-1)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print("\n")
    print("Acurácia LGBMRegressor: {:.2f}".format(model.score(x_test, y_test)))
    print('\nTest RMSLE: {:.2f}'.format(np.sqrt(mse(y_test, y_pred))))
    lgb_rmsle = np.sqrt(mse(y_test, y_pred))
    print("\nCross validation RMSLE: {:.2f}".format(lgb_rmsle))
    # MAE
    print("\nMAE: {:.2f}".format(mae(y_test, y_pred)))
runRegressionWithBestParams()