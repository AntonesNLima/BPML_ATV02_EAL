import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import warnings


# Definição dos atributos e leitura dos dados abalone
data = pd.read_csv('abalone_dataset.csv')

columns = ['sex', 'length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight', 'type'] 

data.columns = columns

sex_dummies = pd.get_dummies(data['sex'])

# Renomear colunas para atingir o formato desejado
sex_dummies.rename(columns={'M': 'Masculino', 'F': 'Feminino', 'I': 'Infantil'}, inplace=True)
sex_dummies['Masculino'] = sex_dummies['Masculino'].astype(int)
sex_dummies['Feminino'] = sex_dummies['Feminino'].astype(int)
sex_dummies['Infantil'] = sex_dummies['Infantil'].astype(int)

# Concatenar as novas colunas binárias com o DataFrame original
data = pd.concat([data, sex_dummies], axis=1)

#Definição das bases de aprendizagem X e teste Y
def random_no_prop(data):
    test = sorted(random.sample([i for i in data.index.values],int(data.shape[0]*.3)+1))
    x = data.iloc[[i for i in data.index if i not in test],:]
    y = data.iloc[test,:]

    x = x.reset_index(drop=True)
    y = y.reset_index(drop=True)

    return x, y

#Definição das bases de aprendizagem X e teste Y matendo proporções originais dos tipos
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

    return x, y

def calc_entropy(data):

    quantity_1 = data.query('type == 1').shape[0]
    quantity_2 = data.query('type == 2').shape[0]
    quantity_3 = data.query('type == 3').shape[0]

    quantity_total = data.shape[0]
    
    probabilidade_1  = quantity_1/quantity_total
    probabilidade_2  = quantity_2/quantity_total
    probabilidade_3  = quantity_3/quantity_total
    
    entropy = -1*(probabilidade_1*math.log(probabilidade_1,2) + probabilidade_2*math.log(probabilidade_2,2) + probabilidade_3*math.log(probabilidade_3,2))
    return entropy

def calc_info_gain(leafe, initial_entropy, entropy_1, entropy_2):
    total = leafe.shape[0]

    wheight_1 = leafe.query('type == 1').shape[0]/total
    wheight_2 = leafe.query('type == 2').shape[0]/total
    wheight_3 = leafe.query('type == 3').shape[0]/total

    if i == [0, 1]:
        gain = -1*initial_entropy + (wheight_1*entropy_1 + wheight_2*entropy_2 + wheight_3*entropy_2)
    elif i == [1, 2]:
        gain = -1*initial_entropy + (wheight_1*entropy_1 + wheight_2*entropy_1 + wheight_3*entropy_2)
    return gain

def divide_pelo_limar(data,initial_entropy,threshold, name, array):

    leafe = pd.DataFrame(columns = {'type','value'})
    leafe1 = pd.DataFrame(columns = {'type','value'})
    leafe2 = pd.DataFrame(columns = {'type','value'})

    for i in range(data.shape[0]):
         if data[i] > threshold:
            leafe1 = leafe1.append({'type':data.at[i, 'type'],'value':data.at[i, name]},
                           ignore_index=True)
            leafe = leafe.append({'type':data.at[i, 'type'],'value':data.at[i, name]},
                            ignore_index=True)
         elif data[i] <= threshold:
            leafe2 = leafe2.append({'type':2,'value':data[i]},
                           ignore_index=True)
            leafe = leafe.append({'type':data.at[i, 'type'],'value':data.at[i, name]},
                            ignore_index=True)

    entropy_1 = calc_entropy(leafe1)
    entropy_2 = calc_entropy(leafe2)
    
    GH = calc_info_gain(leafe,initial_entropy,entropy_1, entropy_2, i)
    
    return GH,threshold

def caca_limiar(data,name,initial_entropy):
    
    valores = data.sort_values(name)
    valores.reset_index(drop=True,inplace=True)
    GH_limiar = []
    for i in range(valores.shape[0]-1):
        if valores.Classe[i] != valores.Classe[i+1]:
            array = [valores.Classe[i], valores.Classe[i+1]]
            limiar = 0
            limiar = (valores.iloc[i,1]+valores.iloc[i+1,1])/2
            GH_limiar.append(divide_pelo_limar(data,initial_entropy,limiar, array))
    GH_best,limiar_best = sorted(GH_limiar,reverse=True)[0]
    return GH_best,limiar_best

def rotula_rec(base,GHs,i):
    # @param base Dataframe of the specific attribute
    # @param GHs Dataframe of information gains and thresholds
    # @param i Dataframe position
    # return labels Vector of assigned labels

    rotulos = pd.DataFrame({})
    base.reset_index(drop=True,inplace=True)
    if i < GHs.shape[1]:
        for j in range(base.shape[0]):
            if base[GHs['Nome'][i]][j] >= GHs['Limiar'][i]:
                base['Classes_pred'][j] = 1
            else:
                base['Classes_pred'][j] = 2
    else:
        return
    rotulos = base.query('Classes_pred==1')
    rotulos = rotulos.append(base.query('Classes_pred==2'),ignore_index=True)
    
    rotula_rec(base.query('Classes_pred==1'),GHs,i+1)
    rotula_rec(base.query('Classes_pred==2'),GHs,i+1)
    
    return rotulos['Classes_pred']

