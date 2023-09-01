import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

url = 'https://raw.githubusercontent.com/lac1908/mlclass/master/03_Validation/abalone_dataset.csv'
pd.read_csv(url)

data = pd.read_csv(url, sep=',')
data

data.info()

data.head()

def string2int(sex):
  if sex == 'M':
    sex = 0
  elif sex == 'F':
    sex = 1
  else:
    sex = 2
  return sex

data['sex'] = data.sex.apply(string2int)

data.head()

data.info()

columns = ['sex', 'length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight']
X = data[columns]
y = data.type
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)

clf = svm.SVC(kernel='linear')

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))