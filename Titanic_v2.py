#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 15:07:46 2017

@author: gaurav
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score
from sklearn.metrics import cohen_kappa_score
from sklearn.ensemble import BaggingClassifier

train = pd.read_csv('/home/gaurav/Desktop/titanic/train.csv')
test1 = pd.read_csv('/home/gaurav/Desktop/titanic/test.csv')
data = pd.concat([train,test],ignore_index=True)
labels=train["Survived"]

def sex(value):
    if value == "male":
        return 0
    else:
        return 1
    
def embarked(value):
    if value == "C":
        return 0
    elif value =="Q":
        return 1
    else:
        return


data["Sex"] = data["Sex"].apply(sex)

data["Embarked"] = data["Embarked"].apply(embarked)

data.drop('Survived',inplace=True,axis=1)

data["TitleClean"] = data["Name"].str.extract('(\w*\.)', expand=True)


data['TitleClean'].value_counts()

remove=['Name','Cabin','Ticket','Embarked']
for column in remove:
    data = data.drop(column, 1)

for c in data.columns:
    if data[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(data[c].values.astype('str')))
        data[c]= lbl.transform(list(data[c].values.astype('str')))
        
for col in data.columns.values:
    if data[col].dtypes != 'object' and data[col].isnull().sum() > 0:
        data[col].fillna(data[col].mean(), inplace=True)

data.drop('PassengerId',inplace=True,axis=1)
train = data.ix[0:890]
test = data.ix[891:1308]
labels

skf = cv.KFold(n=891,n_folds=3, shuffle=True)
score_metric = 'roc_auc'
scores = {}

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
clf = ExtraTreesClassifier(n_estimators=200)
clf = clf.fit(train, labels)

features = pd.DataFrame()
features['feature'] = data.columns
features['importance'] = clf.feature_importances_

features
from sklearn import cross_validation as cv
from sklearn.cross_validation import StratifiedKFold
forest = RandomForestClassifier()

parameter_grid = {'max_depth':[1,2,3,4,5],'n_estimators': [50,100,150,200,250],'criterion': ['gini','entropy']}

cross_validation = StratifiedKFold(labels, n_folds=5)

grid_search = GridSearchCV(forest,
                           param_grid=parameter_grid,
                           cv=cross_validation)

grid_search.fit(train, labels)

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))


pipeline = grid_search
output = pipeline.predict(test).astype(int)
df_output = pd.DataFrame()
df_output['PassengerId'] = test1['PassengerId']
df_output['Survived'] = output
df_output[['PassengerId','Survived']].to_csv('output3.csv',index=False)




#skf = cv.KFold(n=891,n_folds=3, shuffle=True)
#score_metric = 'roc_auc'
#scores = {}

#def score_model(model):
#    return cv.cross_val_score(model, train, labels, cv=skf, scoring=score_metric)

#scores['tree'] = score_model(tree.DecisionTreeClassifier()) 

#from sklearn import ensemble

#scores['ada_boost'] = score_model(ensemble.AdaBoostClassifier())

#parameters = {'n_estimators': [10,20,30,40,50],
 #              'learning_rate' :[0.2,0.4,0.6,0.8,1,1.2]
  #           }

#ada = ensemble.AdaBoostClassifier()
#cross_validation = StratifiedKFold(labels, n_folds=5)

#grid_search = GridSearchCV(ada,
 #                          param_grid=parameters,
  #                         cv=cross_validation)

#grid_search.fit(train, labels)

#print('Best score: {}'.format(grid_search.best_score_))
#print('Best parameters: {}'.format(grid_search.best_params_))



#pipeline = grid_search
#output = pipeline.predict(test).astype(int)
#df_output = pd.DataFrame()
#df_output['PassengerId'] = test1['PassengerId']
#df_output['Survived'] = output
#df_output[['PassengerId','Survived']].to_csv('output2.csv',index=False)





