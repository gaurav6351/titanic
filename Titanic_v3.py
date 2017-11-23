#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 15:07:46 2017

@author: gaurav
"""
# Import the needed referances
import pandas as pd
import numpy as np
import csv as csv

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier


#Shuffle the datasets
from sklearn.utils import shuffle

#Learning curve
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

train_dataset = pd.read_csv('/home/gaurav/Desktop/titanic/train.csv')
test_dataset = pd.read_csv('/home/gaurav/Desktop/titanic/test.csv')

train_dataset.dtypes.value_counts()

###Analyze the data

#Check for missing data & list them 
nas = pd.concat([train_dataset.isnull().sum(), test_dataset.isnull().sum()], axis=1, keys=['Train Dataset', 'Test Dataset']) 
print('Nan in the data sets')
print(nas[nas.sum(axis=1) > 0])

# Class vs Survived
print(train_dataset[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))

# sex vs Survived
print(train_dataset[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))


# SibSp vs Survived
#Sibling = brother, sister, stepbrother, stepsister
#Spouse = husband, wife (mistresses and fiancés were ignored)
print(train_dataset[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))

# Parch vs Survived
#Parent = mother, father
#Child = daughter, son, stepdaughter, stepson
#Some children travelled only with a nanny, therefore parch=0 for them.
print(train_dataset[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False))

#Manage Age
train_random_ages = np.random.randint(train_dataset["Age"].mean() - train_dataset["Age"].std(),
                                          train_dataset["Age"].mean() + train_dataset["Age"].std(),
                                          size = train_dataset["Age"].isnull().sum())


test_random_ages = np.random.randint(test_dataset["Age"].mean() - test_dataset["Age"].std(),
                                          test_dataset["Age"].mean() + test_dataset["Age"].std(),
                                          size = test_dataset["Age"].isnull().sum())

train_dataset["Age"][np.isnan(train_dataset["Age"])] = train_random_ages
test_dataset["Age"][np.isnan(test_dataset["Age"])] = test_random_ages

train_dataset['Age'] = train_dataset['Age'].astype(int)
test_dataset['Age']    = test_dataset['Age'].astype(int)

#Embarked

train_dataset["Embarked"].fillna('S', inplace=True)
test_dataset["Embarked"].fillna('S', inplace=True)

train_dataset['Port'] = train_dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
test_dataset['Port'] = test_dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
del train_dataset['Embarked']
del test_dataset['Embarked']

#Fare
test_dataset["Fare"].fillna(test_dataset["Fare"].median(), inplace=True)

#Engineer New Features

train_dataset['Has_Cabin'] = train_dataset["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
test_dataset['Has_Cabin'] = test_dataset["Cabin"].apply(lambda x: 0 if type(x) == float else 1)


full_dataset = [train_dataset, test_dataset]

##engineer the family size feature
for dataset in full_dataset:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1


for dataset in full_dataset:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    

train_dataset['Title'] = train_dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
test_dataset['Title'] = test_dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

for dataset in full_dataset:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
    
for dataset in full_dataset:
    dataset['FamilySizeGroup'] = 'Small'
    dataset.loc[dataset['FamilySize'] == 1, 'FamilySizeGroup'] = 'Alone'
    dataset.loc[dataset['FamilySize'] >= 5, 'FamilySizeGroup'] = 'Big'
    
for dataset in full_dataset:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
    
for dataset in full_dataset:    
    dataset.loc[ dataset['Age'] <= 14, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 14) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4

for dataset in full_dataset:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
    

# map the new features
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
family_mapping = {"Small": 0, "Alone": 1, "Big": 2}
for dataset in full_dataset:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['FamilySizeGroup'] = dataset['FamilySizeGroup'].map(family_mapping)

# engineer a new  features
for dataset in full_dataset:
    dataset['IsChildandRich'] = 0
    dataset.loc[(dataset['Age'] <= 0) & (dataset['Pclass'] == 1 ),'IsChildandRich'] = 1  
    dataset.loc[(dataset['Age'] <= 0) & (dataset['Pclass'] == 2 ),'IsChildandRich'] = 1  

del train_dataset['Name']
del test_dataset['Name']

del train_dataset['SibSp']
del test_dataset['SibSp']

del train_dataset['Parch']
del test_dataset['Parch']

del train_dataset['FamilySize']
del test_dataset['FamilySize']

#del train_dataset['FamilySizeGroup']
#del test_dataset['FamilySizeGroup']

del train_dataset['Cabin']
del test_dataset['Cabin']

# Delete Ticket column from datasets  (No need for them in the analysis)
del train_dataset['Ticket']
del test_dataset['Ticket']


print('train dataset: %s, test dataset %s' %(str(train_dataset.shape), str(test_dataset.shape)) )


del train_dataset['PassengerId']
del test_dataset['PassengerId']


#X_train = train_dataset.drop("Survived",axis=1).as_matrix()
#Y_train = train_dataset["Survived"].as_matrix()
#X_test  = test_dataset.drop("PassengerId",axis=1).copy().as_matrix()

X_train = train_dataset.drop("Survived",axis=1)
Y_train = train_dataset["Survived"]
X_test  = test_dataset.drop("PassengerId",axis=1).copy()

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)

from sklearn.preprocessing import MinMaxScaler,PolynomialFeatures

all_data = pd.concat((X_train,
                          X_test), ignore_index=True)

scaler = MinMaxScaler()
scaler.fit(all_data)
all_data=scaler.transform(all_data)
poly = PolynomialFeatures(2)
all_data=poly.fit_transform(all_data)

X_train = all_data[:train_dataset.shape[0]]
X_test = all_data[train_dataset.shape[0]:]
    ##
print(X_train.shape)
print(X_train.shape)
print(X_test.shape)


from sklearn import cross_validation as cv
from sklearn.cross_validation import StratifiedKFold
from sklearn.model_selection import GridSearchCV
forest = RandomForestClassifier()

parameter_grid = {'max_depth':[1,2,3,4,5],'n_estimators': [50,100,150,200,250],'criterion': ['gini','entropy']}

cross_validation = StratifiedKFold(Y_train, n_folds=5)

grid_search = GridSearchCV(forest,
                           param_grid=parameter_grid,
                           cv=cross_validation)

grid_search.fit(X_train, Y_train)

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))

test_dataset1 = pd.read_csv('/home/gaurav/Desktop/titanic/test.csv')

pipeline = grid_search
output = pipeline.predict(X_test).astype(int)
df_output = pd.DataFrame()
df_output['PassengerId'] = test_dataset1['PassengerId']
df_output['Survived'] = output
df_output[['PassengerId','Survived']].to_csv('output4.csv',index=False)
