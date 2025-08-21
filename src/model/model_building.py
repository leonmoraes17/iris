import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import pickle
import os
import yaml

#Params.yaml file parameter hypertuning 
max_iter = yaml.safe_load(open('params.yaml', 'r'))['model_building']['max_iter']
penalty = yaml.safe_load(open('params.yaml', 'r'))['model_building']['penalty']

#fetch data from source, since we building model and not evaluating model we only fetching training data
train_data = pd.read_csv('./data/processed/train_processed.csv')

X_train = train_data.iloc[:, 0:-1].values  # wealready divided the entire dataset in train and test - 2 csv, 
y_train = train_data.iloc[:,-1].values

clf = LogisticRegression(max_iter=max_iter, penalty= penalty)
clf.fit( X_train, y_train)

#save the model
pickle.dump(clf, open('model.pkl', 'wb'))

#now add to the dvc stage
#dvc stage add -n model_building -d src/model_building.py -d data/processed -o model.pkl python src/model_building.py
