import numpy as np
import pandas as pd
import json

import pickle
from sklearn.metrics import accuracy_score, precision_score

#load model
clf = pickle.load(open('model.pkl','rb'))

#load test data
test_data = pd.read_csv('./data/processed/test_processed.csv')

X_test = test_data.iloc[:, 0:-1].values  # we already divided the entire dataset in train and test - 2 csv, 
y_test = test_data.iloc[:,-1].values

#make predictions
y_pred = clf.predict(X_test)

#calculate accuracy score 
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')

#writing in json format so that we can dump it to json file.
metrics_dict = {
    'accuracy': accuracy,
    'precision': precision
}

with open('reports/metrics.json', 'w') as file:
    json.dump(metrics_dict, file, indent=4)
# basically we dumping the metrics_dict we created into a json file with filename as 'metrics.json'

#now add to the dvc stage
#dvc stage add -n model_evaluation -d src/model_evaluation.py -d model.pkl --metrics metrics.json python src/model_evaluation.py
 #WE USE DEPENDENCIES -d metrics metrics.json as this is a special function which will help us track the metrics with command dvc metrics
