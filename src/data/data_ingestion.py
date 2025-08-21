import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import joblib
import os
import yaml

test_size = yaml.safe_load(open('params.yaml', 'r'))['data_ingestion']['test_size'] #how to connect the params.yaml file which has 
# data_ingestion stage so that we can tweak the test_size in params.yaml and it automatically gets udpated here

df = pd.read_csv('../iris/irisdataset.csv')
df = df.rename(columns={"sepal length (cm)": "sepal_length", "sepal width (cm)": "sepal_width","petal length (cm)": 'petal_length',"petal width (cm)": "petal_width" })

# X = df.drop(columns=['target'])
# y = df['target']
train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)

data_path = os.path.join("data", "raw") # basically creating a folder/directory - data and creating another folder raw inside this folder
os.makedirs(data_path)

train_data.to_csv(os.path.join(data_path, 'train.csv'),index = False)
test_data.to_csv(os.path.join(data_path, 'test.csv'), index = False)

# SEE SESSION 10 IMPROVING ML PIPELINES -MLOPS REVISTED TO SEE HOW TO MKAE THIS CODE MODULAR 42 MINS
#shows to write try and except and loggin error and functions. 