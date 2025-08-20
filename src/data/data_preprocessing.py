import numpy as np
import pandas as pd
import os

# fetch data from data/raw 
train_data = pd.read_csv('./data/raw/train.csv')
test_data = pd.read_csv('./data/raw/test.csv')

#transform data
# columns_to_transform = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

# def transform_cols(data, cols):
#     for col in cols:
#         data[col] = data[col].apply(lambda x: x * 2)
#     return data

# train_preprocessed = transform_cols(train_data, columns_to_transform)
# test_preprocessed = transform_cols(test_data, columns_to_transform)

def transform_cols(data): 
    data['sepal_length'] = data['sepal_length'].apply(lambda x:x*2) 
    data['sepal_width'] = data['sepal_width'].apply(lambda x:x*2) 
    data['petal_length'] = data['petal_length'].apply(lambda x:x*2) 
    data['petal_width'] = data['petal_width'].apply(lambda x:x*2) 
    return data 

train_preprocessed = transform_cols(train_data) 
test_preprocessed = transform_cols(test_data)

#stroe data inside data/preprocessed
data_path = os.path.join("data", "processed") # basically creating a folder/directory - data and creating another folder raw inside this folder
os.makedirs(data_path)

train_preprocessed.to_csv(os.path.join(data_path, 'train_processed.csv'), index=False)
test_preprocessed.to_csv(os.path.join(data_path, 'test_processed.csv'), index=False)

 