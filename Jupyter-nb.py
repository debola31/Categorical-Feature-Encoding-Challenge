# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%%
from IPython import get_ipython

#%%
from zipfile import ZipFile
import pandas as pd
import numpy as np
import subprocess
import os


#%%
def fetchdata():
    cli_command = "kaggle competitions download -c cat-in-the-dat -w"
    subprocess.run(cli_command, shell=True) # Get data from kaggle
    zip_files = [files for files in os.listdir() if files.endswith('.zip')]
    # Unzip files if not already unzipped
    unzipped_filenames = [files.replace('.zip','') for files in zip_files]
    if not all(x in os.listdir() for x in unzipped_filenames):
        for file in zip_files:
            with ZipFile(file, 'r') as zip:
                zip.extractall()


#%%
fetchdata()
data = pd.read_csv('train.csv')


#%%
print(data.head())
print(data.info())
print(data.describe())


#%%
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
data.hist(bins=50, figsize=(20,15))
plt.show()


#%%
categorical_data = data.drop('target', axis=1)


#%%
categorical_data_labels = data['target'].copy()


#%%
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
data_prepared = cat_encoder.fit_transform(data)


#%%
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
tree_reg = DecisionTreeRegressor()
tree_reg.fit(data_prepared, categorical_data_labels)


#%%
data_predictions = tree_reg.predict(data_prepared)
tree_mse = mean_squared_error(categorical_data_labels, data_predictions)
tree_rmse = np.sqrt(tree_mse)
print(tree_mse)


#%%
from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, data_prepared, categorical_data_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)


#%%
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(tree_rmse_scores)


#%%



