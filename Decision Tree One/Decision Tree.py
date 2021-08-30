#!/usr/bin/env python
# coding: utf-8

# # Import Libraries and Data
# 

# In[1]:


import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
Weather = pd.read_csv('Weather_Forecast_Training.csv')
WeatherTesting = pd.read_csv('Weather_Forecast_Testing.csv')


# # Missing Data

# In[2]:


from scipy.stats.mstats import winsorize
print(f"There are {Weather.isna().sum().sum()} missing values.")
print(f"There are {Weather.duplicated().sum()} duplicated records.")


# Remove the 19 duplicate values

# In[3]:


#count NA values
Weather.isnull().sum()


# In[4]:


plt.figure(figsize=(12,9))
plt.subplot(1,2,1)
plt.title("Training Data")
sns.heatmap(Weather.isnull(),yticklabels=False,cbar=False,cmap="viridis")


# Evaporation, Sunshine and cloud have the bulk of the missing data which can also be seen above
# 

# In[5]:


Weather.head()    


# In[6]:


Weather.shape


# In[7]:


WeatherTesting.shape


# In[8]:


obj_col = Weather.select_dtypes('float').columns
for col in obj_col:
    print(f"{col} has {Weather[col].nunique()} unique values")


# Seen below Data is evenly split

# In[9]:


sns.countplot(y=Weather["RainTomorrow"])


# In[10]:


sns.countplot(y=Weather["RainToday"])


# Data above shows it is well balanced 

# In[11]:


from sklearn.impute import SimpleImputer

meanimputer = SimpleImputer(missing_values=np.nan, strategy = 'mean')
Weather.Sunshine = meanimputer.fit_transform(Weather['Sunshine'].values.reshape(-1,1))[:,0]
Weather.Evaporation = meanimputer.fit_transform(Weather['Evaporation'].values.reshape(-1,1))[:,0]
Weather.Cloud = meanimputer.fit_transform(Weather['Cloud'].values.reshape(-1,1))[:,0]
Weather.MinTemp = meanimputer.fit_transform(Weather['MinTemp'].values.reshape(-1,1))[:,0]
Weather.MaxTemp = meanimputer.fit_transform(Weather['MaxTemp'].values.reshape(-1,1))[:,0]
Weather.Rainfall = meanimputer.fit_transform(Weather['Rainfall'].values.reshape(-1,1))[:,0]
Weather.Humidity = meanimputer.fit_transform(Weather['Humidity'].values.reshape(-1,1))[:,0]
Weather.Pressure = meanimputer.fit_transform(Weather['Pressure'].values.reshape(-1,1))[:,0]
Weather.Temp = meanimputer.fit_transform(Weather['Temp'].values.reshape(-1,1))[:,0]


WeatherTesting.Sunshine = meanimputer.fit_transform(WeatherTesting['Sunshine'].values.reshape(-1,1))[:,0]
WeatherTesting.Evaporation = meanimputer.fit_transform(WeatherTesting['Evaporation'].values.reshape(-1,1))[:,0]
WeatherTesting.Cloud = meanimputer.fit_transform(WeatherTesting['Cloud'].values.reshape(-1,1))[:,0]
WeatherTesting.MinTemp = meanimputer.fit_transform(WeatherTesting['MinTemp'].values.reshape(-1,1))[:,0]
WeatherTesting.MaxTemp = meanimputer.fit_transform(WeatherTesting['MaxTemp'].values.reshape(-1,1))[:,0]
WeatherTesting.Rainfall = meanimputer.fit_transform(WeatherTesting['Rainfall'].values.reshape(-1,1))[:,0]
WeatherTesting.Humidity = meanimputer.fit_transform(WeatherTesting['Humidity'].values.reshape(-1,1))[:,0]
WeatherTesting.Pressure = meanimputer.fit_transform(WeatherTesting['Pressure'].values.reshape(-1,1))[:,0]
WeatherTesting.Temp = meanimputer.fit_transform(WeatherTesting['Temp'].values.reshape(-1,1))[:,0]

Weather.dropna(inplace=True)
WeatherTesting.dropna(inplace=True)

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder() 

WeatherTesting['RainToday'] = lb.fit_transform(WeatherTesting['RainToday'])
WeatherTesting['Location'] =lb.fit_transform(WeatherTesting['Location'])
WeatherTesting.drop('WindGustDir', inplace=True, axis=1)
WeatherTesting.drop('WindDir', inplace=True, axis=1)
WeatherTesting.drop('ID', inplace=True, axis=1)


# Above i had to take care of all the messing values in both data sets. Below i had to remake true false and location values to be numerical values not strings

# In[12]:


data = Weather.copy()
datatesting = WeatherTesting.copy()
from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder() 
lbdata = LabelEncoder() 
data['RainTomorrow'] = lb.fit_transform(data['RainTomorrow'])
data['RainToday'] = lb.fit_transform(data['RainToday'])
data['Location'] =lb.fit_transform(data['Location'])
data.drop('WindGustDir', inplace=True, axis=1)
data.drop('WindDir', inplace=True, axis=1)
data.head()


# In[13]:


print(f"There are {data.isna().sum().sum()} missing values.")
print(f"There are {data.duplicated().sum()} duplicated records.")


# In[14]:



for feature in Weather:
    
    data.groupby(feature)['RainTomorrow'].mean().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('RainTomorrow')
    plt.show()


# Sunshine, Wind guest, Pressure, clouds and max temp look to be the biggest indicator of ran in the future. 
# Location, Rainfal, mintemp, and wind directions looks to have a minimal impact.

# # Run desicion Tree

# In[15]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
np.random.seed(66)


# In[16]:


## Model Training
X = data.drop('RainTomorrow', axis=1)
y = data['RainTomorrow']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
print(f"train data size is {X_train.shape}")

clf = DecisionTreeClassifier()
clf


# In[17]:


## Model Predicting

clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
clf.tree_.max_depth
print(f"Accuracy: {round(metrics.accuracy_score(y_test, y_pred)*100)}%")


# # Test different parameters

# Testing different parameters and datasets

# In[18]:


param_grid = {'criterion': ['gini', 'entropy'],
              'min_samples_split': [2, 10,20],
              'max_depth': [5, 10, 20, 25, 30],
              'min_samples_leaf': [1, 5,10],
              'max_leaf_nodes': [2, 5,10,20]}
grid = GridSearchCV(clf, param_grid, cv=10, scoring='accuracy')
grid.fit(X_train, y_train)


# In[19]:


print(grid.best_score_)
for hps, values in grid.best_params_.items():
  print(f"{hps}: {values}")


# Applying the hold out method gives us slightly better perfomance. 

# In[20]:


## Repeated Hold-Out Method
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_validate, ShuffleSplit, LeaveOneOut
from sklearn import metrics
bstrap = ShuffleSplit(n_splits=10, test_size=0.3, random_state=16)
grid_bstrap = GridSearchCV(clf, param_grid, cv=bstrap)
grid_bstrap.fit(X_train, y_train)


# In[21]:


print(f"Accuracy: {round(grid_bstrap.best_score_*100, 2)}%")
for key, value in grid_bstrap.best_params_.items():
  print(f"Hyperparameter: {key}; Value: {value}")


# In[22]:


clfinal = DecisionTreeClassifier(max_depth = 5, max_leaf_nodes = 20, min_samples_leaf = 1, min_samples_split =2)
clfinal.fit(X_train,y_train)
x_test = datatesting.values
y_pred = clfinal.predict(x_test)
submission = pd.DataFrame({
        "RainTomorrow": y_pred
    })


# In[23]:


submission
submission.to_csv('submission.csv', index=False)


# In[24]:


submission


# In[ ]:




