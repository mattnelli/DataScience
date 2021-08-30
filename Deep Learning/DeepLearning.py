#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import seaborn as sns
import sklearn 
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_validate, ShuffleSplit, LeaveOneOut
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
np.random.seed(66)


CleanData = pd.read_csv('Popular_Movies.csv')
MovieData = pd.read_csv('Popular_Movies.csv')


# # Reorganize

# In[2]:


#Reorganize the data set so the rottentomatoes in in the last column
columns = ['ID','Title','Rating','IMDb','Genre','Netflix','Amazon Prime Video','Rotten Tomatoes']
MovieData = MovieData.reindex(columns= columns)
CleanData = CleanData.reindex(columns= columns)


# # Data Standerdization

# In[3]:


#See if there are any duplicates in the data set
from scipy.stats.mstats import winsorize
print(f"There are {MovieData.duplicated().sum()} duplicated records.")


# In[4]:


#count NA values
MovieData.isnull().sum()


# In[5]:


MovieData.head(20)


# In[6]:


from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder() 

MovieData['Title'] =lb.fit_transform(MovieData['Title'])

MovieData['Genre'] =lb.fit_transform(MovieData['Genre'])



MovieData.head(20)


# Title may need to be removed since it has such a large number and affects the output very little.

# In[7]:


MovieData = MovieData.drop('ID', axis=1)
CleanData = CleanData.drop('ID',axis=1)
MovieData = MovieData.dropna()
CleanData = CleanData.dropna()


# In[8]:


MovieData = MovieData.drop('Title', axis=1)


# In[9]:


MovieData = MovieData.replace({'18+': 1,"16+":2,"13+":3,"7+":4,"all":5})


# Replace values of rating to be able to standardize them

# In[10]:


MovieData = MovieData.rename(columns={"Rotten Tomatoes": "output"})
CleanData =CleanData.rename(columns={"Rotten Tomatoes": "output"})


# In[11]:


#Scale data 
from sklearn import preprocessing

scaler = preprocessing.MinMaxScaler()
names = MovieData.columns
d = scaler.fit_transform(MovieData)
MovieData = pd.DataFrame(d, columns=names)
MovieData.head()


# In[12]:


MovieData['output'] = pd.cut(MovieData['output'], bins=2,labels=[0,1] ) # equal interval discretization


# In[13]:



MovieData.head(20)


# # Visualization

# for feature in CleanData:
# 
#     CleanData.groupby(feature)['output'].mean().plot.bar()
#     plt.xlabel(feature)
#     plt.ylabel('output')
#     plt.show()

# In[14]:


sns.scatterplot(data=CleanData, x="IMDb",y="output")


# In[15]:


CleanData = CleanData.sort_values(by=['output'])
sns.set(rc={'figure.figsize':(18,16)})
sns.set(color_codes=True)
sns.scatterplot( data=CleanData, x="Genre", y="output",sizes=(20, 200), hue_norm=(0, 7), legend="full")


# In[16]:


CleanData['IMDb'] = pd.to_numeric(CleanData['IMDb'])
CleanData['output'] = pd.to_numeric(CleanData['output'])
CleanData.info()


# In[17]:


sns.set(rc={'figure.figsize':(40,16)})
sns.barplot( data=CleanData, x="Netflix", y="output")


# In[18]:


sns.set(rc={'figure.figsize':(40,16)})
sns.barplot( data=CleanData, x="Amazon Prime Video", y="output")


# In[19]:


sns.histplot(CleanData['output'])


# In[20]:


sns.histplot(CleanData['IMDb'])


# In[21]:


sns.lineplot(data=CleanData, x="Rating", y= "output")


# In[22]:


sns.lineplot(data=CleanData, x="Title", y= "output")


# # Deep Learning 

# In[25]:


num_vars = MovieData.select_dtypes(['int64', 'float64']).columns
X = MovieData[num_vars]


y = MovieData.output

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
X_test.shape
y_train.shape

print(X_test)


# In[26]:


print(y_train)


# In[27]:


## Attribute Normalization and Standardization

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_train.shape


# In[79]:


import warnings
warnings.filterwarnings('ignore')
from keras.models import Sequential
from keras.layers import Dense

#creating the model
model = Sequential()
model.add(Dense(20, activation='tanh', input_shape=(5, ))) # 
model.add(Dense(20, activation='tanh')) 
model.add(Dense(20, activation='tanh')) 
model.add(Dense(1, activation='sigmoid'))
model.summary()


# In[80]:


model.get_config()
model.get_weights()


# In[81]:


get_ipython().run_line_magic('load_ext', 'tensorboard')
from datetime import datetime
from packaging import version

import tensorflow as tf
from tensorflow import keras

logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=2000, batch_size=1)


# In[82]:


y_pred = model.predict(X_test)
model.evaluate(X_test, y_test, verbose=1)


# In[ ]:


#ytest = pd.DataFrame(y_test)
#ytest["predicted"] = y_pred
#ytest.head()

ypred = pd.DataFrame(y_pred)
ypred.drop(ypred.tail(1).index,inplace=True)

ytest = pd.DataFrame(y_test)


# In[ ]:


ypred.head()


# In[ ]:


y_test.head()


# In[ ]:




