#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from scipy.spatial.distance import cdist
from sklearn.pipeline import make_pipeline

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, ShuffleSplit
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

train = pd.read_csv('Disease_Prediction_Training.csv')
test = pd.read_csv('Disease_Prediction_Testing.csv')


# # Missing Values
# 

# In[2]:


from scipy.stats.mstats import winsorize
print(f"There are test {test.isna().sum().sum()} missing values.")
print(f"There are test {test.duplicated().sum()} duplicated records.")

print(f"There are Train {train.isna().sum().sum()} missing values.")
print(f"There are Train {train.duplicated().sum()} duplicated records.")


# In[3]:


df = pd.concat([test.assign(ind="test"), train.assign(ind="train")])


# In[4]:


print(f"There are {df.isna().sum().sum()} missing values.")
print(f"There are {df.duplicated().sum()} duplicated records.")


# In[5]:


df.isnull().sum()


# In[6]:


df.drop_duplicates()


# # EDA

# In[7]:


df.head(10)


# In[8]:


for feature in df:
    
    df.groupby(feature)['Disease'].mean().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('Disease')
    plt.show()


# In[9]:


from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder() 

df['Glucose'] =lb.fit_transform(df['Glucose'])
df['Gender'] =lb.fit_transform(df['Gender'])
df['Cholesterol'] =lb.fit_transform(df['Cholesterol'])
df['ind'] =lb.fit_transform(df['ind'])


# In[10]:


from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
names = df.columns
d = scaler.fit_transform(df)
df = pd.DataFrame(d, columns=names)
df.head()


# In[11]:


test, train = df[df["ind"].eq(0)], df[df["ind"].eq(1)]
test = test.drop('ind', axis=1)
train = train.drop('ind', axis=1)
train = train.drop('ID', axis=1)
test = test.drop('ID', axis=1)
test = test.drop('Disease', axis=1)
test.head()


# In[12]:


train.isna().sum().sum()
train.duplicated().sum()


# In[13]:


X = train.iloc[:, :-1].values
# X = StandardScaler().fit_transform(X)
y = train.iloc[:, -1]
X.shape
y.shape


# In[14]:


from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(X)
pd.DataFrame(X).applymap(lambda x: abs(x))


# # Deep Learning

# Setting up data set to train

# In[15]:


num_vars = train.select_dtypes(['int64', 'float64']).columns


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)
X_test.shape
y_train.shape

print(y_train)


# In[16]:


## Attribute Normalization and Standardization

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_train.shape


# # BNB bernoulli Perdiction

# Previous code from HW3

# In[17]:


bnb = BernoulliNB()
bnb_pred = bnb.fit(X, y).predict(X)
metrics.accuracy_score(y, bnb_pred)


# In[18]:


bnb_pred = bnb.predict(test)
BNBsubmission = pd.DataFrame({
        "BNB": bnb_pred
    })
BNBsubmission.to_csv('BNBsubmission.csv', index=False)


# In[19]:


X = train.drop('Disease', axis=1)
y = train['Disease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=16)
classifier = KNeighborsClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(f"Accuracy: {round(metrics.accuracy_score(y_test, y_pred)*100, 2)}%")
df_confusion = pd.crosstab(y_test, y_pred)


# In[20]:


classifier.get_params()


# In[21]:


print(y_pred)
y_pred = classifier.predict(test)
KNNsubmission = pd.DataFrame({
        "Sick": y_pred
    })
KNNsubmission.to_csv('KNNsubmission.csv', index=False)


# # KNN with Pipeline

# Previous code from HW3

# In[22]:


## Use Pipeline to Streamline the Analysis

knn_pipe = make_pipeline(StandardScaler(), KNeighborsClassifier())
knn_pipe.fit(X_train, y_train)
', '.join(dir(knn_pipe))
knn_pipe.get_params()
knn_pipe.set_params(kneighborsclassifier__n_neighbors=8)
pipe_pred = knn_pipe.predict(X_test)
pd.Series(pipe_pred).value_counts()


# In[23]:


knn_pipe.get_params()


# In[24]:


pipe_pred = knn_pipe.predict(test)
submission = pd.DataFrame({
        "knn": pipe_pred
    })
#EMsubmission.to_csv('KNNPsubmission.csv', index=False)
#print(pipe_pred)


# In[25]:


## Get Repeated Hold Out Accurary of Model

cv = ShuffleSplit(n_splits=100, test_size=0.3, random_state=16)
from sklearn.model_selection import KFold
cv = KFold(n_splits=10, shuffle=True, random_state=16)
cross_val_score(knn_pipe, X_train, y_train, cv=cv).mean()


# # Ensemble Learning - Bagging

# Previous code from HW3.

# In[26]:


## Ensemble Learning - Bagging

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

cart = DecisionTreeClassifier()
num_trees = 100
modelb = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=16)
results = cross_val_score(modelb, X_train, y_train, cv=cv)
print(f"Accuracy: {round(results.mean()*100, 2)}%")


# In[27]:


modelb.get_params()


# In[56]:


modelb.fit(X_train, y_train)
y_predictem = modelb.predict(test)
submission = pd.DataFrame({
        "EL": y_predictem
    })
#EMsubmission.to_csv('EMsubmission.csv', indexy_predictem=False)
print(y_predictem)


# # Random Forest

# In[29]:


## Random Forest

from sklearn.ensemble import RandomForestClassifier

modelrf = RandomForestClassifier(n_estimators=num_trees, max_features=5, random_state=16)
results = cross_val_score(modelrf, X_train, y_train, cv=cv)
print(f"Accuracy: {round(results.mean()*100, 2)}%")
modelrf


# In[30]:


modelrf.get_params()


# # ANN0

# In[31]:


import warnings
warnings.filterwarnings('ignore')
from keras.models import Sequential
from keras.layers import Dense

#creating the model
ANN0 = Sequential()

ANN0.add(Dense(1, activation='sigmoid',input_shape=(11, )))
ANN0.summary()


# Using the Sigmoid function as the only output. This allows for a more smooth transition between output values and is good to use in the final layer. 

# In[32]:


ANN0.get_config()
ANN0.get_weights()


# In[33]:


ANN0.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
history = ANN0.fit(X_train, y_train, epochs=50, batch_size=5)


# Running this through 50 times seems to be a good number of itteration. Running 8 does not allow the model to fully reach the highest accuracy. Annything greater doesn't add additional accuuracy. 

# In[34]:


y_pred = ANN0.predict(X_test)
ANN0.evaluate(X_test, y_test, verbose=1)


# In[73]:


y_ANN0 = ANN0.predict(test)


# In[75]:


y_ANN0 =y_ANN0.transpose()
y_ANN0 = pd.DataFrame (y_ANN0, columns = ["ANN"])
print(y_ANN0)


# # ANN1

# In[37]:


import warnings
warnings.filterwarnings('ignore')
from keras.models import Sequential
from keras.layers import Dense
#creating the model
ANN1 = Sequential()



ANN1.add(Dense(6, activation='relu', input_shape=(11, ))) #  
ANN1.add(Dense(1, activation='sigmoid'))
ANN1.summary()


# Rectifier provides a good slope for hidden layers. The output layer is still sigmoid. The hidden layer has more nodes at 6 this allows for increased accuracy without over complicating the model.

# In[38]:


ANN1.get_config()
ANN1.get_weights()


# In[39]:


ANN1.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
history = ANN1.fit(X_train, y_train, epochs=50, batch_size=5)


# In[40]:


y_ANN1 = ANN1.predict(X_test)
ANN1.evaluate(X_test, y_test, verbose=1)


# In[41]:


y_ANN1 = ANN1.predict(test)


# In[79]:


y_ANN1 =y_ANN1.transpose()
y_ANN1 = pd.DataFrame (y_ANN1, columns = ["ANN"])
print(y_ANN1)


# # ANN2

# In[42]:


#creating the model
ANN2 = Sequential()



ANN2.add(Dense(6, activation='relu', input_shape=(11, ))) #  
ANN2.add(Dense(6, activation='relu'))
ANN2.add(Dense(1, activation='sigmoid'))
ANN2.summary()


# In[43]:


ANN1.get_config()
ANN1.get_weights()


# In[44]:


ANN2.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
history = ANN2.fit(X_train, y_train, epochs=50, batch_size=5)


# 50 runs to have the accuracy level out.

# In[45]:


y_pred = ANN2.predict(X_test)
ANN2.evaluate(X_test, y_test, verbose=1)


# In[46]:


y_ANN2 = ANN2.predict(test)


# The difference between 1 and 2 hidden layers is not that noticable. It has diminishing returns with this data set.

# In[83]:


ANN2.fit(X_train, y_train)
y_ANN2 = ANN2.predict(test)


# In[85]:


y_ANN2 = y_ANN2.transpose()
y_ANN2 = pd.DataFrame (y_ANN2, columns = ["ANN"])
print(y_ANN2)


# In[86]:


modelrf.fit(X_train, y_train)
y_predict = modelrf.predict(test)
submission = pd.DataFrame({
        "RFC": y_predict,
        "EL": y_predictem,
        "KNN": pipe_pred,
        "BNB": bnb_pred,
        "ANN0":y_ANN0["ANN"],
        "ANN1":y_ANN1["ANN"],
        "ANN2":y_ANN2["ANN"],
    })


# In[90]:


submission.to_csv('hwsubmission.csv', index=False)


# In[91]:


outputtable = {"BNB": ["5 NEIGHBORS","minkowski","71%"],
               "KNN": ["8 NEIGHBORS","minkowski", "60%"],
               "EMSAMBLE LEARNING": ["DecisionTreeClassifier()","gini", "70%"],
               "RANDOM FOREST": ["max_features=5","gini", "70%"],
               "ANN0": ["0 HIDDEN LAYERS", "1 NODE", "70%"],
               "ANN1": ["1 HIDDEN LAYER", "6 NODE & 1 NODE", "70%"],
               "ANN2":["2 HIDDEN LAYERS", "6 NODE & 1 NODE", "70%"]}


# In[92]:


df = pd.DataFrame (outputtable, columns = ["BNB","KNN","EMSAMBLE LEARNING","RANDOM FOREST","ANN0","ANN1","ANN2"])


# In[93]:


print (df)


# BNB turned out to be the most accurate. ANN was a close second with 70. The artifical nueral net being very similar accross the 3 different types seems weird. I thought they would be much more different but the data set may not be large enough to take advantage of the approach.

# In[ ]:




