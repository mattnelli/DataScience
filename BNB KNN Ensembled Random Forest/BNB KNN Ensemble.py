#!/usr/bin/env python
# coding: utf-8

# # Import Data

# In[2]:


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

# In[3]:


from scipy.stats.mstats import winsorize
print(f"There are test {test.isna().sum().sum()} missing values.")
print(f"There are test {test.duplicated().sum()} duplicated records.")

print(f"There are Train {train.isna().sum().sum()} missing values.")
print(f"There are Train {train.duplicated().sum()} duplicated records.")


# In[4]:


df = pd.concat([test.assign(ind="test"), train.assign(ind="train")])


# In[5]:



print(f"There are {df.isna().sum().sum()} missing values.")
print(f"There are {df.duplicated().sum()} duplicated records.")


# In[6]:


df.isnull().sum()


# In[45]:


df.drop_duplicates()


# The missing data is from the training set not having an id value so they are fine. The 

# # EDA

# In[46]:


df.head(10)


# Data above is not weighted well. Age,height and weight have large vartions that will mess up machine learning algthoms. 

# In[47]:


for feature in df:
    
    df.groupby(feature)['Disease'].mean().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('Disease')
    plt.show()


# Standerdize age, height,Weight,high blood pressure, low blood pressure
# Change glucose,cholesterol, gender

# In[48]:


from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder() 

df['Glucose'] =lb.fit_transform(df['Glucose'])
df['Gender'] =lb.fit_transform(df['Gender'])
df['Cholesterol'] =lb.fit_transform(df['Cholesterol'])
df['ind'] =lb.fit_transform(df['ind'])


# In[49]:



scaler = preprocessing.MinMaxScaler()
names = df.columns
d = scaler.fit_transform(df)
df = pd.DataFrame(d, columns=names)
df.head()


# In[50]:


test, train = df[df["ind"].eq(0)], df[df["ind"].eq(1)]
test = test.drop('ind', axis=1)
train = train.drop('ind', axis=1)
train = train.drop('ID', axis=1)
test = test.drop('ID', axis=1)
test = test.drop('Disease', axis=1)
test.head()


# In[51]:


from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(X)
pd.DataFrame(X).applymap(lambda x: abs(x))


# In[52]:


train.isna().sum().sum()
train.duplicated().sum()


# In[53]:


X = train.iloc[:, :-1].select_dtypes('number')
# X = StandardScaler().fit_transform(X)
y = train.iloc[:, -1]
X.shape
y.shape
type(y)


# # BNB bernoulli Perdiction

# In[54]:


bnb = BernoulliNB()
bnb_pred = bnb.fit(X, y).predict(X)
metrics.accuracy_score(y, bnb_pred)


# In[55]:


bnb_pred = model.predict(test)
BNBsubmission = pd.DataFrame({
        "Sick": bnb_pred
    })
BNBsubmission.to_csv('BNBsubmission.csv', index=False)


# In[56]:



X = train.drop('Disease', axis=1)
y = train['Disease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=16)
classifier = KNeighborsClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(f"Accuracy: {round(metrics.accuracy_score(y_test, y_pred)*100, 2)}%")
df_confusion = pd.crosstab(y_test, y_pred)


# In[57]:


print(y_pred)
y_pred = model.predict(test)
KNNsubmission = pd.DataFrame({
        "Sick": y_pred
    })
KNNsubmission.to_csv('KNNsubmission.csv', index=False)


# # KNN with Pipeline

# In[58]:


## Use Pipeline to Streamline the Analysis

knn_pipe = make_pipeline(StandardScaler(), KNeighborsClassifier())
knn_pipe.fit(X_train, y_train)
', '.join(dir(knn_pipe))
knn_pipe.get_params()
knn_pipe.set_params(kneighborsclassifier__n_neighbors=8)
pipe_pred = knn_pipe.predict(X_test)
pd.Series(pipe_pred).value_counts()


# In[59]:


pipe_pred = knn_pipe.predict(test)
EMsubmission = pd.DataFrame({
        "Sick": pipe_pred
    })
EMsubmission.to_csv('KNNPsubmission.csv', index=False)
print(pipe_pred)


# In[60]:


## Get Repeated Hold Out Accurary of Model

cv = ShuffleSplit(n_splits=100, test_size=0.3, random_state=16)
from sklearn.model_selection import KFold
cv = KFold(n_splits=10, shuffle=True, random_state=16)
cross_val_score(knn_pipe, X_train, y_train, cv=cv).mean()


# # Ensemble Learning - Bagging

# In[61]:


## Ensemble Learning - Bagging

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

cart = DecisionTreeClassifier()
num_trees = 100
modelb = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=16)
results = cross_val_score(modelb, X_train, y_train, cv=cv)
print(f"Accuracy: {round(results.mean()*100, 2)}%")


# In[64]:


modelb.fit(X_train, y_train)
y_predictem = model.predict(test)
EMsubmission = pd.DataFrame({
        "Sick": y_predictem
    })
EMsubmission.to_csv('EMsubmission.csv', index=False)


# # Random Forest

# In[65]:


## Random Forest

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=num_trees, max_features=5, random_state=16)
results = cross_val_score(model, X_train, y_train, cv=cv)
print(f"Accuracy: {round(results.mean()*100, 2)}%")
model


# In[66]:


model.fit(X_train, y_train)
y_predict = model.predict(test)
RFCsubmission = pd.DataFrame({
        "Sick": y_predict
    })
RFCsubmission.to_csv('RFCsubmission.csv', index=False)


# Out of all the different approaches, random forest had the highest accuracy.
