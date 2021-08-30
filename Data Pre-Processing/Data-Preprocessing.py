#!/usr/bin/env python
# coding: utf-8

# # Import Libraries and Data
# 

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
employees = pd.read_csv('employee_attrition.csv')


# # Reorganize 

# Reorganize my columns so all numerical values are first and all x inputs are first. Now the dependent Y variable is in the last column

# In[2]:



columns = ['Age','DailyRate','DistanceFromHome','Education','EmployeeCount', 'EmployeeNumber', 'EnvironmentSatisfaction',
 'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel', 'TotalWorkingYears',
 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager',
 'Over18', 'OverTime', 'MaritalStatus', 'JobRole', 'Gender', 'EducationField', 'Department', 'BusinessTravel', 'Attrition']


# In[3]:


employees = employees.reindex(columns= columns)


# # Fill in Missing Data

# Checking data for duplicates 

# In[4]:


from scipy.stats.mstats import winsorize
print(f"There are {employees.isna().sum().sum()} missing values.")
print(f"There are {employees.duplicated().sum()} duplicated records.")


# Clean the data by filling out missing intereger values with the average of the column. Clean up interger values. 

# In[5]:



from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy = 'mean')
employees.DistanceFromHome = imputer.fit_transform(employees['DistanceFromHome'].values.reshape(-1,1))[:,0]
employees.JobLevel = imputer.fit_transform(employees['JobLevel'].values.reshape(-1,1))[:,0]
employees.PercentSalaryHike = imputer.fit_transform(employees['PercentSalaryHike'].values.reshape(-1,1))[:,0]
employees.PerformanceRating = imputer.fit_transform(employees['PerformanceRating'].values.reshape(-1,1))[:,0]
employees.RelationshipSatisfaction = imputer.fit_transform(employees['RelationshipSatisfaction'].values.reshape(-1,1))[:,0]
employees.TotalWorkingYears = imputer.fit_transform(employees['TotalWorkingYears'].values.reshape(-1,1))[:,0]
employees.YearsSinceLastPromotion = imputer.fit_transform(employees['YearsSinceLastPromotion'].values.reshape(-1,1))[:,0]


# In[6]:


from scipy.stats.mstats import winsorize
print(f"There are {employees.isna().sum().sum()} missing values.")
print(f"There are {employees.duplicated().sum()} duplicated records.")


# # Discretization

#  Making Salaries into discrete steps to make the data easier to process. 

# In[8]:



employees['DailyRate'] = pd.cut(employees['DailyRate'], bins=3,labels=['low', 'Medium','high'] )

employees['MonthlyIncome'] = pd.cut(employees['MonthlyIncome'], bins=3,labels=['low', 'Medium','high'] ) # equal interval discretization

employees['MonthlyRate'] = pd.cut(employees['MonthlyRate'], bins=3,labels=['low', 'Medium','high'] ) # equal interval discretization

employees['PercentSalaryHike'] = pd.cut(employees['PercentSalaryHike'], bins=3,labels=['low', 'Medium','high'] ) # equal interval discretization


# # Convert Catagory Attributes

# In[9]:


obj_col = employees.select_dtypes('object').columns
for col in obj_col:
    print(f"{col} has {employees[col].nunique()} unique values")


# In[10]:


pd.concat([pd.get_dummies(employees['MaritalStatus']), employees['MaritalStatus']], axis=1)


# In[11]:


pd.concat([pd.get_dummies(employees['JobRole']), employees['JobRole']], axis=1)


# In[12]:


pd.concat([pd.get_dummies(employees['JobRole']), employees['JobRole']], axis=1)


# In[13]:


import scipy
employees['ages'] = (employees['Age'] - employees['Age'].mean())/employees['Age'].std()
employees['ages'].describe()


# # EDA

# What is the difference between rates and income? 
# 
# What are some of the bigest factors to people leaving? 
# 
# How long do people stay and at what age? 
# 
# What are some of the values that appear the same for every employee?

# In[35]:


# Load in python visualization libraries

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[36]:


# Scatter Plot
sns.scatterplot(x="Age", y="YearsAtCompany", data=employees)


# A majority of the people only stay for 10 years. 

# In[37]:


employees['MonthlyRate'].hist()


# In[38]:


employees['MonthlyIncome'].hist()


# Looking at monthly rate it is confusing. Compared to monthly income they are opposites, what is the monthly rate related too? They also have the hourly rate. These 3 variables are very different in amount but I assume they come down to the same thing, pay.

# # Varibles that are the same

# In[18]:


employees['StandardHours'].hist()


# In[30]:


employees['EmployeeCount'].hist()


# In[31]:


employees['Over18'].hist()


# The entire data set has some values that are not useful for analysis. What is employee count? Everyone has 1.Is it a requirement to be over 18? Hence why there is no one under 18. Is the standard hours for everyone 80 hours? 

# # Looking at Attrition as a whole

# In[39]:


employees['Attrition'].hist()


# A large majority of the people stayed at the company according to the data provided.

# # Small sub to get insight to who leaves

# In[40]:


# permutation of the data frame
employees.sample(frac=.6).head(5)


# # Check Attrition vs other varibles

# In[41]:


dis_by_Attr = employees.groupby('Attrition')['DistanceFromHome'].mean().reset_index()
sns.catplot(x='DistanceFromHome', y='Attrition', data=dis_by_Attr , kind='bar')


# In[27]:


ed_by_attr = employees.groupby('Attrition')['Education'].mean().reset_index()
sns.catplot(x='Education', y='Attrition', data=ed_by_attr , kind='bar')


# In[28]:


hrate_by_attr = employees.groupby('Attrition')['HourlyRate'].mean().reset_index()
sns.catplot(x='HourlyRate', y='Attrition', data=hrate_by_attr , kind='bar')


# In[29]:


yrs_by_attr = employees.groupby('Attrition')['YearsAtCompany'].mean().reset_index()
sns.catplot(x='YearsAtCompany', y='Attrition', data=yrs_by_attr , kind='bar')


# Looking at a few varibles it looks like distance to office has had the biggest impact on employee Attrition. Is this something that potential change after being employeed? 
# 
# Time is also a factor, the longer they are at the company the more likely they are to stay. 

# In[ ]:




