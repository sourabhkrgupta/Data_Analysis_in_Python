#!/usr/bin/env python
# coding: utf-8

# ## Analyzing a dataset - Titanic - Machine Learning from Disaster

# In[1]:


import numpy as np
import pandas as pd


# ##### Load the dataset of Titanic(Dataset Link: https://www.kaggle.com/c/titanic )

# In[4]:


data= pd.read_csv("train.csv")


# ##### Display the top 5 rows of dataset

# In[7]:


data.head(5)


# In[8]:


data.shape         ####Find the shape of dataset


# ##### To Get Information About Our Dataset Like Total Number Rows, Total Number of Columns,Datatypes of Each Column And Memory Requirement

# In[11]:


data.info()


# In[12]:


##### Here we find some missing values are available on Age, Cabin and Embarked columns


# ##### Get Overall Statistics About The Dataframe

# In[16]:


data.describe()


# In[14]:


##### Shows statistics of all numerical value columns.


# #### Data Filtering

# In[17]:


data.columns


# In[22]:


sum(data['Sex']=='male')                      #### Find the total number of male persons


# In[23]:


data[data['Sex']=='male']                    #### Information of all male travellor


# In[24]:


data[data['Survived']==1]                           #####Information about survived persons


# #### Checking the null values in dataset

# In[25]:


data.isnull().sum()


# In[26]:


##### Age column have 177 null values, Cabin have 687 and Embarked have 2 null values.


# In[28]:


data.isnull().sum()*100/len(data)                  #### Finding missing values in percent


# In[29]:


##### Cabin have 77% empty data, so we can neglect this column. Embarked have 0.2% empty data so it is acceptable in dataset
##### whereas age column is also acceptable at 20% empty data and later we handle this missing data.


# #### Drop the Cabin column

# In[30]:


data.drop('Cabin', axis=1,inplace=True)


# In[31]:


data.isnull().sum()


# #### Handle missing values

# In[32]:


data['Embarked'].mode()                             #### Find the mode of column


# In[35]:


data['Embarked'].fillna('S',inplace=True)            #### Fill the null values with 'S'.


# In[34]:


data.isnull().sum()                                  #### Recheck the datsets


# In[37]:


data['Age'].fillna(data['Age'].mean(),inplace=True)             #### Fill the null values with the mean values of age.


# In[38]:


data.isnull().sum()                    


# In[39]:


#### All missing values are handled using statistics.


# #### Categorical data encoding

# In[40]:


data.Sex.unique()


# In[41]:


data['Gender']= data['Sex'].map({'male':1,'female':0})


# In[42]:


data.head(2)


# In[43]:


data.Embarked.unique()


# In[46]:


data1=pd.get_dummies(data,columns=['Embarked'])


# In[47]:


data1.head(5)


# #### Let's find how many survived and how many died?

# In[48]:


len(data[data['Survived']==1])


# In[49]:


len(data[data['Survived']==0])


# In[50]:


#### 342 persons are survived and 549 persons are died.


# #### How Many Passengers Were In First Class, Second Class, and Third Class?

# In[51]:


data['Pclass'].value_counts()


# In[52]:


#### So, 491 persons on Third class, 184 persons on Second class and 216 persons on First class.


# #### Number of Male And Female Passengers

# In[53]:


data['Sex'].value_counts()


# In[54]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[56]:


sns.countplot(data['Sex'])


# In[57]:


plt.hist(data['Age'])


# In[58]:


sns.boxplot(data['Age'])


# In[59]:


#### Hence most of the passengers are between 20 to 40 age group.


# #### How Has Better Chance of Survival Male or Female?

# In[60]:


sns.barplot(x='Sex',y='Survived',data=data)


# In[61]:


#### Female have more chances of survival as compared to male.


# #### Which Passenger Class Has Better Chance of Survival (First, Second, Or Third Class)? 

# In[63]:


sns.barplot(x='Pclass', y='Survived', data=data)


# In[64]:


#### As per the bar chart, First class passengers have better chance of Survival as compared to Second and Third class. 


# In[65]:


data.columns


# In[66]:


data['Family_size']=data['SibSp']+data['Parch']


# In[67]:


data.head(4)


# #### Find fares per persons

# In[68]:


data['Fares_per_person']= data['Fare']/(data['Family_size']+1)


# In[69]:


data.head(3)


# In[ ]:




