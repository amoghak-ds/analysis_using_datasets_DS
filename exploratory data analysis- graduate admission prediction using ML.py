#!/usr/bin/env python
# coding: utf-8

# In[2]:


#import the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


data=pd.read_csv("C:\\Users\\amogh\\Downloads\\archive (2)\\Admission_Predict.csv")
df=pd.DataFrame(data)
print(df)


# In[14]:


df.columns


# In[15]:


df.head()


# In[16]:


df.describe()


# In[17]:


df.dtypes


# In[18]:


df.isnull()


# In[21]:


df.isnull().sum()


# In[53]:


df.head()


# In[60]:


# IDENTIFYING AND REMOVING OUTLIERS (interquartile range method)
df.boxplot(column=["GRE Score","TOEFL Score"])  


# In[58]:


df.boxplot(column=["University Rating","SOP","CGPA","Research"])


# In[65]:


# Calculate quartiles for each feature
Q1=df.quantile(0.25)
Q3=df.quantile(0.75)
IQR= Q3-Q1
print(IQR)
    


# In[67]:


# Identifying outliers 
df_out1= df[((df<(Q1-1.5 *IQR)) | (df> (Q3+1.5*IQR))).any(axis=1)]
df_out1.head()


# In[68]:


# Removing the outliers
df_out= df[~((df <(Q1-1.5*IQR)) |(df>(Q3+1.5*IQR))).any(axis=1)]
df=df_out.copy()
print(df.shape)          # outliers are removed from the data


# In[71]:


# UNIVARIATE ANALYSIS ( Finding the patterns of the data)


## target variable
df["University Rating"].plot.hist()
plt.xlabel('Rating', fontsize=12)

# maximum students are getting between 3-3.5 rating.


# In[72]:


df['Research'].value_counts()

## 218 students have research experience and 178 students don't have any research experience


# In[76]:


# BIVARIATE ANALYSIS

df.plot.scatter('GRE Score','CGPA')


# In[ ]:




