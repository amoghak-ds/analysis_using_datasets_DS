#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the required libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


# In[5]:


data_frame = pd.read_csv("C:\\Users\\amogh\\Downloads\\archive (4)\\Mall_Customers.csv")
data_frame.head()


# In[6]:


data_frame.shape


# In[7]:


data_frame.isnull().sum()


# In[8]:


data_frame.info()


# In[9]:


data_frame.describe()


# In[11]:


data_frame.columns


# In[13]:


plt.style.use('fivethirtyeight')
fig = plt.figure(figsize = (20, 10))
plt.subplot(1, 2, 1)
sns.set(style = 'whitegrid')
sns.histplot(my_frame['Annual Income (k$)'])
plt.title('Distribution of Annual Income', fontsize = 22)
plt.xlabel('Range of Annual Income')
plt.ylabel('Count of Annual Income')

plt.show()


# In[15]:


plt.style.use('fivethirtyeight')
fig = plt.figure(figsize = (20, 10))
plt.subplot(1, 2, 2)
sns.set(style = 'whitegrid')
sns.histplot(my_frame['Age'], color = 'green')
plt.title('Distribution of Age', fontsize = 22)
plt.xlabel('Range of Age')
plt.ylabel('Count of Age')

plt.show()


# In[17]:


plt.style.use('fivethirtyeight')

genders = ['Male', 'Female']
size = my_frame['Gender'].value_counts()
my_colors = ['red', 'blue']
exp = [0, 0.1]

fig = plt.figure(figsize = (20, 10))
plt.pie(size, labels = genders, colors = my_colors, explode = exp, shadow = True, autopct = '%.2f%%')
plt.title('Gender', fontsize = 22)
plt.legend()

plt.show()


# In[20]:


plt.style.use('fivethirtyeight')
fig = plt.figure(figsize = (20, 10))

sns.pairplot(my_frame)
plt.title('Pairplot of dataset', fontsize = 22)
plt.show()


# In[33]:


x = my_frame.iloc[:, [3, 4]].values
# check the shape of x
print(x.shape)
x


# In[35]:


# Clusters using K-means Algorithm
from sklearn.cluster import KMeans


# In[37]:


# Elbow Curve
fig = plt.figure(figsize = (20, 10))

wcss = [] #within cluster sum of squares
for i in range(1, 11):
    km = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    km.fit(x)
    wcss.append(km.inertia_)
    
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Curve', fontsize = 20)
plt.xlabel('No. of Clusters')
plt.ylabel('wcss')
plt.show()


# In[38]:


fig = plt.figure(figsize = (20, 10))

km = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_means = km.fit_predict(x)

plt.scatter(x[y_means == 0, 0], x[y_means == 0, 1], s = 100, c = 'red', label = 'miser')
plt.scatter(x[y_means == 1, 0], x[y_means == 1, 1], s = 100, c = 'blue', label = 'general')
plt.scatter(x[y_means == 2, 0], x[y_means == 2, 1], s = 100, c = 'magenta', label = 'target')
# plt.scatter(x[y_means == 3, 0], x[y_means == 3, 1], s = 100, c = 'pink', label = 'spendthrift')
# plt.scatter(x[y_means == 4, 0], x[y_means == 4, 1], s = 100, c = 'yellow', label = 'careful')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 50, c = 'green' , label = 'centeroid')

plt.style.use('fivethirtyeight')
plt.title('K Means Clustering', fontsize = 20)
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.grid()
plt.show()


# In[ ]:




