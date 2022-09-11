#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[5]:


pip install plotly


# In[6]:


import plotly.express as px


# In[7]:


import plotly.graph_objects as go


# In[8]:


data = pd.read_csv("Downloads/waiter_tips_data.csv")


# In[9]:


print(data.head())


# In[10]:


figure = px.scatter(data_frame = data, x="total_bill", y= "tip", size = "size" , color = "day", trendline = "ols" )


# In[11]:


figure.show()


# In[12]:


figure= px.scatter(data_frame = data, x="total_bill",y="tip", size="size", color="sex", trendline="ols")


# In[13]:


figure.show()


# In[14]:


figure= px.scatter(data_frame = data, x="total_bill",y="tip", size="size", color="time", trendline="ols")


# In[15]:


figure.show()


# In[22]:


figure= px.pie(data, values ='tip',names='day',hole= 0.4)


# In[23]:


figure.show()


# In[24]:


figure=px.pie(data, values='tip', names='sex', hole=0.4)


# In[26]:


figure.show()


# In[27]:


figure=px.pie(data, values='tip',names='smoker', hole = 0.4)


# In[28]:


figure.show()


# In[29]:


figure=px.pie(data, values='tip',names='time', hole = 0.4)


# In[30]:


figure.show()

Before training a waiter tips prediction model, I will do some data transformation by transforming the categorical values into numerical values
# In[31]:


data["sex"] = data["sex"].map({"Female": 0, "Male": 1})
data["smoker"]= data["smoker"].map({"No": 0, "Yes": 1})
data["day"]= data["day"].map({"Thur": 0,"Fri": 1,"Sat": 2,"Sun": 3})
data["time"]= data["time"].map({"Lunch": 0,"Dinner": 1})
data.head()


# In[32]:


x= np.array(data[["total_bill","sex","smoker","day","time","size"]])


# In[33]:


y= np.array(data["tip"])


# In[34]:


from sklearn.model_selection import train_test_split


# In[35]:


xtrain, xtest, ytrain, ytest= train_test_split(x, y,test_size=0.2 ,random_state=42)


# In[36]:


from sklearn.linear_model import LinearRegression


# In[37]:


model = LinearRegression()


# In[38]:


model.fit(xtrain,ytrain)


# In[39]:


# features = [["total_bill","sex","smoker","day","time","size"]]
features = np.array([[24.50,1,0,0,1,4]])


# In[40]:


model.predict(features)


# In[ ]:




