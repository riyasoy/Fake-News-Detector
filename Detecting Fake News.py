#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# In[2]:


#Read the data
df=pd.read_csv('D:\\RIYA\\news.csv')
#Get shape and head
df.shape
df.head()


# In[3]:


labels=df.label
labels.head()


# In[5]:


#DataFlair - Split the dataset
x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)


# In[8]:


tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df= 0.7)

tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test  =tfidf_vectorizer.transform(x_test)


# In[11]:


pac= PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test, y_pred)
print(f'Accuracy: {round(score*100,2)}%')


# In[13]:


#Building the confusion matrix
confusion_matrix( y_test,y_pred, labels=['FAKE','REAL'])


# In[ ]:




