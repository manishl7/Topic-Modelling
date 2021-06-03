#!/usr/bin/env python
# coding: utf-8

# In[1]:


# npr.csv ;We'll use this data file to load up some article

import pandas as pd


# In[2]:


npr=pd.read_csv('C:/Users/me626/Desktop/NLP/UPDATED_NLP_COURSE/UPDATED_NLP_COURSE/05-Topic-Modeling/npr.csv')


# In[3]:


npr.head()


# In[ ]:


#each rows contains full text of one of the article
#We donot have any labels


# In[12]:


len(npr)


# In[14]:


#Before we run LDA , We need to do a bit of preprocessing


# In[4]:


from sklearn.feature_extraction.text import CountVectorizer


# In[5]:


cv=CountVectorizer(max_df=0.9,min_df=2,stop_words='english')


# In[6]:


dtm=cv.fit_transform(npr['Article'])


# In[7]:


dtm


# In[9]:


from sklearn.decomposition import LatentDirichletAllocation


# In[10]:


LDA=LatentDirichletAllocation(n_components=7,random_state=42)


# In[15]:


LDA.fit(dtm)  


# In[16]:


len(cv.get_feature_names())


# In[27]:


type(cv.get_feature_names())


# In[12]:


# to get some words off of cv
cv.get_feature_names()[50000] #grabbing word from 50k index


# In[13]:


#to print bunch of random words from this list
import random
random_word_id=random.randint(0,54777) #calling random , random and range
cv.get_feature_names()[random_word_id]


# In[30]:


#Step 2 : Grab the topic


# In[17]:


# we'll grab info off of the trained LDA
len(LDA.components_)


# In[32]:


type(LDA.components_) #It's a numpy array containing probabilites for each words


# In[34]:


LDA.components_.shape


# In[18]:


single_topic=LDA.components_[0] #checking topic for first index


# In[19]:


single_topic.argsort()
#argsort()returns the index position thatwill sort this array


# In[20]:


#ets create a simply array to understand the arg sort process
import numpy as np


# In[21]:


single_topic.argsort()[-10:]


# In[22]:


top_ten_words=single_topic.argsort()[-10:]


# In[23]:


for index in top_ten_words:
    print(cv.get_feature_names()[index])


# In[24]:


#lets set a loop that prints out top 15 words from each of the seven topic
for i,topic in enumerate(LDA.components_):
   
    print(f"The TOP 15 WORDS FOR TOPIC # {i}")
    print([cv.get_feature_names()[index] for index in topic.argsort()[-15:]])
    print('\n')
    print('\n')


# In[25]:


topic_results=LDA.transform(dtm)


# In[26]:


topic_results 
print(topic_results[0])
topic_results[0].round(2)


# In[27]:


# npr['Article'][0]


# In[28]:


topic_results[0].argmax()


# In[29]:


npr['Topic']=topic_results.argmax(axis=1)


# In[30]:


npr.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[35]:





# In[ ]:




