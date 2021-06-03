#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
npr=pd.read_csv('C:/Users/me626/Desktop/NLP/UPDATED_NLP_COURSE/UPDATED_NLP_COURSE/05-Topic-Modeling/npr.csv')


# In[2]:


npr.head()


# In[4]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[5]:


tfidf=TfidfVectorizer(max_df=0.95,min_df=2,stop_words='english')


# In[6]:


dtm=tfidf.fit_transform(npr['Article'])


# In[7]:


dtm
#it's a sparse matrix like before


# In[8]:


#Lets perfom NMF
from sklearn.decomposition import NMF


# In[9]:


nmf_model=NMF(n_components=7,random_state=42)
#here random_state will initialize h and w matrix


# In[10]:


nmf_model.fit(dtm)


# In[11]:


#something we should notice is NMF perfroms faster than LDA 
#specially because of the way numpy works
#it's really well suited for matrix factorization problems


# In[12]:


# feature names
tfidf.get_feature_names()[200]


# In[14]:


#for top 15
for index ,topic in enumerate(nmf_model.components_):
    print(f"#Topic{index}")
    print([tfidf.get_feature_names()[i]for i in topic.argsort()[-15:]])
    print('\n')
    print('\n')
#previously with lda we were dealing with words that had the highest probabiltiy of falling into the topic
#with NMF , we are dealing with words with highest coeffiecent values inside of that matrix


# In[18]:


topic_results=nmf_model.transform(dtm)


# In[19]:


topic_results[0]


# In[20]:


topic_results[0].argmax()


# In[21]:


#for entire array
npr['Topic']=topic_results.argmax(axis=1)


# In[22]:


npr.head()


# In[26]:


#let's create a dictionary for the topics
# mytopic_dict={1:"Politics",2:'someother topic'}
mytopic_dict={0:'health',1:'election',2:'Legislation',3:'politics',4:'election',5:'music',6:'edu'}


# In[27]:


npr['Topic_label']=npr['Topic'].map(mytopic_dict)


# In[28]:


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




