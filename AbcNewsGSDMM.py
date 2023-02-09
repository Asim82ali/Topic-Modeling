#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from pprint import pprint
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gsdmm import MovieGroupProcess 
import pandas as pd;
import numpy as np;
import sys;
from nltk.corpus import stopwords;
import nltk;
from gensim.models import ldamodel
import gensim.corpora;


# In[2]:


data = pd.read_csv('abcnews-date-text.csv', error_bad_lines=False)


# In[3]:


print(data.head(5))
len(data)


# In[4]:


data_text = data[['headline_text']];


# In[5]:


print(data_text.head(5))


# In[6]:


np.random.seed(1024);
data_text = data_text.iloc[np.random.choice(len(data_text), 10000)]


# In[7]:


len(data_text)


# In[8]:


data_text.head(10)


# In[9]:


data_text = data_text.astype('str')


# In[10]:


from nltk.corpus import stopwords;

import nltk;


# In[11]:


for idx in range(len(data_text)):
    
    
    #go through each word in each data_text row, remove stopwords, and set them on the index.
    data_text.iloc[idx]['headline_text'] = [word for word in data_text.iloc[idx]['headline_text'].split(' ') if word not in stopwords.words()];
    
    if idx % 1000 == 0:
        sys.stdout.write('\rc = ' + str(idx) + ' / ' + str(len(data_text)));


# In[12]:


print(data_text.head(10))


# In[13]:


train_headlines = [value[0] for value in data_text.iloc[0:].values]


# In[14]:


train_headlines[0]


# In[15]:


id2word = gensim.corpora.Dictionary(train_headlines)


# In[16]:


corpus = [id2word.doc2bow(text) for text in train_headlines]


# In[17]:


# Train STTM model
mgp = MovieGroupProcess(K=10, alpha=0.1, beta=0.1, n_iters=20)
vocab = set(x for doc in train_headlines for x in doc)
n_terms = len(vocab)
y = mgp.fit(train_headlines, n_terms)


# In[18]:


doc_count = np.array(mgp.cluster_doc_count)
print('Number of documents per topic :', doc_count)
print('*'*50)


# In[19]:


top_index = doc_count.argsort()[-20:][::-1]
print('\nMost important clusters (by number of docs inside):', top_index)


# In[20]:


def top_words(cluster_word_distribution, top_cluster, values):
    for cluster in top_cluster:
        sort_dicts =sorted(mgp.cluster_word_distribution[cluster].items(), key=lambda k: k[1], reverse=True)[:values]
        print("\nCluster %s : %s"%(cluster,sort_dicts))


# In[21]:


top_words(mgp.cluster_word_distribution, top_index, 20) #print top 15 words for each cluster


# In[22]:


def get_topics_lists(model, top_clusters, n_words):
    '''
    Gets lists of words in topics as a list of lists.
    
    model: gsdmm instance
    top_clusters:  numpy array containing indices of top_clusters
    n_words: top n number of words to include
    
    '''
    # create empty list to contain topics
    topics = []
    
    # iterate over top n clusters
    for cluster in top_clusters:
        #create sorted dictionary of word distributions
        sorted_dict = sorted(model.cluster_word_distribution[cluster].items(), key=lambda k: k[1], reverse=True)[:n_words]
         
        #create empty list to contain words
        topic = []
        
        #iterate over top n words in topic
        for k,v in sorted_dict:
            #append words to topic list
            topic.append(k)
            
        #append topics to topics list    
        topics.append(topic)
    
    return topics

# get topics to feed to coherence model
topics = get_topics_lists(mgp, top_index, 20) 


# In[23]:


cm_gsdmm = CoherenceModel(topics=topics, 
                          dictionary=id2word, 
                          corpus=corpus, 
                          texts=train_headlines, 
                          coherence='c_v')

# get coherence value
coherence_gsdmm = cm_gsdmm.get_coherence()  
print(coherence_gsdmm)


# In[ ]:




