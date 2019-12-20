#!/usr/bin/env python
# coding: utf-8

# ### Segmenting and Clustering Neighborhoods in Toronto. Part 1.
# 

# In[58]:


import requests
import lxml.html as lh
import pandas as pd


# #####  Load website

# In[59]:


# Create a handle, page, to handle the contents of the website
url='https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M'
page = requests.get(url)#Store the contents of the website under doc
doc = lh.fromstring(page.content)#Parse data that are stored between <tr>..</tr> of HTML
tr_elements = doc.xpath('//tr')#Create empty list
col=[]
i=0#For each row, store each first element (header) and an empty list
for t in tr_elements[0]:
    i+=1
    name=t.text_content()
    #print '%d:"%s"'%(i,name)
    col.append((name,[]))
for j in range(1,len(tr_elements)):
    T=tr_elements[j]
    if len(T)!=3:
        break
    i=0
    for t in T.iterchildren():
        data=t.text_content() 
        if i>0:
            try:
                data=int(data)
            except:
                pass
        col[i][1].append(data)
        i+=1


# In[60]:


# check if the data are loaded
[len(C) for (title,C) in col]


# ##### create  data frame

# In[61]:


#create a data frame and delete 'not assigned' from the dataframe
Dict={title:column for (title,column) in col}
df=pd.DataFrame(Dict)
df.astype(str)
indexNames = df[ df['Borough'] == 'Not assigned' ].index
df.drop(indexNames , inplace=True)
df.head(3)


# In[62]:


# remove \n from the column name and rows
df = df.replace('\n','', regex=True)
df.rename(columns={"Neighbourhood\n": "Neighbourhood"}, inplace=True)
df.head(3)


# In[63]:


# combine multiple Neighbourhood rows into one. THEN regroup by 'Postcode' and 'Borough'
df=df.groupby(['Postcode','Borough'])['Neighbourhood'].apply(', '.join).reset_index()
df.head(5)


# In[64]:


# find the 'Not assigned' entries from the 'Neighbourhood'
indexNames = df[ df['Neighbourhood'] == 'Not assigned']
print(indexNames)


# In[69]:


# replace the 'Not assigned' entries based on the corresponding 'Borough's entries
indexNames = df[ df['Neighbourhood'] == 'Not assigned'].index
df.loc[indexNames,'Neighbourhood'] = df1.loc[indexNames,'Borough']
indexNames = df[ df['Neighbourhood'] == 'Not assigned']
print(indexNames)


# In[66]:


# check the Neighbourhood column
indexNames = df[ df['Neighbourhood'] == 'Queen\'s Park']
print(indexNames)


# ##### Final shape

# In[67]:


# shape the data set
df.shape

