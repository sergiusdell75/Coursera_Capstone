#!/usr/bin/env python
# coding: utf-8

# ### Segmenting and Clustering Neighborhoods in Toronto. Part 2.

# In[1]:


from geopy.geocoders import Nominatim # convert an address into latitude and longitude values
import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe
import pandas as pd
import numpy as np
from time import sleep


# ##### read csv as data frame

# In[2]:


df=pd.read_csv('Postcode.csv')
df.head(5)


# In[28]:


postcodes=df['Postcode'].to_list()
print(postcodes)


# ##### create latitude/lognitudes

# In[5]:


import pgeocode
nomi = pgeocode.Nominatim('ca')
nomi.query_postal_code("M1B")


# In[18]:


latitudes=[]
longitudes=[]
for codes in postcodes:
    tdf=nomi.query_postal_code(codes)
    lat=tdf[9]
    longi=tdf[10]
    print(codes, lat, longi)
    latitudes.append(tdf[9])
    longitudes.append(tdf[10])
    sleep(1)


# In[30]:


df['Latitude']=latitudes
df['longitude']=longitudes
df.head(3)


# In[32]:


# fill the nan entry
a=43.651890
b=-79.381710
indexNames = df[ df['Postcode'] == 'M7R'].index
print(indexNames)
df.loc[indexNames,'Latitude'] =a 
df.loc[indexNames,'longitude']=b
indexNames = df[ df['Postcode'] == 'M7R']
print(indexNames)


# In[35]:


df.to_csv(r'Coordinates.csv',index=False)

