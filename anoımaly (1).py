#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
df = pd.read_csv('anomaly.csv')

print(df.groupby('Quality')['Quality'].count())

df.drop(['Date'], axis=1, inplace=True)

df.dropna(inplace=True,axis=1)


# In[16]:


df.Quality[df.Quality == 'Good'] = 1
df.Quality[df.Quality == 'Bad'] = 2


good_mask = df['Quality']== 1 
bad_mask = df['Quality']== 2 


df.drop('Quality',axis=1,inplace=True)

df_good = df[good_mask]
df_bad = df[bad_mask]

print(f"Good count: {len(df_good)}")
print(f"Bad count: {len(df_bad)}")


# In[17]:


#neural net
x_good = df_good.values
x_bad = df_bad.values

from sklearn.model_selection import train_test_split

x_good_train, x_good_test = train_test_split(
        x_good, test_size=0.25, random_state=42)

print(f"Good train count: {len(x_good_train)}")
print(f"Good test count: {len(x_good_test)}")


# In[18]:


from sklearn import metrics
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[19]:


model = Sequential()
model.add(Dense(10, input_dim=x_good.shape[1], activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(x_good.shape[1])) 
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()

model.fit(x_good_train,x_good_train,verbose=1,epochs=100)


# In[20]:


pred = model.predict(x_good_test)
score1 = np.sqrt(metrics.mean_squared_error(pred,x_good_test))

pred = model.predict(x_good)
score2 = np.sqrt(metrics.mean_squared_error(pred,x_good))

pred = model.predict(x_bad)
score3 = np.sqrt(metrics.mean_squared_error(pred,x_bad))


# In[21]:


print(f"Insample Good Score (RMSE): {score1}".format(score1))
print(f"Out of Sample Good Score (RMSE): {score2}")
print(f"Bad sample Score (RMSE): {score3}")


# In[ ]:





# In[ ]:




