#!/usr/bin/env python
# coding: utf-8

# # POKEMON DATASET ANALYSIS

# The dataset I chose for this analysis is the Pokemon dataset whose features are number, name, type1, type2, total, hp, attack, defense, sp_attack, sp_defense, speed, generation, and legendary. The first type of analysis is simple bar charts for the generation and attack variables, and then a more complex bar chart that includes the type breakdowns for each generation. I then plotted the correlation between each of the pokemon's stats. The final piece of analysis I performed was checking the distribution of each of the pokemon's stats, plotting it, and then transforming the data so that each of the stats were normalized.

# In this dataset I also employed a k-nearest neighbors alogorithm in an attempt to predict the primary type of each pokemon.

# ### Imports

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv(r"C:\Users\Owner\Downloads\DSC 360\pokemon.csv")
df.head()


# ### Baseline Analysis

# In[2]:


count_number_gen=df["generation"].value_counts()
sns.barplot(x=count_number_gen.index, y=count_number_gen.values)


# The above bar chart details how many pokemon first appeared in each of the generations, with generation 1 having the most and generation 6 having the least.

# In[3]:


plt.hist(df['attack'], color='grey', edgecolor='black', bins=int(180/5))
sns.distplot(df['attack'], hist=True,kde=False, bins=int(180/5),color='blue',hist_kws={'edgecolor':'black'})


# Above is a distribution plot of the attack variable for each pokemon, and you can see that most pokemon have an attack statistic between 50 and 100.

# In[4]:


df.groupby(["generation","type1"]).size().unstack().plot(kind='bar', stacked=True, figsize=(20,10))


# This bar chart shows the bar chart of each pokemon per generation in the same way as before, except now it is broken down by the type of each pokemon. Based on this chart you can see that water and normal types are the most common across each generation.

# In[5]:


df=df[['hp','attack','defense','sp_attack','sp_defense','speed']]
corr_df=df.corr()
sns.heatmap(corr_df, xticklabels=corr_df.columns.values, yticklabels=corr_df.columns.values,annot=True, annot_kws={'size':12})
heat_map=plt.gcf(); heat_map.set_size_inches(10,5)
plt.xticks(fontsize=10); plt.yticks(fontsize=10); plt.show()


# Based on the above heatmap, there is a:
# -  high correlation between pokemon with a high special defense stat and those with high defense and special attack stats
# -  low correlation between those with high speed stats and high defense stats

# ### Normalization

# In[6]:


import numpy as np
import numpy as np;import pandas as pd;import seaborn as sns;import time
import re;import os;import matplotlib.pyplot as plt
sns.set(style="ticks")

import sklearn as sk
from scipy import stats
from sklearn import preprocessing
df_array=np.array(df)
loop_c=-1
col_for_normalization=list()
for column in df_array.T:
    loop_c+=1
    x=column
    k2,p=stats.normaltest(x)
    alpha=0.001
    print("p={:g}".format(p))
    if p<alpha:
        test_result="non_normal_distr"
        col_for_normalization.append((loop_c))
        print("The null hypothesis can be rejeccted: non-normal distribution")
    else:
        test_result="normal_distr"
        print("The null hypothesis cannot be rejected: normal distribution")


# Based on the above analysis, none of the pokemon statistics follow a normal distribution. This can be seen in the density plot shown below.

# In[7]:


columns_to_normalize=df[df.columns[col_for_normalization]]
names_col=list(columns_to_normalize)
columns_to_normalize.plot.kde(bw_method=3)


# In[8]:


pt=preprocessing.PowerTransformer(method='yeo-johnson', standardize=True,copy=True)
normalized_columns=pt.fit_transform(columns_to_normalize)
normalized_columns=pd.DataFrame(normalized_columns, columns=names_col)
normalized_columns.plot.kde(bw_method=3)


# After normalizing each of the variables, we can see that they are all now standardized, they follow a bell shaped curve and they are centered at 0.

# ### KNN Analysis

# In[9]:


df=pd.read_csv(r"C:\Users\Owner\Downloads\DSC 360\pokemon.csv")
X=df[['total','hp','attack','defense','sp_defense','sp_attack','speed','generation']]
Y=df['type1']


# In[10]:


from sklearn.model_selection import train_test_split
X_train,X_val,y_train,y_val=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=20)


# In[11]:


from sklearn import neighbors
KNN_model=neighbors.KNeighborsClassifier(n_neighbors=10,n_jobs=-1)
KNN_model.fit(X_train,y_train)


# In[12]:


pred=KNN_model.predict(X_val)
print("Accuracy={}%".format((sum(y_val==pred)/y_val.shape[0])*100))


# In[13]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit_transform(X_train)
scaler.transform(X_val)


# In[16]:


from sklearn import neighbors
KNN_model=neighbors.KNeighborsClassifier(n_neighbors=10,n_jobs=-1)
KNN_model.fit(X_train,y_train)


# In[17]:


pred=KNN_model.predict(X_val)
print("Accuracy={}%".format((sum(y_val==pred)/y_val.shape[0])*100))


# Overall there are a lot of interesting things that can be found within the dataset, including the generation breakdowns, the correlation between each of the features, and the fact that none of the features are normally distributed. The k nearest neighbors test was quite unsuccessful in this instance, as I was not able to predict the primary type of any pokemon with over 23% accuracy.

# In[ ]:




