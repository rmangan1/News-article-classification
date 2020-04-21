import pandas as pd
import numpy as np
import json
import ast
import re
from core import process_text

# load train data
file = 'train_data.json'
with open(file) as train_file:
    dict_train = json.load(train_file)

# convert to dataframe
train = pd.io.json.json_normalize(dict_train)

# convert lists to dicts
train['labels'] = train.apply(lambda row: dict((x, y) for x, y in row['labels']),axis=1)

# split labels into separate columns
tmp = pd.io.json.json_normalize(train['labels'].tolist()).fillna(0)
train = train.join(pd.io.json.json_normalize(train['labels'].tolist()).fillna(0)).drop('labels', axis=1)

train['content.sections'] = train['content.sections'].apply(lambda x: x[0])

# consolidate some section names
train.loc[train['content.sections'] == 'the-new-york-times -> world', 'content.sections'] = 'news -> world'
train.loc[train['content.sections'] == 'news', 'content.sections'] = 'news -> world'

# add missing column with default value 0
train.insert(76,"health>men's health", 0)

# create list of tuples of categories (e.g. sport) and subcategories (e.g. sport>football)
category_list = []
num_columns = len(train.columns)
i = 4
while i < num_columns:
    main_cat = train.columns[i]
    subcat_list = []
    i += 1
    for j in range(i, num_columns):
        sub_cat = train.columns[j]
        # group together categories based on common starting characters
        if main_cat[:4] == sub_cat[:4]:
            subcat_list.append(sub_cat)
        else:
            category_list.append((main_cat,subcat_list))
            i = j
            break
        if j == num_columns - 1:
            category_list.append((main_cat,subcat_list))
            i = j+1

# map section names to most similar taxonomy names
map_sections = [('bi -> tech', 'science and technology'), ('bi -> politics', 'politics'),
                ('bi -> finance', 'economy, business and finance'), ('sports -> football','sport>football'),
                ('sports -> football','sport'),
               ('bi -> lifestyle','lifestyle'), ('lifestyle -> mens-health','lifestyle'),
                ('lifestyle -> mens-health','health'), ('lifestyle -> mens-health','health>men\'s health'),
               ('lifestyle -> womens-health','lifestyle'),
               ('lifestyle -> womens-health','health>women\'s health'), ('lifestyle -> womens-health','health'),
               ('bi -> sports','sport'),
               ('entertainment','arts, culture and entertainment'), ('the-new-york-times -> entertainment','arts, culture and entertainment'),
               ('news -> politics','politics'), ('lifestyle -> beauty-health','lifestyle'),
                ('lifestyle -> relationships-and-weddings','society>family and relationship'),
                ('lifestyle -> relationships-and-weddings','society'),
                ('lifestyle -> relationships-and-weddings','lifestyle'),
                ('lifestyle -> food-travel','lifestyle'),
                ('sports','sport'),
                ('lifestyle -> money','lifestyle'), ('lifestyle -> money','economy, business and finance'),
                ('Style','lifestyle>style & fashion'), ('Style','lifestyle'),
                ('lifestyle','lifestyle'),
                ('People','lifestyle>people', 'People','lifestyle')]

# assume section names are accurate, since they were manually entered
# assign confidence=1 to taxonomy label based on section maps
for pair in map_sections:
    train.loc[train['content.sections'] == pair[0], pair[1]] = 1

# indices for "main" categories
col_index = np.concatenate((np.array([-1]), np.cumsum(np.array([4] + [len(i[1]) + 1 for i in category_list]))))[:-1]

# some section names are lacking obvious corresponding taxonomy
# find taxonomies which are correlated with these sections
# then if confidence>0.3 set value to 1
section_list = ['news -> world', 'bi -> strategy']
for section in section_list:
    train['content.sections_binary'] = [1 if x == section else 0 for x in train['content.sections']]
    corr = train.iloc[:,col_index].corr()['content.sections_binary']
    corr_index = corr[corr > 0.04].nlargest(9).index
    for index in corr_index[1:]:
        train.loc[(train['content.sections'] ==  section) & (train[index] > 0.3),index] = 1
train = train.drop(['content.sections_binary'], axis=1)

# change label to 1 if >= 0.5, else 0
train.iloc[:,4:] = train.iloc[:,4:].apply(lambda x: [1 if y >= 0.5 else 0 for y in x])

train = process_text(train)

# split into features (X) and labels (y)
X_train = train['content.fullTextHtml']
y_train = train.iloc[:,4:]
y_train = y_train.loc[:,(y_train!=0).any(axis=0)] # drop columns with all zeros

# save to X, y to file
train_store = pd.HDFStore('train_store.h5')
train_store["X_train"] = X_train
train_store["y_train"] = y_train
