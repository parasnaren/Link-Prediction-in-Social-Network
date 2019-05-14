import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics.pairwise import cosine_similarity as cs
import time
import lightgbm as lgb
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import gc
import matplotlib.pyplot as plt
import networkx as nx

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df

# Load train data
train = pd.read_csv('train.csv')
train = reduce_mem_usage(train)
_, train = tts(train, test_size=0.1, random_state=0)
del _

# Load test data
#test = pd.read_csv('test.csv')
#test = reduce_mem_usage(test)

# Load user feature
user_features = pd.read_csv('user_features.csv')
user_features = reduce_mem_usage(user_features)

uf = user_features.head(100)
tf = train.head(100).reset_index(drop=True)
t2 = train_2.head(100).reset_index(drop=True)

# Merging the node and features
train_1 = pd.merge(train, user_features, how='left', left_on='node1_id', right_on='node_id').drop(['node_id'],axis=1)
unwanted = ['node_id','node1_id','node2_id']
train_2 = pd.merge(train_1, user_features, how='left', left_on='node2_id', right_on='node_id').drop(unwanted,axis=1)

for col in train_2.columns:
    train_2[col] = train_2[col].astype('category')
    
# Features
y = train_2['is_chat']
X = train_2.drop(['is_chat'],axis=1)

# LGB parameters
params_tuned = {
    'boost': 'gbdt',
    'learning_rate': 0.15,
    'max_depth': -1,  
    'metric':'auc',
    'num_threads': -1,
    'tree_learner': 'serial',
    'objective': 'binary', 
    'verbosity': 1,
    'num_leaves': 100,
}


# Splitting data
X_train, X_valid, Y_train, Y_valid =tts(X, y, test_size=0.2, stratify=y)

train_data = lgb.Dataset(X_train, label=Y_train,free_raw_data=False)
valid_data = lgb.Dataset(X_valid, label=Y_valid,free_raw_data=False)

gc.collect()

def save_model():
    def callback(env):
        model=env.model
        if env.iteration%100==0:
            model.save_model('model_'+str(env.iteration)+'.txt')
    callback.before_iteration = False
    callback.order = 0
    return callback

lgb_model = lgb.train(params_tuned,train_data,num_boost_round=10000,
                valid_sets = [valid_data],verbose_eval=100,early_stopping_rounds = 300,callbacks=[save_model()])


# New features
train_node_id = set(list(train.node1_id))
train_node2_id = set(list(train.node2_id))
train_node_id.update(train_node2_id)

train_node1 = train.groupby('node1_id').agg({'is_chat':['sum','mean']}).reset_index()
train_node1.columns = ['node1_id','is_chat_node1_sum','is_chat_node1_mean']

train_node2 = train.groupby('node2_id').agg({'is_chat':['sum','mean']}).reset_index()
train_node2.columns=['node2_id','is_chat_node2_sum','is_chat_node2_mean']

train_node1['count_chat'] = np.round(train_node1['is_chat_node1_sum']/train_node1['is_chat_node1_mean'])
train_node2['count_chat'] = np.round(train_node2['is_chat_node2_sum']/train_node2['is_chat_node2_mean'])

train_node1 = train_node1.fillna(0)
train_node2 = train_node2.fillna(0)

train_fe = pd.DataFrame()
train_fe['node_id'] = list(train_node_id)
train_fe = train_fe.merge(train_node1, how='left',left_on='node_id',right_on='node1_id')
del train_fe['node1_id']
train_fe = train_fe.merge(train_node2, how='left',left_on='node_id',right_on='node2_id')
del train_fe['node2_id']

train_fe = train_fe.fillna(0)

train_fe['full_chat_sum'] = train_fe['is_chat_node1_sum']+train_fe['is_chat_node2_sum']
train_fe['full_chat_count'] = train_fe['count_chat_x']+train_fe['count_chat_y']


# Graph features
chatted = train.loc[train['is_chat'] == 1]
chatted = chatted.reset_index(drop=True)

G = nx.Graph()
G = nx.from_pandas_edgelist(chatted, source='node1_id', target='node2_id',
                            edge_attr = True,)

def get_number_of_neighbors(chatted, col):
    t = [G.neighbors(chatted[col][ind]) for ind in range(len(chatted))]
    count_neighbors = []
    for ind in range(len(t)):
        temp = [val for val in t[ind]]
        count_neighbors.append(len(temp))
    return count_neighbors

#common_neighbors = [sum(1 for x in nx.common_neighbors(G, chatted['node1_id'][ind], chatted['node2_id'][ind])) for ind in range(len(chatted))]
common_neighbors = [sum(1 for x in nx.common_neighbors(G, chatted['node1_id'][ind], chatted['node2_id'][ind])) for ind in range(2)]

number_of_neighbors_1 = get_number_of_neighbors(chatted, 'node1_id')
number_of_neighbors_2 = get_number_of_neighbors(chatted, 'node2_id')

chatted['no_for_node1'] = number_of_neighbors_1
chatted['no_for_node2'] = number_of_neighbors_2

node1_number = chatted[['node1_id','no_for_node1']]
node2_number = chatted[['node2_id','no_for_node2']]
node_number = node1_number

train_fe = train_fe.merge(node1_number, how='left', left_on='node_id', right_on='node1_id')
train_fe = train_fe.merge(node2_number, how='left', left_on='node_id', right_on='node2_id')
train_fe.drop(['node1_id','node2_id'], axis=1, inplace=True)
train_fe = train_fe.fillna(0)

# astype int
train_fe['no_for_node1'] = train_fe['no_for_node1'].astype('int8')
train_fe['no_for_node2'] = train_fe['no_for_node2'].astype('int8')

train_fe['no_of_neigh'] = train_fe.no_for_node1 | train_fe.no_for_node2

chatted['common_neighbors'] = common_neighbors
#final = train_fe[['node_id','full_chat_mean','full_chat_sum','full_chat_count']].reset_index(drop=True)

#############################################################

# New features
train_1 = pd.merge(train, user_features, how='left', left_on='node1_id', right_on='node_id').drop(['node_id'],axis=1)
#unwanted = ['node_id','node1_id','node2_id']
train_2 = pd.merge(train_1, user_features, how='left', left_on='node2_id', right_on='node_id').drop(unwanted,axis=1)

for col in train_2.columns:
    train_2[col] = train_2[col].astype('category')
    
train_2 = train_2.merge(train_fe, how='left', left_on='node1_id', right_on='node_id').drop(['node1_id','node2_id'], axis=1)
train_2 = train_2.merge(train_fe, how='left', left_on='node2_id', right_on='node_id').drop(['node1_id','node2_id','node_id'], axis=1)
    
# Features
y = train_2['is_chat']
X = train_2.drop(['is_chat'],axis=1)

# LGB parameters
params_tuned = {
    'boost': 'gbdt',
    'learning_rate': 0.15,
    'max_depth': -1,  
    'metric':'auc',
    'num_threads': -1,
    'tree_learner': 'serial',
    'objective': 'binary', 
    'verbosity': 1,
    'num_leaves': 100,
}


# Splitting data
from sklearn.model_selection import train_test_split as tts
X_train, X_valid, Y_train, Y_valid =tts(X, y, test_size=0.2, stratify=y)

train_data = lgb.Dataset(X_train, label=Y_train,free_raw_data=False)
valid_data = lgb.Dataset(X_valid, label=Y_valid,free_raw_data=False)

gc.collect()

def save_model():
    def callback(env):
        model=env.model
        if env.iteration%100==0:
            model.save_model('model_'+str(env.iteration)+'.txt')
    callback.before_iteration = False
    callback.order = 0
    return callback

lgb_model = lgb.train(params_tuned,train_data,num_boost_round=10000,
                valid_sets = [valid_data],verbose_eval=100,early_stopping_rounds = 300,callbacks=[save_model()])





