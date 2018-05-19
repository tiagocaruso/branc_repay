import numpy as np
import pandas as pd
import pytz
from datetime import datetime
import time
import re

###########################################################################
# This file takes as input df_contact, df_call, df_log and df_default
# Creates relevant variables
# Cleans the data
# Exports as output df_repaid ready for ML model
###########################################################################

# Importing files
df_contact = pd.read_pickle('df_contact')
df_call = pd.read_pickle('df_call')
df_log = pd.read_pickle('df_log')

# Importing tags
df_default = pd.DataFrame.from_csv('/Users/tiagocaruso/Documents/Jobs/Branch/user_logs/user_status.csv').reset_index()


#################################
# Working on df_default
#################################
# Creating Disbursement_epoch
pattern = '%Y-%m-%d'
df_default['disbursement_epoch'] = df_default['disbursement_date'].apply(lambda x: int(time.mktime(time.strptime(str(x)[:10], pattern))))

#################################
#Creating variables from df_call
#################################
df_call['user_id'] = df_call['user'].apply(lambda x: int(x[5:]))

# Working with cached_name
df_call['cached_name'] = df_call['cached_name'].fillna('NA')
df_call['cached_name_small'] = df_call['cached_name'].apply(lambda x: x.encode('utf-8').lower())
mum_list = ['mum','mom','mòm','my mum','mumy','mummy','mother','mama']
dad_list = ['dad','daddy','father','papa']
love_list = ['darling','wife','chèr','babe','sweetheart','sweetiebabey','hubby']
df_call['mum_call'] = df_call['cached_name_small'].isin(mum_list)
df_call['dad_call'] = df_call['cached_name_small'].isin(dad_list)
df_call['love_call'] = df_call['cached_name_small'].isin(love_list)

# Working on country_iso
df_call['country_iso'] = df_call['country_iso'].fillna("NA")

# working on datetime
tz = pytz.timezone('Africa/Nairobi')
df_call['local_time'] = df_call['datetime'].apply(lambda x: datetime.fromtimestamp(int(x)/1000, tz= tz))
df_call['local_hour'] = df_call['local_time'].apply(lambda x: x.hour)
df_call['late_call'] = df_call['local_hour'].apply(lambda x: x>21 or x<5)
df_call['early_call'] = df_call['local_hour'].apply(lambda x: x>=5 and x<9)

## Creating df_call_agg
f = {'mum_call':np.mean, 'dad_call': np.mean, 'love_call':np.mean ,
     'late_call': np.mean, 'early_call': np.mean,
     'is_read': np.nanmean,'features_video': np.nanmean,
     'country_iso':lambda x:x.value_counts().index[0],
     'datetime':'count'}
df_call_agg = df_call.groupby('user_id').agg(f).reset_index()
df_call_agg['is_read'] = df_call_agg['is_read'].fillna(0)
df_call_agg['features_video'] = df_call_agg['features_video'].fillna(0)
df_call_agg.columns = ['user_id', 'early_call', 'is_read', 'mum_call', 'late_call',
                       'features_video','dad_call','country_iso','love_call','n_calls']

###################################
#Creating variables from df_contact
###################################
df_contact['user_id'] = df_contact['user'].apply(lambda x: int(x[5:]))

# Creating df_contact_agg
g = {'display_name':'nunique', 'times_contacted': np.sum, 'device': 'nunique'}
df_contact_agg = df_contact.groupby('user_id').agg(g).reset_index()
df_contact_agg.columns = ['user_id','n_devices','times_contacted','n_contacts']

###################################
#Creating variables from df_log
###################################
df_log['user_id'] = df_log['user'].apply(lambda x: int(x[5:]))

# Getting rid of Branch contacts to avoid spillover
df_log = df_log[df_log['sms_address'] != 'Branch-Co']

# Creating variables based on the body of the message
bet_list = ['s-pesa','betway','sportpesa','betpawa','bet:','lotto']
df_log['message_body_small'] = df_log['message_body'].apply(lambda x: x.encode('utf-8').lower())
df_log['m_pesa_log'] = df_log['message_body_small'].apply(lambda x: 'm-pesa' in x)
df_log['loan_log'] = df_log['message_body_small'].apply(lambda x: 'loan' in x)
df_log['bet_log'] = df_log['message_body_small'].apply(lambda x: any(b in x for b in bet_list))
df_log['overdue_log'] = df_log['message_body_small'].apply(lambda x: 'overdue' in x)

# Finding m-pesa balance
def find_mpesa_balance(x):
    try:
        balance = x.split('m-pesa balance')[1].split('.')[0]
        balance = round(float(re.sub("[^0-9]", "", balance)),2)
        return balance
    except:
        return np.nan

df_log['mpesa_balance'] = df_log['message_body_small'].apply(find_mpesa_balance)

# Creating df_log_agg
g = {'m_pesa_log':np.sum,'loan_log':np.mean,'bet_log':np.mean,'overdue_log':np.mean,
     'sms_type':lambda x:x.value_counts().index[0],
    'mpesa_balance': np.nanmean, 'message_body_small':'count'}
df_log_agg = df_log.groupby('user_id').agg(g).reset_index()
df_log_agg.columns = ['user_id','sms_type','loan_log','overdue_log','mpesa_balance','m_pesa_log','bet_log','n_logs']

#Filling NA with median
df_log_agg['mpesa_balance'] = df_log_agg['mpesa_balance'].fillna(df_log_agg['mpesa_balance'].median())

###################################
# Merging the data frames
###################################
df_full = pd.merge(df_default,df_call_agg, how='left',on='user_id')
df_full = pd.merge(df_full,df_contact_agg, how='left',on='user_id')
df_full = pd.merge(df_full,df_log_agg, how='left', on='user_id')

###################################
# Cleaning the data
###################################

# Creating depandent variable
df_full['repaid'] = df_full['status'].apply(lambda x: int(x=='repaid'))

# Getting the dummies
df_dummies = pd.get_dummies(df_full, columns=['country_iso','sms_type'])

# Keeping the relevant columns
keep_cols = ['disbursement_epoch','early_call','is_read','mum_call','late_call','features_video','dad_call',
             'love_call','n_devices','times_contacted','n_contacts','n_logs','n_calls','overdue_log','bet_log',
             'm_pesa_log','mpesa_balance','loan_log','repaid','country_iso_KE','country_iso_NA','country_iso_US',
             'sms_type_1.0','sms_type_2.0']
df_clean = df_dummies[keep_cols]

###################################
# Exporting the data
###################################
df_clean.to_csv('df_repaid')

