import os
import numpy as np
import pandas as pd
import json 

# This file gets all txt files from local computer and exports to pickle 3 dataframes
# for contacts, calls and logs

current_path = '/Users/tiagocaruso/Documents/Jobs/Branch/user_logs'
status_path = '/Users/tiagocaruso/Documents/Jobs/Branch/user_logs/user_status.csv'

# This loop gets all files and creates dataframes with user_id, device and path
# One dataframe for contacts, one for calls, one for logs
contact_l =[] 
call_l =[]
log_l = []
for subdir, dirs, files in os.walk(current_path):
    for file in files:
        filepath = subdir + os.sep + file
        user = str(subdir).split('/')[-2]
        device = str(subdir).split('/')[-1]
        file_dict = {'user': user, 'device': device, 'filepath':filepath}
        if filepath == status_path:
            continue
        elif file == 'collated_sms_log.txt':
            log_l.append(file_dict)
        elif file == 'collated_call_log.txt':
            call_l.append(file_dict)
        elif file == 'collated_contact_list.txt':
            contact_l.append(file_dict)

df_contact_path = pd.DataFrame(contact_l)
df_call_path = pd.DataFrame(call_l)
df_log_path = pd.DataFrame(log_l)

# Function takes df_with_filepah as input and outputs df_with_obs
def from_path_to_obs(data):
    final_df = pd.DataFrame()
    for index, row in data.iterrows():
        with open(row['filepath']) as f:
            lines = f.read().splitlines()
        obs = json.loads(lines[0]) 
        df = pd.DataFrame(obs)
        df['user'] = row['user']
        df['device'] = row['device']
        final_df = pd.concat([final_df,df], axis=0)
    return final_df   

df_contact = from_path_to_obs(df_contact_path)
df_call = from_path_to_obs(df_call_path)
df_log = from_path_to_obs(df_log_path)

#exporting to pickle
df_contact.to_pickle('df_contact')
df_call.to_pickle('df_call')
df_log.to_pickle('df_log')
