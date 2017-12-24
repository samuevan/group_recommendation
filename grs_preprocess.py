import os
import sys
import argparse
import utils
import groups_generator as gg
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import ipdb

'''receives a dataframe and modifies this dataframe inserting new ids to
users and items. These new ids are sequential. Returns the objects used to 
do the transformations.
'''
def encode_users_items(df,columns=['user_id','item_id']):

    label_encoders = {}
        
    for col in columns:
        le = LabelEncoder()
        uniq_values = df[col].unique()
        le.fit(uniq_values)
        df['new_'+col] = le.transform(df[col].values)
        label_encoders[col] = le

    return label_encoders

'''
Receives the groups checkins and a label encoder and apply the encoder to the 
item ids in the groups checkins.

Modify the groups checkins dataframe inplace.
'''
def encode_group_checkins(df_grs_chk,label_encoders,item_col="item_id"):
    df_grs_chk["new_"+item_col] = label_encoders[item_col].transform(
                                df_grs_chk[item_col].values)


'''
Receives the groups dictionary (key is the group_id and value is the
list of users ids that are part of the group) and a label encoder.

Returns a new dictionary where the users ids are encoded using the received 
label encoder.
'''
def encode_group_members(groups,label_encoders,user_col="user_id"):

    if type(groups) == dict:
        encoded_groups = defaultdict(list)
        for group_key in groups:
            encoded_groups[group_key] = label_encoders[user_col].transform(
                                                                groups[group_key])
    else:
        encoded_groups = []
        for group in groups:        
            encoded_groups.append(label_encoders[user_col].transform(group))
        
    return encoded_groups
    

def encode_all(df,df_grp_chk,groups,
                columns=['user_id','item_id'],
                replace=False):

    label_encoders = encode_users_items(df,columns)
    encoded_groups = encode_group_members(groups,label_encoders,user_col=columns[0])
    encode_group_checkins(df_grp_chk,label_encoders,item_col=columns[1])

    if replace:
        groups = encoded_groups
        #update df column names
        for col in columns:
            del df[col]
            df.rename(index=str,columns={'new_'+col:col},inplace=True)
        #update items column name
        del df_grp_chk[columns[1]]
        df_grp_chk.rename(index=str,columns={'new_'+columns[1]:columns[1]},inplace=True)

    return df,df_grp_chk,encoded_groups,label_encoders


def save_encoded(df,outfile):
    df.sort_values(by='user_id',inplace=True)
    df.to_csv(outfile,columns=['user_id','item_id','timestamp'],
              header=None,index=False,sep='\t')

def save_encoders(label_encoders,out_dir):

    for enc_key in label_encoders:
        df_encoder = pd.DataFrame(pd.Series(label_encoders[enc_key].classes_))
        df_encoder.to_csv(os.path.join(out_dir,enc_key+'.map'),
                        index=True,
                        header=None,
                        sep='\t')
        



def parse_args():

    p = argparse.ArgumentParser()
    p.add_argument('-b','--base',type=str,required=True)
    p.add_argument('-o','--out_dir',type=str)
    p.add_argument('--groups_file',type=str)
    p.add_argument('--groups_chk',type=str)       



    parsed = p.parse_args()


    if parsed.out_dir is None:
        parsed.out_dir = os.path.dirname(parsed.base)

    if not os.path.isdir(parsed.out_dir):
        os.mkdir(parsed.out_dir)

    return parsed


if __name__ == "__main__":

    args = parse_args()
    
    data = pd.read_csv(args.base,sep='\t',
            names=['user_id','item_id','timestamp'],
            header=None)

    groups = utils.read_groups(args.groups_file,return_dict=True)
    groups_chk = pd.read_csv(args.groups_chk,sep='\t',
                  names=['user_id','item_id','timestamp'],
                  header=None) 

    datax,groups_chkx,groupsx,lesx = encode_all(data,groups_chk,groups,replace=True)


    out = os.path.join(args.out_dir,'u.data.encoded')
    save_encoded(datax,out)

    out = os.path.join(args.out_dir,'u.groups_checkins.encoded')
    save_encoded(groups_chkx,out)

    out = os.path.join(args.out_dir,'u.groups.encoded')
    gg.save_groups(groupsx,out)

    
     
