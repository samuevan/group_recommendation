"""
With this script one can generates groups using as input recommendation 
datasets in the format of the Movielens datasets. 

For the recommendation aggregation:
    The groups are outputed as an auxiliar file containing a group id followed 
    by the users's ids that compose the group    
"""


import pandas as pd
import numpy as np
import ipdb


#TODO colocar as funcoes de leitura em um script separado

input_headers = ("user_id", "item_id", "rating")#, "timestamp")
types = {'user_id':np.int32,'item_id':np.int32,'rating':np.float64}#,'timestamp':np.int32}

def read_ratings_file(addr, sep="\t",default_headers=input_headers,default_types=types):
    
    frame = pd.read_csv(addr, sep,header=None)
    #remove the timestamp column
    if len(frame.columns) == 4:
        del frame[3]
    frame.columns = input_headers            
    return frame


def random_groups(initial_dataset,out_file,group_size=5):

    users_ids = initial_dataset.user_id.unique()
    np.random.shuffle(users_ids)

    groups = []
    ipdb.set_trace()
    for ini in range(0,len(users_ids),group_size):
        groups.append(users_ids[ini:min(ini+group_size,len(users_ids))])
       
    with open(out_file,'w') as out_f:
        for group_id in range(len(groups)):
            group_str = ",".join([str(x) for x in groups[group_id]])
            out_f.write("{0}:{1}\n".format(group_id,group_str))





def similarity_groups(initial_dataset, out_file, group_size=5, threshold=0.5):
    #TODO
    print("Unimplemented")


def run(dataset,out_file):
    initial_dataset = read_ratings_file(dataset)
    random_groups(initial_dataset,out_file)        
