import pandas as pd
import numpy as np




input_headers = ("user_id", "item_id", "rating")#, "timestamp")
types = {'user_id':np.int32,'item_id':np.int32,'rating':np.float64}#,'timestamp':np.int32}

def read_ratings_file(addr, sep="\t",default_headers=input_headers,default_types=types):

    frame = pd.read_csv(addr, sep,header=None)
    #remove the timestamp column
    if len(frame.columns) == 4:
        del frame[3]
    frame.columns = input_headers
    return frame



'''
receives a path to a file whose each line contains group_id:u1,u2,u3,u4,u5
and returns a dictionary or an array like data structure containing the groups
'''
def read_groups(path,return_dict=False):

    if return_dict:
        groups = {}
    else:
        groups = []
    with open(path,'r') as inputf:
        for line in inputf:
            group_id,users = line.strip().split(':')
            users = [int(x) for x in users.split(',')]

            if return_dict:
                groups[int(group_id)]  = users
            else:
                groups.append(users)

    return groups



def construct_mapping(dataset):

    users = sorted(dataset.user_id.unique())
    users_pos_map = {}
    pos_users_map = {}
    for i,u in enumerate(users):

        users_pos_map[u] = i+1 #since i starts from 0 and we want that users ids starts from 1
        pos_users_map[i+1] = u
    return users_pos_map, pos_users_map
