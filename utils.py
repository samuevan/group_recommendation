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



results_user_id_regex_str = "(\d+)"
results_user_id_regex = re.compile(results_user_id_regex_str)

float_regex = r'[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?'
results_items_regex_str = "(\d+):({0})".format(float_regex)
results_items_regex = re.compile(results_items_regex_str)

def rankings_dict(paths,rank_size=100):
    """Generates a dictionary from a file of ranking algorithm results.

    path -- path to the results file.

    output: Dict where each key contains [(item,item_score)]
    """
    d = defaultdict(list)

    for path in paths:	
        f = open(path, 'r')
        for line in f:
            user_id_result = results_user_id_regex.match(line)
            user_id = int(user_id_result.group(0))
            ranking = results_items_regex.findall(line, user_id_result.end())
            # Assuming results are already sorted in descending order
            items = [(int(i[0]),float(i[1])) for i in ranking[:rank_size]]
            d[user_id].append(items)
        f.close()
    return d




'''
Receives a pandas dataset which the columns user_id, item_id and ratings
returns two dictionaries

1) user_pos_map : key = user_id, value = position
2) pos_user_map : key = position, value = user_id
'''
def construct_mapping(dataset):

    users = sorted(dataset.user_id.unique())
    users_pos_map = {}
    pos_users_map = {}
    for i,u in enumerate(users):

        users_pos_map[u] = i+1 #since i starts from 0 and we want that users ids starts from 1
        pos_users_map[i+1] = u
    return users_pos_map, pos_users_map



'''
Receives a set rankings composed by pairs (item,score) and returns a list
of the items in the intersection of all rankings 
'''
def rank_intersection(rankings):

    ranks_without_score = []
    for rank in rankings:
        ranks_without_score.append([x[0] for x in rank])

    initial = set(ranks_without_score[0])
    for l in ranks_without_score[1:]:
        initial = initial.intersection(l)
    return initial


'''
Receives a set rankings composed by pairs (item,score) and returns a list
of all the items in the union of rankings 
'''
def rank_union(rankings):

    ranks_without_score = []
    for rank in rankings:
        ranks_without_score.append([x[0] for x in rank])

    initial = set(ranks_without_score[0])
    for l in ranks_without_score[1:]:
        initial = initial.union(l)
    return initial


'''
Receives a groups of users (the users_ids) and a dataset listing their ratings
and returns only items rated by all the users in the group, i.e. the ratings 
intersection for the group's users.

group: list of user_ids
test_f: Pandas DataFrame, dictionary. Both contain the ratings given by each 
user indexed by the user_id and item_id
 
'''
def group_intersection(group,test_f):

    if len(group) == 0:
        return set()
    if test_f.__class__ == pd.DataFrame:
        initial = set(list((test_f[test_f.user_id == group[0]]['item_id'])))
        for user in group[1:]:
            initial = initial.intersection(list((test_f[test_f.user_id == user]['item_id'])))
        return initial
    elif test_f.__class__ == dict:
        try:
            initial = set(test_f[group[0]])
        except:
            print("ERRO"+str(group[0]))
            ipdb.set_trace()

        for user in group[1:]:
            initial = initial.intersection(test_f[user])
        return initial


'''
Receives a groups of users (the users_ids) and a dataset listing their ratings
and returns all items rated by any user in the group, i.e. the ratings 
union for the group's users.

group: list of user_ids
test_f: Pandas DataFrame, dictionary. Both contain the ratings given by each 
user indexed by the user_id and item_id
 
'''
def group_union(group,test_f):
    initial = set(list((test_f[test_f.user_id == group[0]]['item_id'])))
    for user in group[1:]:
        initial = initial.union(list((test_f[test_f.user_id == user]['item_id'])))
    return initial

