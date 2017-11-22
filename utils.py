import pandas as pd
import numpy as np
import ipdb
import re
import os
import glob
from collections import defaultdict



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





def read_letor_data(datafile):
    results_relevance_regex_str = "(\d+)"
    results_relevance_regex = re.compile(results_relevance_regex_str)

    results_user_id_regex_str = "qid:(\d+)"
    results_user_id_regex = re.compile(results_user_id_regex_str)

    results_item_id_regex_str = "#docid\s*=\s*(\d+)"
    results_item_id_regex = re.compile(results_item_id_regex_str)

    float_regex = r'[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?'
    results_items_regex_str = "\d+:({0})".format(float_regex)
    results_items_regex = re.compile(results_items_regex_str)

       
    dataset = []

    with open(datafile,'r') as f1:
        for line1 in f1:
            relevance = int(results_relevance_regex.match(line1).group(0))
            user_id = int(results_user_id_regex.findall(line1)[0])
            item_id = int(results_item_id_regex.findall(line1)[0])
            attributes = results_items_regex.findall(line1)
            
            atts_values = [float(att[1]) for att in attributes]
            instance = (relevance,user_id,item_id) + tuple(atts_values)            
            dataset.append(instance)

    labels = ['relevance','user_id','item_id']
    [labels.append('att'+str(i+1)) for i in range(len(dataset[0])-len(labels))]
    types = {'user_id':np.int32,'item_id':np.int32,'relevance':np.int32}

    
    data_pandas = pd.DataFrame(dataset,columns=labels)

    for key in types:
        data_pandas[key] = data_pandas[key].astype(types[key])
    #ipdb.set_trace()
    return data_pandas



def contruct_groups(basedir,outdir,dist_file,
                        num_parts=5,
                        num_groups=3000,
                        list_ggroup_sizes=[3,5],
                        list_tresh=[0.27],
                        list_min_items=[5]):

    for gsz in list_group_sizes:
        for thresh in list_tresh:
            for min_items in list_min:    
                 cmd = "python groups_generator.py --before --strong  --base {} "\
                    "--num_parts {} --num_groups {} --group_size {} --t_value {} "\
                    "--min_group_items {} --dist_file {} -o {}"







def save_letor_data_from_pandas(dataframe,out_file):

    with open(out_file,'w') as out:
        for item in dataframe.iterrows():
            #item[0] stores the item index and data[1] stores the actual row values
            data_item = item[1]
            s = '{relevance:.0f} qid:{user_id:.0f} '
            for i in range(1,len(data_item[2:])):
                s += str(i)+':{att'+str(i)+'} '

            s += '#docid = {item_id:.0f} inc = 0.0 prob = 0.0\n'
            out.write(s.format(**data_item))


        
def merge_letor_files(file1,file2):


    data1 = read_letor_data(file1)
    data2 = read_letor_data(file2)

    data1.sort_values(by=['user_id','item_id'], inplace = True)
    data2.sort_values(by=['user_id','item_id'], inplace = True)

    del(data2['relevance'])
    del(data2['user_id'])
    del(data2['item_id'])
    
    #the columns of the dataset data1 minus the columns relevance, user_id and item_id
    num_valued_atts = len(data1.columns)-3 
    data2_columns = data2.columns

    new_column_names = {data2_columns[pos]:'att'+str(num_valued_atts+pos+1) 
                        for pos in range(len(data2_columns))}


    data2.rename_axis(new_column_names,axis=1,inplace=True)
    data2.reset_index(drop=True,inplace=True)

    data3 = pd.concat([data1,data2],axis=1)

    return data3           


def merge_letor_files_in_folders(folder1,folder2,out_folder):

    if not os.path.isdir(out_folder):
        os.mkdir(out_folder)

    letor_files_folder1 = sorted(glob.glob(os.path.join(folder1,'*.letor')))
    letor_files_folder2 = sorted(glob.glob(os.path.join(folder2,'*.letor')))

    try:
        if not len(letor_files_folder1) == len(letor_files_folder2):
            raise Exception
    except:
        print("The folders cointain different number of letor files")

    for f1,f2 in zip(letor_files_folder1,letor_files_folder2):
        merged = merge_letor_files(f1,f2)
        save_letor_data_from_pandas(merged,os.path.join(out_folder,
                                                os.path.basename(f1)))


