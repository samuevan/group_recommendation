'''
This script computes metrics to rankings generated to groups of users
'''


import pandas as pd
import numpy as np
import math
import ipdb
import os
import sys
import random
import matplotlib.pyplot as plt
import argparse
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine
from itertools import combinations

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



def rankings_dict(path,rank_size=100):
    """Generates a dictionary from a file of ranking algorithm results.

    path -- path to the results file.

    output: Dict where each key contains [(item,item_score)]
    """
    f = open(path, 'r')
    d = {}
    for line in f:
        user_id_result = results_user_id_regex.match(line)
        user_id = int(user_id_result.group(0))
        ranking = results_items_regex.findall(line, user_id_result.end())
        # Assuming results are already sorted in descending order
        items = [(int(i[0]),float(i[1])) for i in ranking[:rank_size]]
        d[user_id] = items
    f.close()
    return d


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

'''
This function constructs the ground truth for the groups
We consisider an item as relevant to a group if all the users in the group
rated the item with 4 or more in the test data*

*actually, if we are using the same pre processing we used in ERA we filter 
we are considering all the items in the test set as relevants, since they were 
previously filtred by the users median
'''
def construct_groups_ground_truth(groups,test_ratings):
#TODO fazer a implementacao tambem como dicionario

    groups_gt = []
        
    for group_index, group in enumerate(groups):
        ground_truth = set(test_ratings[test_ratings.user_id == group[0]]['item_id'])
        #since we previously filtered the ratings by a treshold to construct 
        #the GT we just computes the users' items intersection
        for user in group[1:]:
            user_items = test_ratings[test_ratings.user_id == user]['item_id']
            ground_truth = ground_truth.intersection(user_items)

        groups_gt.append(ground_truth)

    return groups_gt



def grs_precision_at(rank_recomm, ground_truth,num_items_to_eval=10):
    hits = 0.0
    prec = 0.0
    #print rank_recomm
    #print test
    #Considerando somente os 100 primeiros
    for i in range(num_items_to_eval):
        kx = rank_recomm[i] #cada linha tem um item e o score da regressao
        #kx_in_test = [True for item_rat in test if item_rat[0] == kx] 
        if kx in ground_truth: #len(kx_in_test) > 0:
            hits = hits+1;

    prec = hits/num_items_to_eval

    return prec



if __name__ == "__main__":

    input_ranking = sys.argv[1]
    groups_file = sys.argv[2]
    test_file = sys.argv[3]

    input_ranking = rankings_dict(input_ranking,10)
    groups = read_groups(groups_file)
    test = read_ratings_file(test_file)
        
    ipdb.set_trace()
    groups_gt = construct_groups_ground_truth(groups,test)

    for group in groups:
        prec = grs_precision_at()
    
