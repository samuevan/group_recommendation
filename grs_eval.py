'''
This script computes metrics to rankings generated to groups of users
'''


import pandas as pd
import numpy as np
import math
import ipdb
import os
import re
import sys
import random
import glob
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


results_user_id_regex_str = "(\d+)"
results_user_id_regex = re.compile(results_user_id_regex_str)

float_regex = r'[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?'
results_items_regex_str = "(\d+):({0})".format(float_regex)
results_items_regex = re.compile(results_items_regex_str)

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


def construct_groups_ground_truth(groups,test_ratings):
    #TODO fazer a implementacao tambem como dicionario
    #TODO incluir os ratings no retor de retorno
    '''
    This function constructs the ground truth for the groups
    We consisider an item as relevant to a group if all the users in the group
    rated the item with 4 or more in the test data*

    *actually, if we are using the same pre processing we used in ERA we filter
    we are considering all the items in the test set as relevants, since they were
    previously filtred by the users median
    '''


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
    #ipdb.set_trace()
    for i in range(min(num_items_to_eval,len(rank_recomm))):
        kx = rank_recomm[i] #cada linha tem um item e o score da regressao
        if kx.__class__ == tuple:
            kx = kx[0]
        #kx_in_test = [True for item_rat in test if item_rat[0] == kx]
        if kx in ground_truth: #len(kx_in_test) > 0:
            #print(kx)
            hits = hits+1;

    prec = hits/num_items_to_eval

    return prec

def grs_recall_at(rank_recomm, ground_truth,num_items_to_eval=10):
    hits = 0.0

    for i in range(min(num_items_to_eval,len(rank_recomm))):
        kx = rank_recomm[i] #cada linha tem um item e o score da regressao
        if kx.__class__ == tuple:
            kx = kx[0]

        if kx in ground_truth:
            hits = hits+1;
    if len(ground_truth) == 0:
        prec = 0.0
    else:
        prec = hits/len(ground_truth)


    return prec


METRIC_FUNCTION = {'precision':grs_precision_at,
                'recall':grs_recall_at}

def grs_avg_metric_at(group_rankings,groups_gt,metric='precision',num_items_to_eval=10):

    prec = 0.0
    num_groups_with_test = 0

    for grp_id, group in enumerate(groups):
        prec += METRIC_FUNCTION[metric](group_rankings[grp_id],groups_gt[grp_id],num_items_to_eval=num_items_to_eval)
        #computes the number of groups with items in test set
        if len(groups_gt[grp_id]) > 0:
            num_groups_with_test += 1

    final_prec = prec/num_groups_with_test
    return final_prec




'''rank_comm : Recommended list
test : list with the items the user already seen
'''
def grs_weighted_precision_at(rank_recomm, test,num_items_to_eval):
    hits = 0.0
    avg_prec = 0.0

    #if len(rank_recomm) < num_items_to_eval:
    #ipdb.set_trace()

    for i in range(min(num_items_to_eval,len(rank_recomm))):
        kx = rank_recomm[i] #cada linha tem um item e o score da regressao
        #kx_in_test = [True for item_rat in test if item_rat[0] == kx]
        kx_in_test = [True for item_rat in test if item_rat == kx[0]]
        if len(kx_in_test) > 0:
            hits = hits+1;
            avg_prec += hits/(i + 1);

    #print "hits: " + str(hits) + "avg_prec: " +str(avg_prec)
    #a = input()
    if  hits != 0:
        prec = avg_prec/min(len(test),num_items_to_eval)
        return prec
    else:
        return 0




def grs_MAP(groups_rankings, groups_gt,size_at=10):

    map_value = 0.0
    num_users_test_has_elem = 0

    if groups_rankings.__class__ == dict:
        group_keys = groups_rankings.keys()
    else:
        group_keys = range(len(groups_rankings))

    for key in group_keys:
        if len(groups_gt[key]) > 0:
            #print "user: " + str(key)
            #ipdb.set_trace()
            map_value += grs_weighted_precision_at(groups_rankings[key],groups_gt[key],size_at)
            num_users_test_has_elem += 1

    map_value = map_value/num_users_test_has_elem

    return map_value

def grs_avg_hits(groups_rankings, groups_gt,size_at=10):

    avg_hits = 0.0
    num_users_test_has_elem = 0

    if groups_rankings.__class__ == dict:
        group_keys = groups_rankings.keys()
    else:
        group_keys = range(len(groups_rankings))


    for key in group_keys:
        if len(groups_gt[key]) > 0:
            item_hits = [x for x in groups_rankings[key] if x[0] in groups_gt[key]]
            avg_hits += len(item_hits)
            num_users_test_has_elem += 1

    avg_hits /= num_users_test_has_elem

    return avg_hits



if __name__ == "__main__":

    basedir = sys.argv[1]
    groups_file = sys.argv[2]
    test_file = sys.argv[3]

    groups = read_groups(groups_file)



    files = sorted(glob.glob(os.path.join(basedir,'*.gout')))
    partitions = set([re.search('u[1-9]+',fname).group(0) for fname in files])
    partitions = sorted(list(partitions))

    for part in partitions:
        #read the test to the specific partition
        test = read_ratings_file(os.path.join(basedir,part+'.test'))
        #construct the groups ground truth considering the specified test file
        groups_gt = construct_groups_ground_truth(groups,test)
        #select the recommended files to the specified partition
        part_files = [curr_file for curr_file in files if part in curr_file]

        for input_ranking_file in part_files:
            input_rankings = rankings_dict(input_ranking_file,10)
            map_value = grs_MAP(input_rankings,groups_gt)
            prec = grs_avg_metric_at(input_rankings,groups_gt,metric='precision')
            prec1 = grs_avg_metric_at(input_rankings,groups_gt,metric='precision',num_items_to_eval=1)
            prec5 = grs_avg_metric_at(input_rankings,groups_gt,metric='precision',num_items_to_eval=5)
            recall = grs_avg_metric_at(input_rankings,groups_gt,metric='recall')
            avg_hits = grs_avg_hits(input_rankings,groups_gt)
            basename = os.path.basename(input_ranking_file)

            print("{0}\t{1:.4f}\t{2:.4f}\t{3:.4f}\t{4:.4f}\t{5:.4f}\t{6:.4f}".format(basename,map_value,prec1,prec5,prec,recall,avg_hits))



    #ipdb.set_trace()
