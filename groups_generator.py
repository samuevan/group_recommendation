"""
With this script one can generates groups using as input recommendation
datasets in the format of the Movielens datasets.

For the recommendation aggregation:
    The groups are outputed as an auxiliar file containing a group id followed
    by the users's ids that compose the group
"""


import pandas as pd
import numpy as np
import math
import ipdb
import os
import random
import time
#import matplotlib.pyplot as plt
import argparse
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine
from itertools import combinations
import networkx as nx
import utils
import multiprocessing as mp
from multiprocessing import Process
from multiprocessing import Queue,Manager


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




def make_random_group(users_ids,group_size):

    group = []
    for _ in range(group_size):
        user = users_ids[random.randint(0,len(users_ids)-1)]
        while user in group:
            user = users_ids[random.randint(0,len(users_ids)-1)]
        group.append(user)

    return group

'''
This function construct a random group considering the group size, the minimum number
of items in the group intersection and a minimum thresold of similarity.
One can use this to construct a completely random group by seting the parameters
min_intersection and thresh to zero

return one group of users
'''

def make_random_groups_with_minimum(dataset,group_size,min_intersection,users_ids,thresh=0, similarity_function=pearsonr):
    #users_ids = dataset.user_id.unique()

    group = make_random_group(users_ids,group_size)
    simi = 1
    if thresh != 0:
        simi = group_average_similarity(group,dataset)

    while (len(group_intersection(group,dataset)) < min_intersection) or simi < thresh:
        group = make_random_group(users_ids,group_size)
        if thresh != 0:
            simi = group_average_similarity(group,dataset)


    return group



def random_groups_with_minimum(dataset,
                    min_intersection=5,
                    threshold  = 0.3,
                    group_size=5,
                    num_groups=3000,
                    similarity_function=pearsonr):

    users_ids = dataset.user_id.unique()

    groups = []
    constructed_groups = 0
    while constructed_groups < num_groups:

        #if constructed_groups%50 == 0:
        print(constructed_groups)
        g = make_random_groups_with_minimum(dataset,group_size,min_intersection, users_ids, thresh = threshold, similarity_function = similarity_function)
        groups.append(g)
        constructed_groups += 1



    return groups


def save_groups(groups,out_file):
    print("Saving group file")

    with open(out_file,'w') as out_f:
        if type(groups) == dict:
            groups_ids = sorted(groups.keys())
        else:
            groups_ids = range(len(groups))

        for group_pos,group_id in enumerate(groups_ids):
            group_str = ",".join([str(x) for x in groups[group_id]])
            out_f.write("{0}:{1}\n".format(group_pos,group_str))





def group_average_similarity(group,dataset,simi_func=pearsonr):
    avg_corr_g = 0.0
    i = 0
    for u1,u2 in combinations(group,2):
        #x = calc_similarity(u1,u2,dataset,simi_func=simi_func)[0]
        x = calc_similarity(dataset[dataset.user_id == u1],dataset[dataset.user_id == u2])[0]
        #print(x)
        #ipdb.set_trace()
        #print("{0} {1} {2}".format(u1,u2,x))
        avg_corr_g += x
        i += 1

    avg_corr_g /= i

    return avg_corr_g



'''
Pearson correlation coeficient for recommender systems
Only consider the ratings for items in commom between both users.
However, the average rating of a user is computed over user's complete set
of ratings
'''
def PCC(user_a_ratings,user_b_ratings,user_a_mean,user_b_mean):

    num = sum([(arat-user_a_mean)*(brat-user_b_mean) for arat,brat in zip(user_a_ratings,user_b_ratings)])
    dena = sum([(arat-user_a_mean)**2 for arat in user_a_ratings])**0.5
    denb = sum([(brat-user_b_mean)**2 for brat in user_b_ratings])**0.5
    similarity = num/(dena * denb)

    return (similarity,1.0)


'''
deprecated
'''
"""def calc_similarity(user_a,user_b,dataset,user_a_DF = None, min_intersec_size=5,simi_func=pearsonr):

    #we can pass the DataFrame for the first user directly
    if user_a_DF.__class__ == pd.DataFrame:
        items_intersec = list(set(user_a_DF['item_id']) &
                set(dataset[dataset.user_id == user_b]['item_id']))

        user_a_all_ratings = user_a_DF['rating']

    else:
        items_intersec = list(set(dataset[dataset.user_id == user_a]['item_id']) &
                set(dataset[dataset.user_id == user_b]['item_id']))

        user_a_all_ratings = dataset[dataset.user_id==user_a]['rating']

    if len(items_intersec) < min_intersec_size:
        return (np.nan,1.0)


    if user_a_DF.__class__ == pd.DataFrame:
        user_a_ratings = [user_a_DF[user_a_DF.item_id==item_id]['rating'].get_values()[0] for item_id in items_intersec]

    else:
        user_a_ratings = [dataset[(dataset.user_id == user_a) &
                    (dataset.item_id==item_id)]['rating'].get_values()[0] for item_id in items_intersec]


    user_b_ratings = [dataset[(dataset.user_id == user_b) &
                    (dataset.item_id==item_id)]['rating'].get_values()[0] for item_id in items_intersec]

    user_b_all_ratings = dataset[dataset.user_id==user_b]['rating']

    #trata dos casos onde nao existe variabilidade nos ratings dados por um
    #usuario e a funcao de similaridade eh pearson.
    #Nao sei se eh a melhor forma de fazer isso
    '''if ((np.var(user_a_ratings) < 0.0001) or (np.var(user_b_ratings) < 0.0001)) and (simi_func == pearsonr):
        corr = 1 - ((np.mean(user_a_ratings) - np.mean(user_b_ratings))**2) / (np.mean(user_a_ratings)*np.mean(user_b_ratings))
        #print(corr)
        return (corr,1.0)'''

    if simi_func == PCC:
        simi = simi_func(user_a_ratings,user_b_ratings,np.average(user_a_all_ratings),np.average(user_b_all_ratings))

    elif (simi_func == jaccard):
        if user_a_DF.__class__ == pd.DataFrame:
            simi = simi_func(user_a_DF['item_id'],dataset[dataset.user_id == user_b]['item_id'])
        else:
            simi = simi_func(dataset[dataset.user_id == user_a]['item_id'],dataset[dataset.user_id == user_b]['item_id'])

    else:
        simi = simi_func(user_a_ratings,user_b_ratings)
    #print(simi)

    return simi
"""

#def calc_similarity(user_a,user_b,dataset,user_a_DF = None, min_intersec_size=5,simi_func=pearsonr):
#TODO remove the second value of the return
def calc_similarity(user_a_DF,user_b_DF,min_intersec_size=5,simi_func=pearsonr):
    #we can pass the DataFrame for the first user directly

    items_intersec = list(set(user_a_DF['item_id']) &
            set(user_b_DF['item_id']))

    if len(items_intersec) < min_intersec_size:
        return (np.nan,1.0)

    user_a_ratings = [user_a_DF[user_a_DF.item_id==item_id]['rating'].get_values()[0] for item_id in items_intersec]
    user_b_ratings = [user_b_DF[user_b_DF.item_id==item_id]['rating'].get_values()[0] for item_id in items_intersec]

    user_a_all_ratings = user_a_DF['rating']
    user_b_all_ratings = user_b_DF['rating']


    #trata dos casos onde nao existe variabilidade nos ratings dados por um
    #usuario e a funcao de similaridade eh pearson.
    #Nao sei se eh a melhor forma de fazer isso
    '''if (np.var(user_a_ratings) < 0.0001) or (np.var(user_b_ratings) < 0.0001) and (simi_func == pearsonr):
        corr = 1 - ((np.mean(user_a_ratings) - np.mean(user_b_ratings))**2) / (np.mean(user_a_ratings)*np.mean(user_b_ratings))
        #print(corr)
        return (corr,1.0)'''


    if simi_func == PCC:
        simi = simi_func(user_a_ratings,user_b_ratings,np.average(user_a_all_ratings),np.average(user_b_all_ratings))
    elif (simi_func == jaccard):
        simi = simi_func(user_a_DF['item_id'],user_b_DF['item_id'])
    else:
        simi = simi_func(user_a_ratings,user_b_ratings)

    return simi

def jaccard(user_a_ratings,user_b_ratings):

    num = len(set(user_a_ratings).intersection(user_b_ratings))
    den = len(set(user_a_ratings).union(user_b_ratings))

    jacc = float(num)/float(den)

    return (jacc,1.0)

def store_intersections(base,min_num_intersect = 5):
    '''Computes the size of the intersection between two users and returns
    the users wich have an intersection bigger than min_num_intersect'''

    users_keys = base.user_id.unique()
    user_neigh = []
    users_size_neigh = np.zeros(len(users_keys))
    init = time.time()
    for u1_idx,u1 in enumerate(users_keys):
        usr_time = time.time()
        u1_items = set(base[base.user_id == u1]['item_id'])
        for u2_idx,u2 in enumerate(users_keys[u1_idx+1:]):
            u1_u2_inter = u1_items.intersection(base[base.user_id == u2]['item_id'])
            if len(u1_u2_inter) > min_num_intersect:
                #user_neigh.append(u2)
                users_size_neigh[u1_idx] += 1


    #TODO incomplete, I need to construct the dataframe of users and neighboors

        print("u1 {0} u2 {1} time {2} num {3}".format(u1_idx,u2_idx, (time.time() - usr_time),users_size_neigh[u1_idx]))
    end = time.time() - init

    print("end "+str(end))




def construct_group(initial_dataset, users_similarity_matrix,
                    similarity_function=pearsonr,
                    group_size=4,
                    threshold=0.27):

    users = initial_dataset.user_id.unique()
    num_users = len(users)

    #while not Q.empty():
    group_users = []

    first_user = users[random.randint(0,num_users-1)]
    #extract the dataframe that represents the first user, so we do not need
    #to make this extrction in every interation of the next loop
    first_user_DF = initial_dataset[initial_dataset.user_id == first_user]
    group_users = [first_user]

    for _ in range(group_size-1):
        usr2 = users[random.randint(0,num_users-1)]

        usr2_DF = initial_dataset[initial_dataset.user_id == usr2]
        simi,p_value = calc_similarity(first_user_DF,usr2_DF,
                                simi_func=similarity_function)

        #    users_similarity_matrix[first_user][usr2] = simi
        #else:
        #    simi = users_similarity_matrix[first_user][usr2]

        attempts = 0
        while simi < threshold or usr2 in group_users:

            #no caso de rodar muitas vezes e nao encontrar outro user similar ao primeiro
            #escolho outro usuario como semente
            if attempts > 200:
                #caso soh tenha um usuario no grupo eu altero esse usuario
                if len(group_users) == 1:
                    first_user = users[random.randint(0,num_users-1)]
                    first_user_DF = initial_dataset[initial_dataset.user_id == first_user]
                    group_users = [first_user]
                #caso tenha mais de um usuario eu passo a usar outro usuario do grupo como semente
                else:
                    first_user = group_users[random.randint(1,len(group_users)-1)]

                attempts = 0

            usr2 = users[random.randint(0,num_users-1)]
            #every time we draw a new pair of users which similarity/correlation
            #was not yet computed we do it
            #if users_similarity_matrix[first_user][usr2] == -999:
            usr2_DF = initial_dataset[initial_dataset.user_id == usr2]
            simi,p_value = calc_similarity(first_user_DF,usr2_DF,
                                simi_func=similarity_function)
            #    users_similarity_matrix[first_user][usr2] = simi

            #else:
            #    simi = users_similarity_matrix[first_user][usr2]

            attempts += 1
        #print(simi)
        #ipdb.set_trace()
        group_users.append(usr2)

    return group_users
    #with mutex:
    #groups.append(group_users)


'''
this function returns statistics for the groups
The number of unique users
The number of groups with items in the dataset
The number medium number of items for each group (for all groups and for the groups with items)
The average groups correlation


'''
def compute_group_statistics(groups,dataset,compute_corr = False):


    intersection = [group_intersection(g,dataset) for g in groups]

    items_per_group = []
    groups_with_items = 0
    unique_items = set()
    unique_groups = set()
    for group in intersection:
        str_group = str(sorted(group))
        unique_groups.add(str_group)

        items_per_group.append(len(group))
        if items_per_group[-1] > 0:
            groups_with_items += 1
        for item in group:
            unique_items.add(item)

    unique_users = set()
    for g in groups:
        for u in g:
            unique_users.add(u)

    avg_corr_total = 0.0

    if compute_corr:
        for g in groups:
            avg_corr_g = 0
            i = 0

            for u1,u2 in combinations(g,2):
                #x = calc_similarity(u1,u2,dataset)[0]
                x = calc_similarity(dataset[dataset.user_id == u1],dataset[dataset.user_id == u2])[0]
                avg_corr_g += x
                i += 1

            avg_corr_total += avg_corr_g/i
            #print(avg_corr_total)
        avg_corr_total /= len(groups)

    out = 'Unique users: {0}\nUnique items: {1}\n'
    out += 'Unique groups {2}\nGroups with items: {3}\n'
    out +='Avg number of group items (total): {4}\n'
    out+='Avg number of group items (valid groups): {5}\nAvg. Group Correlation: {6}'

    #print(out.format(len(unique_users),len(unique_items),len(unique_groups),groups_with_items,sum(items_per_group)/len(groups),sum(items_per_group)/groups_with_items,avg_corr_total))
    return(out.format(len(unique_users),len(unique_items),len(unique_groups),groups_with_items,sum(items_per_group)/len(groups),sum(items_per_group)/groups_with_items,avg_corr_total))




    #print("Size:{0} Thresh:{1} Num groups with elems in test: {2}\n".format(args.group_size,args.t_value, str(groups_with_test)))
    #print("Average correlation {0}\n".format(avg_corr_total))


'''
Input: This function receives as parameter a set of groups and the dataframe used to
construct the groups.

Output: The set of items that should be put in the test set for each user
         according to the groups
'''
def make_test_fold_from_groups(groups, initial_dataframe,grs_checkins=None,
                                perc_test = 0.2, share_items = True):

    global args
    users_test = {}
    groups_with_test = 0
    groups_with_items = 0

    test_indexes = np.zeros(len(initial_dataframe),dtype=int)

    for group_id, group in enumerate(groups):          
        print(group_id)

        if type(groups) == dict:
            group_key = group
            group = groups[group_key]

            #this indicates that the itens shared by the group are in a specific
            #dataframe    
            if not grs_checkins is None:
                items_intersec = grs_checkins[
                                    grs_checkins.user_id == 
                                    group_key]['item_id'].values
            else:
                items_intersec = list(group_intersection(group,initial_dataframe))

        else:
            items_intersec = list(group_intersection(group,initial_dataframe))


        size_test = int(perc_test * len(items_intersec))
        if len(items_intersec) > 0:
            groups_with_items += 1

        '''for user in group:
           if not user in users_test:
               users_test.update([(user,set())])

        #check which items in the group intersection are already placed
        #in the test partition for the group users
        items_already_in_test = users_test[group[0]]
        for user in group[1:]:
            items_already_in_test = items_already_in_test.intersection(users_test[user])'''

        num_items_to_choose = size_test #- len(items_already_in_test)
        #it could happen that the couting of groups_with_test was lower than
        #the number of groups when for a especific group the pairs (user,items)
        #in the intersection are already in the test. This happen bacause a user
        #can belongs to more than one group
        if num_items_to_choose >= 1:
            groups_with_test += 1



        #verify wich items in the intersection are already in the test_indexes
        #we can use this strategy to minimize the impact of removing items that
        #belongs to other groups training set
        #In other words, we will verify if there are shared items in the test 
        #amongst the groups and take advantage from this

        items_already_in_test = []
        if args.share_items:
            for item_to_verify in items_intersec:
                for user in group:
                    #if user == 14726:
                    #    ipdb.set_trace()
                    user_item_index = initial_dataframe[
                                        (initial_dataframe.user_id == user) & 
                                        (initial_dataframe.item_id == item_to_verify)
                                        ].index
                    if test_indexes[user_item_index] == 1 and \
                            not item_to_verify in items_already_in_test:

                        items_already_in_test.append(item_to_verify)

        #ipdb.set_trace()
        #If there are a sufficient number of test items already marked as test
        #by other groups, we do not need to worry adding more items in test to
        #this specific group
        #if len(items_already_in_test) < num_items_to_choose:

        num_items_to_choose -= len(items_already_in_test)
        [items_intersec.remove(item_del) for item_del in items_already_in_test]
            
        #choose the items to be inserted in test
        #receives items_already_in_test to set the flag of all the group users
        #indicating that this item is in the test
        new_items = items_already_in_test
        while num_items_to_choose > 0:
            new_item = items_intersec[random.randint(0,len(items_intersec)-1)]
            if not new_item in new_items:
                new_items.append(new_item)
                num_items_to_choose -= 1


        #insert the items in the final structure
        #ipdb.set_trace()
        for item_to_insert in new_items:
            for user in group:
                user_item_index = initial_dataframe[(initial_dataframe.user_id == user) & (initial_dataframe.item_id == item_to_insert)].index
                test_indexes[user_item_index] = 1


    #ipdb.set_trace()
    df = initial_dataframe.assign(test=pd.Series(test_indexes).values)
    #ipdb.set_trace()    
    remove_not_in_train(df)

    return df

    #return users_test

    #l = len([users_test[x] for x n users_test if len(users_test[x]) > 0])
    #print("{0} groups with test {1} groups with item {2} users with test {3} total users".format(groups_with_test,groups_with_items,l,len(users_test)))
    #return users_test



def remove_not_in_train(dataframe):
    
    unique_items_train = dataframe[dataframe.test == 0].item_id.unique()
    unique_items_test = dataframe[dataframe.test == 1].item_id.unique()
    items = set(unique_items_test) - set(unique_items_train)
    print(items)
    #ipdb.set_trace()
    for item in items:
        dataframe.loc[dataframe.item_id == item,'test'] = 0
        


'''
Imput:
groups:dictionary containing the groups
dataset: pandas Dataframe containg all the data used to construc the groups
perc_test: percentage of group items (or user items case by_group = False)
to be used as test
by_group: defines if the partitioning will be performed considering the groups
or the users (not used in this version)
use_valid : defines if the partitioning will consider a
'''
def construct_partitions_from_groups(groups,dataset,out_dir,grs_checkins=None,
        perc_test=0.2, by_group=True, num_folds = 1, use_valid=True):

    if not dataset.__class__ == pd.DataFrame:
        dataset = read_ratings_file(dataset)

    out_cols = ("user_id", "item_id", "rating")

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    #ipdb.set_trace()
    for part in range(1,num_folds+1):

        print('Part'+str(part))
        if 'test' in dataset.columns:
            del(dataset['test'])

        #ipdb.set_trace()
        if not grs_checkins is None:
            dataset = make_test_fold_from_groups(groups,dataset,grs_checkins,
                                                perc_test)
        else:
            dataset = make_test_fold_from_groups(groups,dataset,perc_test)
        test_dataset = dataset[dataset.test == 1]

        test_file = os.path.join(out_dir,'u{}.test'.format(part))
        test_dataset.to_csv(test_file, header=False, index=False, sep="\t", columns=out_cols)

        #ipdb.set_trace()
        base_and_val = dataset[dataset.test == 0]


        #we use these if statments in order to control the construction of the
        #reeval folder and the validation files


        reeval_dir = out_dir
        if use_valid:
            reeval_dir = os.path.join(out_dir,'reeval')
            if not os.path.isdir(reeval_dir):
                os.mkdir(reeval_dir)

        #construct the files in the reeval dataset
        baseval_file = os.path.join(reeval_dir,'u{}.base'.format(part))
        base_and_val.to_csv(baseval_file, header=False, index=False, sep="\t", columns=out_cols)
        os.system('cp {} {}'.format(test_file,reeval_dir))

        if use_valid:
            #construct train/validation/test in the main file
            del(base_and_val['test'])
            #reconstruct the indexes in order to associate the labels in the
            #make_test_fold_from_groups function
            base_and_val.reset_index(drop=True,inplace=True)
            perc_vali = perc_test / (1-perc_test)
            #ipdb.set_trace()


            if not grs_checkins is None:
                base_and_val = make_test_fold_from_groups(groups,base_and_val,
                                        grs_checkins=grs_checkins,
                                        perc_test=perc_vali)
            else:
                base_and_val = make_test_fold_from_groups(groups,base_and_val, 
                                                        perc_test = perc_vali)


            base_file = os.path.join(out_dir,'u{}.base'.format(part))
            val_file = os.path.join(out_dir,'u{}.validation'.format(part))

            base_and_val[base_and_val.test == 1].to_csv(val_file, header=False, index=False, sep="\t", columns=out_cols)
            base_and_val[base_and_val.test == 0].to_csv(base_file, header=False, index=False, sep="\t", columns=out_cols)


'''
Return groups of similar users
'''
def similarity_groups(initial_dataset,
        similarity_function=pearsonr,
        group_size=4,
        num_groups=3000,
        threshold=0.27,
        num_processes = 1):

    num_users = len(initial_dataset.user_id.unique())
    #I use the +1 to do not need to alter the index dor the users
    #it just works for ML datasets
    users_similarity_matrix = np.zeros((num_users+1,num_users+1))
    users_similarity_matrix.fill(-999)
    #print("Starting computing distances")
    #for usr1,usr2 in combinations(initial_dataset.user_id.unique(),2):
    #    users_similarity_matrix[usr1-1][usr2-1] = similarity_function(usr1,usr2,initial_dataset)
    #print("Finish computing distances")
    #manager = Manager()
    final_groups = []#[[] for _ in range(num_groups)]
    constructed_groups = 0
    #Constructing groups with high innner similarity
    #only considers the similarity with
    print("init")


    init_groups = time.time()
    while constructed_groups < num_groups:

        if constructed_groups%50 == 0:
            print("{0} - {1}".format(constructed_groups,(time.time()-init_groups)))
            init_groups = time.time()

        group_users = construct_group(initial_dataset, users_similarity_matrix,
                    similarity_function,
                    group_size, threshold)

        final_groups.append(group_users)

        #if (constructed_groups % 50) == 0:
        #    print(constructed_groups)

        constructed_groups += 1

    return final_groups





def contruct_group_recursive(g,users,intersec,group_size,all_intersections,attempts=0):


    if len(g) == 0:
        g = [users[random.randint(0,len(users)-1)]]
        intersec = all_intersections[g[0]]
        return contruct_group_recursive(g,users,intersec,group_size,all_intersections,attempts)
    elif len(g) == group_size:
        return g
    elif attempts == 10:
        g = []
        return [] #contruct_group_recursive(g,users,intersec,group_size,all_intersections,0)
    elif len(intersec) < (group_size-len(g)):
        attempts += 1
        g = g[:-1]
        intersec = list(group_intersection(g,all_intersections))
        return contruct_group_recursive(g,users,intersec,group_size,all_intersections,attempts)
    else:
        g.append(intersec[random.randint(0,len(intersec)-1)])
        intersec = list(group_intersection(g,all_intersections))
        return contruct_group_recursive(g,users,intersec,group_size,all_intersections,attempts)



def construct_adjacency_dict(users_similarity_matrix,pos_user_map,threshold):
    '''
    users_similarity_matrix: user x user similarity matrix
    pos_user_map: A dict mapping a position in the users_similarity_matrix to a
    user_id

    Returns a dict which key = user_id and value = neighboors user_id
    '''

    positions_bigger_than_t = {}
    for i in range(1,len(users_similarity_matrix)):
        u1_id = pos_user_map[i]
        positions_bigger_than_t[u1_id] = []
        for j in range(1,len(users_similarity_matrix)):
            if users_similarity_matrix[i][j] >= threshold or users_similarity_matrix[j][i] >= threshold:
                u2_id = pos_user_map[j]
                positions_bigger_than_t[u1_id].append(u2_id)

    return positions_bigger_than_t

'''
Return groups of similar users
'''
def similarity_groups_strong_fast(initial_dataset,
        similarity_function=pearsonr,
        group_size=4,
        num_groups=300,
        threshold=0.27):

    global args

    users = initial_dataset.user_id.unique()
    num_users = len(users)
    users_similarity_matrix = None
    if args.dist_file:
        users_similarity_matrix = np.load(args.dist_file)
        #TODO adicionar calculo de lista de adjacencia

    #positions_bigger_than_t = []
    user_pos_map,pos_user_map = utils.construct_mapping(initial_dataset)
    positions_bigger_than_t  = construct_adjacency_dict(users_similarity_matrix,pos_user_map,threshold)
    '''for i in range(len(users_similarity_matrix)):
        positions_bigger_than_t.append([])
        for j in range(len(users_similarity_matrix)):
            if users_similarity_matrix[i][j] >= threshold or users_similarity_matrix[j][i] >= threshold:
                positions_bigger_than_t[-1].append(j)
    '''


    groups = []
    groups_similarity = []
    constructed_groups = 0
    #Constructing groups with high innner similarity
    #only considers the similarity with
    group_users = []
    unique_groups = {}
    while constructed_groups < num_groups:
        #ipdb.set_trace()
        if constructed_groups%50 == 0 and constructed_groups > 0:
            print(constructed_groups)

        group_users = contruct_group_recursive([],users,[],group_size,positions_bigger_than_t)


        if len(group_users) > 0:
            sorted_users = sorted(group_users)
            if not str(sorted_users) in unique_groups:
                if len(group_intersection(group_users, initial_dataset)) >= args.min_group_items:
                    groups.append(group_users)
                    #i'm using a dict where the key is the sorted str of the group 
                    #members and None as value
                    unique_groups[str(sorted_users)] = None
                    constructed_groups += 1

    return groups

'''
Return groups of similar users
'''
def similarity_groups_strong(initial_dataset,
        similarity_function=pearsonr,
        group_size=4,
        num_groups=300,
        threshold=0.27):

    global args

    users = initial_dataset.user_id.unique()
    num_users = len(users)


    groups = []
    groups_similarity = []
    constructed_groups = 0
    #only considers the similarity with
    #Constructing groups with high innner similarity
    print("init")
    while constructed_groups < num_groups:

        #if (constructed_groups % 50) == 0:
        print(constructed_groups)

        first_user = users[random.randint(0,num_users-1)]
        first_user_DF = initial_dataset[initial_dataset.user_id == first_user]
        group_users = [first_user]
        while len(group_users) < group_size:

            usr2 = users[random.randint(0,num_users-1)]
            usr2_DF = initial_dataset[initial_dataset.user_id == usr2]

            num_users_respect_thresh = sum([1 for usr in group_users if calc_similarity(first_user_DF,usr2_DF,simi_func=similarity_function)[0] >= threshold])

            attempts = 1
            while num_users_respect_thresh < len(group_users) or usr2 in group_users:
                #no caso de rodar muitas vezes e nao encontrar outro user similar ao primeiro
                #escolho outro usuario como semente
                if attempts > 20:
                    first_user = users[random.randint(0,num_users-1)]
                    first_user_DF = initial_dataset[initial_dataset.user_id == first_user]
                    group_users = [first_user]
                    attempts = 0

                usr2 = users[random.randint(0,num_users-1)]
                usr2_DF = initial_dataset[initial_dataset.user_id == usr2]
                num_users_respect_thresh = sum([1 for usr in group_users if calc_similarity(first_user_DF,usr2_DF,simi_func=similarity_function)[0] >= threshold])

                attempts += 1

            #print("Yes")
            group_users.append(usr2)



        groups.append(group_users)

        constructed_groups += 1

    return groups




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



def group_union(group,test_f):
    initial = set(list((test_f[test_f.user_id == group[0]]['item_id'])))
    for user in group[1:]:
        initial = initial.union(list((test_f[test_f.user_id == user]['item_id'])))
    return initial


def len_intersect(usr1_ratings,usr2_ratings):
    if usr1_ratings.__class__ == pd.DataFrame:
        initial = set(list(usr1_ratings['item_id']))
        initial = initial.intersection(list((ussr2_ratings['item_id'])))
        return len(initial)
    res = len(set(usr1_ratings).intersection(usr2_ratings))
    return res



def construct_graph_from_matrix(mat,threshold=0.27):
    edges = []
    for i in range(len(mat)):
        for j in range(i+1,len(mat)):
            if mat[i][j] >= threshold:
                edges.append([i,j])
    G = nx.from_edgelist(edges)
    return G


"""Verify if all the users in a group are present in the dataset used to train the recommendation
algorithms."""
def group_in_train(group,initial_dataframe):
    for user in group:
        if len(initial_dataframe[initial_dataframe.user_id == user]) == 0:
            return False
    return True


def filter_groups(groups,df_grs_checkins,initial_dataframe,min_intersection=10):
    """this function removes the groups whose have less than 10 checkins in
    unique venues or that are not present in the initial_dataset, which
    will be used to train the baseline recommendation algorithms"""
    ipdb.set_trace()
    #remove checkins in the same venue
    df_grs_checkins.drop_duplicates(subset=['user_id','item_id'],inplace=True)
    df_grs_checkins.reset_index(drop=True,inplace=True)


    #remove the group checkins that do not occur in the initial dataset
    #remembering that the groups where constructed in the full dataset and 
    #we, initially, we do not know if the dataset in this stage has overcomed some
    #pre processing
    indexes_to_remove = np.zeros(len(df_grs_checkins))

    for group_id in groups:
        #print(group_id)
        grp_intersect_init_df = utils.group_intersection(groups[group_id],initial_dataframe)
        items_group_checkins = df_grs_checkins[df_grs_checkins.user_id == group_id]['item_id'].values

        for item in items_group_checkins:
            if not item in grp_intersect_init_df:
                grp_item_idx = df_grs_checkins[(df_grs_checkins.user_id == group_id) &
                                              (df_grs_checkins.item_id == item)].index

                indexes_to_remove[grp_item_idx] = 1
    
    #ipdb.set_trace()
    df_grs_checkins = df_grs_checkins.assign(to_remove=pd.Series(indexes_to_remove).values)
    df_grs_checkins = df_grs_checkins[df_grs_checkins.to_remove == 0]
    #df_grs_checkins.drop(indexes_to_remove,inplace=True)
    df_grs_checkins.reset_index(drop=True,inplace=True)


    #count groups checkins
    grs_checkins_counts = df_grs_checkins.groupby(by=['user_id']).apply(len)

    #create a dataframe to merge with the original groups checkin
    ipdb.set_trace()
    grs_checkins_counts = pd.DataFrame({'user_id': list(range(len(grs_checkins_counts))),
                    'num_checkins' : grs_checkins_counts.values})
    
    #merge the tow dataframes
    df_grs_checkins = df_grs_checkins.merge(right=grs_checkins_counts, 
                        on='user_id',how='left')
    
    #grs_checkins_counts_to_remove = grs_checkins_counts[grs_checkins_counts.num_checkins < 10]
    #ipdb.set_trace()
    #TODO prosseguir daqui
    #[groups.pop(gid) for gid in grs_checkins_counts_to_remove.user_id.unique()]
    df_grs_checkins_to_maintain = df_grs_checkins[df_grs_checkins.num_checkins >= args.min_group_items].sort_values(by='user_id')

    group_ids_to_remove = []
    for group_id in df_grs_checkins_to_maintain.user_id.unique():
        if not group_in_train(groups[group_id],initial_dataframe):
            print(group_id)
            #ipdb.set_trace()
            indexes_to_remove = df_grs_checkins_to_maintain[df_grs_checkins_to_maintain.user_id == group_id].index
            df_grs_checkins_to_maintain.drop(indexes_to_remove,inplace=True)
            group_ids_to_remove.append(group_id)

    #for grp in group_ids_to_remove:
    #    groups.pop(grp)




    #construct indexes to the new groups
    new_indexes = pd.DataFrame(
                    {'user_id':df_grs_checkins_to_maintain.user_id.unique(),
                    'new_id':list(range(
                    len(df_grs_checkins_to_maintain.user_id.unique())))})

    #ipdb.set_trace()
    
    df_grs_checkins_to_maintain = df_grs_checkins_to_maintain.merge(
                                    right=new_indexes,on='user_id',how='left')


    new_groups = {new_id:groups[old_id] for new_id,old_id in zip(
                                           new_indexes.new_id.unique(),
                                           new_indexes.user_id.unique())}

    df_grs_checkins_to_maintain['user_id'] =  df_grs_checkins_to_maintain['new_id']
    del(df_grs_checkins_to_maintain['new_id'])

    #ipdb.set_trace()
    return new_groups,df_grs_checkins_to_maintain
    
    

def run(dataset,test_dataset,num_groups,group_size,threshold,groups_file="",
        num_processes=1,strong=False,similarity_function=pearsonr):

    if type(dataset) == str:
        dataset = read_ratings_file(dataset)

    '''if test_dataset:
        test_dataset = read_ratings_file(test_dataset)'''
    #for group_size in [3,4,5]:
    #    for thresh in [0.1,0.3]:
    #print(thresh)

    log_file = open(groups_file+".log",'w')

    if strong:
       compute_similarity = similarity_groups_strong_fast
    elif args.rwm:
       compute_similarity = random_groups_with_minimum
    else:
       compute_similarity = similarity_groups



    groups = compute_similarity(dataset,num_groups=num_groups,
             group_size=group_size,threshold=threshold,
             similarity_function=similarity_function)
    #ipdb.set_trace()
    '''if test_dataset:
        intersect_in_test = [group_intersection(g,test_dataset) for g in groups]
        groups_with_test = sum([1 for x in intersect_in_test if len(x) > 0])
        log_file.write("Size:{0} Thresh:{1} Num groups with elems in test: {2}\n".format(args.group_size,args.t_value, str(groups_with_test)))'''

    intersect_in_base = [group_intersection(g,dataset) for g in groups]
    groups_with_base = sum([1 for x in intersect_in_base if len(x) > 0])

    log_file.write(compute_group_statistics(groups,dataset))



    log_file.close()
    return groups



def arg_parse():
    p = argparse.ArgumentParser()

    p.add_argument('--base',type=str,
        help="Folder containing the training instances")

    p.add_argument('--seed',type=int, default=123456,
        help="Seed used in the experiments")

    p.add_argument('--rwm',action='store_true',
        help='If set uses the function random_groups_with_minimum')

    p.add_argument('--dist_file',type=str,default='',
        help='Numpy dataset contaning the distances previously computed')

    p.add_argument('--simi_func',default=PCC,
        help='Similarity function')

    p.add_argument('--num_parts',type=int, default=5,
        help="NUmber of random partitions to be created")

    p.add_argument('--min_group_items',type=int,default=5,
        help="Defines the minimum number of items a group should have in common.")

    p.add_argument('--share_items',action='store_true',
        help="If this argument was set the construction of the test fold will " \
            "verify if any of the group items are already in the test set."
             "If it was true, those items are 'shared' by the groups in the test set")

    p.add_argument('--before',action='store_true',
        help="If set the groups construction is perfomed using the dataset before the folds partitioning")
    p.add_argument('-o','--out_dir',nargs='?')
    p.add_argument('-p','--part',type=str, default='u1',
        help="Partition to be used")
    p.add_argument('--num_groups',type=int, default=3000,
        help='NUmber of groups')
    p.add_argument('--group_size', type=int, default=5,
        help='Size of the groups')
    p.add_argument('--t_value', type=float,default=0.27,
        help='Threshold value used to construct the groups')
    p.add_argument('-n','--num_proc', type=int,default=1,
        help='Number of threads')
    p.add_argument('--strong',action='store_true',
        help='This option enables the computation of "strong" groups. i.e. for an user be inserted in this group it has to be similar to all group members')

    p.add_argument('--no_grs_filter',action='store_true',
        help='When set the groups will not be filtered whith respect to size')

    p.add_argument('--groups_file',type=str,default='',
        help = 'When used this parameter indicates that the groups are already constructed.' \
                'the parameter sets the path to the groups_file')

    p.add_argument('--gw_chk',type=str,
        help='File containing the gowalla groups checkins. This file will '\
        'be used both to filter the groups and to construct the folds')

    #p.add_argument('-f','--func',type=str, default = pearsonr
    #    help='Similarity function to be used when computing users correlarion. Options: pearsonr,cosine,jaccard')
    parsed = p.parse_args()

    if parsed.out_dir is None:
        parsed.out_dir = parsed.base

    return parsed


if __name__ == "__main__":

    global args
    args = arg_parse()

    random.seed(args.seed)

    if args.before:
        train = os.path.join(args.base,"u.proc.data")
        test = train
    else:
        train = os.path.join(args.base,args.part+".base")
        test = os.path.join(args.base,args.part+".test")

    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)
        #save_groups(groups,groups_file)
    

    train_df = read_ratings_file(train)
    
    if args.groups_file:
        groups = utils.read_groups(args.groups_file,return_dict=True)
        if not args.no_grs_filter:
            
            #I'm calling the group id as user id to use the functions in utils
            #whithout modification
            df_groups_checkins = pd.read_csv(args.gw_chk,header=None,
                names=['user_id','item_id','timestamp'],sep='\t')
            
            groups_filtered,df_groups_checkins_filtered = filter_groups(
                                                            groups,
                                                            df_groups_checkins,
                                                            train_df,
                                                            min_intersection=10)


            grp_base_name = os.path.basename(args.groups_file)
            save_groups(groups_filtered,os.path.join(args.out_dir,
                                                    grp_base_name+'_filtered'))
            #ipdb.set_trace()
            chk_filtered_file = os.path.join(args.out_dir,'groups_checkins_filtered.data')
            df_groups_checkins_filtered.to_csv(chk_filtered_file,
                                                header=None, index=False, 
                                                sep='\t')
                                                        
            construct_partitions_from_groups(groups_filtered,train,args.out_dir,
                grs_checkins = df_groups_checkins_filtered, perc_test=0.2,
                by_group=True, use_valid=True,num_folds=args.num_parts)

    else:
        if args.strong:
            groups_file = os.path.join(args.out_dir,"{0}.groups_ng{1}_sz{2}_thresh{3}_strong".format(args.part,args.num_groups,args.group_size,args.t_value))
        else:
            groups_file = os.path.join(args.out_dir,"{0}.groups_ng{1}_sz{2}_thresh{3}".format(args.part,args.num_groups,args.group_size,args.t_value))


        groups = run(train_df,test,args.num_groups,args.group_size,args.t_value,groups_file,args.num_proc,args.strong,similarity_function=args.simi_func)
        save_groups(groups,groups_file)

        construct_partitions_from_groups(groups,train,args.out_dir,perc_test=0.2,by_group=True, use_valid=True,num_folds=args.num_parts)
