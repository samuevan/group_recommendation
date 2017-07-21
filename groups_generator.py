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
        for group_id in range(len(groups)):
            group_str = ",".join([str(x) for x in groups[group_id]])
            out_f.write("{0}:{1}\n".format(group_id,group_str))





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
def calc_similarity(user_a,user_b,dataset,user_a_DF = None, min_intersec_size=5,simi_func=pearsonr):

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


#def calc_similarity(user_a,user_b,dataset,user_a_DF = None, min_intersec_size=5,simi_func=pearsonr):
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
    if (np.var(user_a_ratings) < 0.0001) or (np.var(user_b_ratings) < 0.0001) and (simi_func == pearsonr):
        corr = 1 - ((np.mean(user_a_ratings) - np.mean(user_b_ratings))**2) / (np.mean(user_a_ratings)*np.mean(user_b_ratings))
        #print(corr)
        return (corr,1.0)

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
def make_test_fold_from_groups(groups, initial_dataframe,perc_test = 0.2):

    users_test = {}
    groups_with_test = 0
    groups_with_items = 0

    test_indexes = np.zeros(len(initial_dataframe),dtype=int)

    for group_id, group in enumerate(groups):
        #ipdb.set_trace()
        items_intersec = list(group_intersection(group,initial_dataframe))
        size_test = int(0.2 * len(items_intersec))
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

        #choose the items to be inserted in test
        new_items = []
        while num_items_to_choose > 0:
            new_item = items_intersec[random.randint(0,len(items_intersec)-1)]
            if not new_item in new_items:
                new_items.append(new_item)
                num_items_to_choose -= 1


        #insert the items in the final structure
        for item_to_insert in new_items:
            for user in group:
                user_item_index = initial_dataframe[(initial_dataframe.user_id == user) & (initial_dataframe.item_id == item_to_insert)].index
                test_indexes[user_item_index] = 1



    return initial_dataframe.assign(test=pd.Series(test_indexes).values)

    #return users_test
                    
    #l = len([users_test[x] for x n users_test if len(users_test[x]) > 0])
    #print("{0} groups with test {1} groups with item {2} users with test {3} total users".format(groups_with_test,groups_with_items,l,len(users_test)))
    #return users_test



'''
Imput:
groups:dictionary containing the groups
dataset: pandas Dataframe containg all the data used to construc the groups
perc_test: percentage of group items (or user items case by_group = False) 
to be used as test
by_group: defines if the partitioning will be performed considering the groups 
or the users
use_valid : defines if the partitioning will consider a 
'''
def construct_partitions_from_groups(groups,dataset,out_dir,perc_test=0.2,by_group=True, use_valid=False):

    out_cols = ("user_id", "item_id", "rating")

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

        
    dataset = make_test_fold_from_groups(groups,dataset)
    test_dataset = dataset[dataset.test == 1]

    test_file = os.path.join(out_dir,'u.test')
    test_dataset.to_csv(test_file, header=False, index=False, sep="\t", columns=out_cols)

    base_and_val = dataset[dataset.test == 0]
    
    if use_valid:
        del(base_and_val['test'])
        perc_vali = perc_test / (1-perc_test)
        base_and_val = make_test_fold_from_groups(groups,dataset, perc_test = perc_vali)

        base_file = os.path.join(out_dir,'u.base')
        val_file = os.path.join(out_dir,'u.validation')     

        base_and_val[base_and_val.test == 1].to_csv(val_file, header=False, index=False, sep="\t", columns=out_cols)
        base_and_val[base_and_val.test == 0].to_csv(base_file, header=False, index=False, sep="\t", columns=out_cols)

    else:
        baseval_file = os.path.join(out_dir,'u.base')
        base_and_val.to_csv(baseval_file, header=False, index=False, sep="\t", columns=out_cols)

    

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

    positions_bigger_than_t = []
    for i in range(len(users_similarity_matrix)):
        positions_bigger_than_t.append([])
        for j in range(len(users_similarity_matrix)):
            if users_similarity_matrix[i][j] >= threshold or users_similarity_matrix[j][i] >= threshold:
                positions_bigger_than_t[-1].append(j)

    
    
    groups = []
    groups_similarity = []
    constructed_groups = 0
    #Constructing groups with high innner similarity
    #only considers the similarity with 
    group_users = []
    while constructed_groups < num_groups:
        #ipdb.set_trace()
        if constructed_groups%50 == 0:
            print(constructed_groups)

        group_users = contruct_group_recursive([],users,[],group_size,positions_bigger_than_t) 
        if len(group_users) > 0:
            groups.append(group_users)
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
    #Constructing groups with high innner similarity
    #only considers the similarity with 
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
    else:
        initial = set(test_f[group[0]])
        for user in group[1:]:
            initial = initial.intersection(test_f[user])
        return initial
        



def group_union(group,test_f):                                                         
    initial = set(list((test_f[test_f.user_id == group[0]]['item_id'])))
    for user in group[1:]:
        initial = initial.union(list((test_f[test_f.user_id == user]['item_id'])))
    return initial


def len_intersect(usr1_ratings,usr2_ratings):
    res = len(set(usr1_ratings).intersection(usr2_ratings))
    return res


def run(dataset,test_dataset,num_groups,group_size,threshold,groups_file="",
        num_processes=1,strong=False,similarity_function=pearsonr):

    
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

    p.add_argument('--before',action='store_true',
        help="If set the groups construction is perfomed using the dataset before the folds partitioning")
    p.add_argument('-o','--out_dir',nargs='?')
    p.add_argument('-p','--part',type=str, default='u1',
        help="Partition to be used")
    p.add_argument('--num_groups',type=int, default=3000,
        help='NUmber of groups')
    p.add_argument('--group_size', type=int, default=5,
        help='Size of the groups')
    p.add_argument('--t_value', type=float,default=0.3,
        help='Threshold value used to construct the groups')
    p.add_argument('-n','--num_proc', type=int,default=1,
        help='Number of threads')
    p.add_argument('--strong',action='store_true',
        help='This option enables the computation of "strong" groups. i.e. for an user be inserted in this group it has to be similar to all group members')
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

    if args.strong:
        groups_file = os.path.join(args.out_dir,"{0}.groups_ng{1}_sz{2}_thresh{3}_strong".format(args.part,args.num_groups,args.group_size,args.t_value))
    else:
        groups_file = os.path.join(args.out_dir,"{0}.groups_ng{1}_sz{2}_thresh{3}".format(args.part,args.num_groups,args.group_size,args.t_value))


    groups = run(train,test,args.num_groups,args.group_size,args.t_value,groups_file,args.num_proc,args.strong,similarity_function=args.simi_func)
    save_groups(groups,groups_file)


    

