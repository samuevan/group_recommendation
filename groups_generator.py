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
import matplotlib.pyplot as plt
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
       


def save_groups(groups,out_file):

    print("Saving group file")
    with open(out_file,'w') as out_f:
        for group_id in range(len(groups)):
            group_str = ",".join([str(x) for x in groups[group_id]])
            out_f.write("{0}:{1}\n".format(group_id,group_str))



def calc_similarity(user_a,user_b,dataset,user_a_DF = None, min_intersec_size=5,simi_func=pearsonr):

    #we can pass the DataFrame for the first user directly
    if user_a_DF.__class__ == pd.DataFrame:
        items_intersec = list(set(user_a_DF['item_id']) & 
                set(dataset[dataset.user_id == user_b]['item_id']))
    else:    
        items_intersec = list(set(dataset[dataset.user_id == user_a]['item_id']) & 
                set(dataset[dataset.user_id == user_b]['item_id']))

    if len(items_intersec) < min_intersec_size:
        return (-1.0,1.0)
    

    if user_a_DF.__class__ == pd.DataFrame:
        user_a_ratings = [user_a_DF[user_a_DF.item_id==item_id]['rating'].get_values()[0] for item_id in items_intersec]

    else:
        user_a_ratings = [dataset[(dataset.user_id == user_a) & 
                    (dataset.item_id==item_id)]['rating'].get_values()[0] for item_id in items_intersec]


    user_b_ratings = [dataset[(dataset.user_id == user_b) & 
                    (dataset.item_id==item_id)]['rating'].get_values()[0] for item_id in items_intersec]


    #trata dos casos onde não existe variabilidade nos ratings dados por um 
    #usuario e a funcao de similaridade eh pearson. 
    #Nao sei se eh a melhor forma de fazer isso
    if (np.var(user_a_ratings) < 0.0001) or (np.var(user_b_ratings) < 0.0001) and (simi_func == pearsonr):
        corr = 1 - ((np.mean(user_a_ratings) - np.mean(user_b_ratings))**2) / (np.mean(user_a_ratings)*np.mean(user_b_ratings))
        #print(corr)
        return (corr,1.0)

    #ipdb.set_trace()    
    simi = simi_func(user_a_ratings,user_b_ratings)
    #print(simi)

    return simi



def jaccard(user_a_ratings,user_b_ratings):

    num = len(set(user_a_ratings).intersection(user_b_ratings))
    den = len(set(user_a_ratings).union(user_b_ratings))

    jacc = float(num)/float(den)

    return (jacc,1.0)    


'''
Return groups of similar users
'''
def similarity_groups_strong(initial_dataset, out_file, 
        similarity_function=pearsonr,group_size=4, num_groups=300,threshold=0.27):

    users = initial_dataset.user_id.unique()
    num_users = len(users)
    users_similarity_matrix = np.zeros((num_users,num_users))
    #users_similarity_matrix.fill(-1.0)
    #print("Starting computing distances")
    #for usr1,usr2 in combinations(initial_dataset.user_id.unique(),2):
    #    users_similarity_matrix[usr1-1][usr2-1] = similarity_function(usr1,usr2,initial_dataset)
    #print("Finish computing distances")

    groups = []
    groups_similarity = []
    constructed_groups = 0
    #Constructing groups with high innner similarity
    #only considers the similarity with 
    print("init")    
    while constructed_groups < num_groups:

        if (constructed_groups % 50) == 0:
            print(constructed_groups)

        first_user = users[random.randint(0,num_users-1)]
        group_users = [first_user]
        while len(group_users) < group_size:
            usr2 = users[random.randint(0,num_users-1)]

            num_users_respect_thresh = sum([1 for usr in group_users if calc_similarity(usr,usr2,initial_dataset,simi_func=similarity_function)[0] > threshold])

            #simi,p_value = calc_similarity(first_user,usr2,initial_dataset,simi_func=similarity_function)
            attempts = 1
            while num_users_respect_thresh < len(group_users) or usr2 in group_users:
                #no caso de rodar muitas vezes e nao encontrar outro user similar ao primeiro
                #escolho outro usuario como semente
                if attempts > 200:
                    first_user = users[random.randint(0,num_users-1)]
                    group_users = [first_user]
                    attempts = 0

                usr2 = users[random.randint(0,num_users-1)]
                num_users_respect_thresh = sum([1 for usr in group_users if calc_similarity(usr,usr2,initial_dataset,simi_func=similarity_function)[0] > threshold])
                attempts += 1
           
            #print("Yes")
            group_users.append(usr2)
            


        groups.append(group_users)
        
        constructed_groups += 1

    return groups






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
            
        #every time we draw a new pair of users which similarity/correlation 
        #was not yet computed we do it
        #if users_similarity_matrix[first_user][usr2] == -999:
        simi,p_value = calc_similarity(first_user,usr2,initial_dataset,
                                user_a_DF=first_user_DF, 
                                simi_func=similarity_function)

        #    users_similarity_matrix[first_user][usr2] = simi
        #else:
        #    simi = users_similarity_matrix[first_user][usr2]

        attempts = 0
        while simi < threshold or usr2 == first_user:
            
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
                
            simi,p_value = calc_similarity(first_user,usr2,
                                initial_dataset, user_a_DF=first_user_DF, 
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
Return groups of similar users
'''
def similarity_groups(initial_dataset, out_file, 
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


def group_intersection(group,test_f):                                                         
    initial = set(list((test_f[test_f.user_id == group[0]]['item_id'])))
    for user in group[1:]:
        initial = initial.intersection(list((test_f[test_f.user_id == user]['item_id'])))
    return initial



def group_union(group,test_f):                                                         
    initial = set(list((test_f[test_f.user_id == group[0]]['item_id'])))
    for user in group[1:]:
        initial = initial.union(list((test_f[test_f.user_id == user]['item_id'])))
    return initial





#TODO reimplementar com pandas
def run(dataset,test_dataset,num_groups,group_size,threshold,groups_file="",
        num_processes=1,strong=False,similarity_function=pearsonr):

    
    initial_dataset = read_ratings_file(dataset)
    test_dataset = read_ratings_file(test_dataset)
    #for group_size in [3,4,5]:
    #    for thresh in [0.1,0.3]:
    #print(thresh)

    log_file = open(groups_file+".log",'w')
        
    if strong:
       compute_similarity = similarity_groups_strong
    else:
       compute_similarity = similarity_groups

    groups = compute_similarity(initial_dataset,'asdf',num_groups=num_groups,
             group_size=group_size,threshold=threshold,
             similarity_function=similarity_function,num_processes=num_processes)
    #ipdb.set_trace()
    intersect_in_test = [group_intersection(g,test_dataset) for g in groups]
    intersect_in_base = [group_intersection(g,initial_dataset) for g in groups]
    groups_with_base = sum([1 for x in intersect_in_base if len(x) > 0])
    groups_with_test = sum([1 for x in intersect_in_test if len(x) > 0])

    log_file.write("Size:{0} Thresh:{1} Num groups with elems in base: {2}\n".format(args.group_size,args.t_value, str(groups_with_base)))
    log_file.write("Size:{0} Thresh:{1} Num groups with elems in test: {2}\n".format(args.group_size,args.t_value, str(groups_with_test)))

    avg_corr_total = 0.0
    for g in groups:
        avg_corr_g = 0
        i = 0
        for u1,u2 in combinations(g,2):
            x = calc_similarity(u1,u2,initial_dataset)[0]
            #print(x)
            #ipdb.set_trace()
            #print("{0} {1} {2}".format(u1,u2,x))                    
            avg_corr_g += x
            i += 1
                
        avg_corr_total += avg_corr_g/i
        #print(avg_corr_total)
    avg_corr_total /= len(groups)

    log_file.write("Average correlation {0}\n".format(avg_corr_total))
            

    #print("Min size : " + str(min([len(g) for g in intersect_in_test])))
    log_file.close()
    return groups

    #random_groups(initial_dataset,out_file)        


def arg_parse():
    p = argparse.ArgumentParser()
    
    p.add_argument('--base',type=str,
        help="Folder containing the training instances")
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
    
    args = arg_parse()

    train = os.path.join(args.base,args.part+".base")
    test = os.path.join(args.base,args.part+".test")

    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    if args.strong:
        groups_file = os.path.join(args.out_dir,"{0}.groups_ng{1}_sz{2}_tresh{3}_strong".format(args.part,args.num_groups,args.group_size,args.t_value))
    else:
        groups_file = os.path.join(args.out_dir,"{0}.groups_ng{1}_sz{2}_tresh{3}".format(args.part,args.num_groups,args.group_size,args.t_value))


    groups = run(train,test,args.num_groups,args.group_size,args.t_value,groups_file,args.num_proc,args.strong)
    save_groups(groups,groups_file)


    

