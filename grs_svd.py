import os
import utils
import glob
import argparse
import pandas as pd
import numpy as np
import ipdb
from fancyimpute import NuclearNormMinimization as NNM
from fancyimpute import SoftImpute as SI, MatrixFactorization as MF, IterativeSVD as ISVD, BiScaler as BS, MICE
import grs_recommendation_agg as grs_rec
from sklearn.decomposition import NMF
from collections import defaultdict



def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument('--base_dir',type=str,required=True,
        help = 'folder containg the datasets')
    p.add_argument('-p','--part',type=str,default='u1',
        help='Partition to be used')
    p.add_argument('--base_recs',nargs='*',
        help='Indicate the recommenders containing the recommendations used ' \
        'to construct the group recommendation. When none recommender is set ' \
        'all the recommenders in the base_dir will be used')
    p.add_argument('--base_rec',type=str,
        help='The algorithm currently used in the group recommendation.'
             'Usually is one of the algorithms in args.base_recs')
    p.add_argument('--groups_file',type=str,required=True,
        help='The file containing the groups of users to whom the recommendation will be made')
    p.add_argument('--test_file', type=str,
        help='The file containing the test file. It will be used to construct the oraculus recommendation')
    p.add_argument('--train_file',type=str,
        help='The file containing the training file used by the base recommender')
    p.add_argument('--val_file',type=str,
        help='The file containing the validation file used by the base recommender')
    p.add_argument('-o','--out_dir',type=str,default='',
        help='Output folder. The group recommendations will be saved in this foder')
    p.add_argument('--i2use',type=int,default=20,
        help='Size of the input rankings')
    p.add_argument('--i2sug',type=int,default=10,
        help='Size of the outrput rankings')
    p.add_argument('--save_dataset',action='store_true',
        help="When set the the user_item matrix constructed by the goups' users "\
             "will be saved as a letor dataset")

    parsed = p.parse_args()
    #ipdb.set_trace()
    if parsed.out_dir == '':
        parsed.out_dir = parsed.base_dir

    #base_rec_path = parsed.part + "-" + parsed.base_rec + ".out"
    #parsed.base_rec = os.path.join(parsed.base_dir,base_rec_path)

    parsed.test_file = os.path.join(parsed.base_dir,parsed.part+'.test')
    parsed.train_file = os.path.join(parsed.base_dir,parsed.part+'.base')

    if not 'reeval' in parsed.base_dir:
        parsed.val_file = os.path.join(parsed.base_dir,parsed.part+'.validation')

    #we use the same groups for all partitions
    #parsed.groups_file = parsed.groups_file.replace('u1',parsed.part)
     
    if not parsed.base_recs:
        recs_in_dir = set().union([rec[3:-4] for rec in sorted(glob.glob(os.path.join(parsed.base_dir,"*.out")))])
        parsed.base_recs = list(recs_in_dir)    
	        


    return parsed





'''
Receives a ranking which elements are composed by tuples (item,score)
and returns a rankings solely with the items
'''
def remove_scores(ranking):
    just_items = [item for item,score in ranking]
    return just_items



#TODO implement
def construct_item_item_matrix(group,users_rankings,with_ones=False,with_scores=False):


    group_rankings = [recomm_rank for user in group for recomm_rank in users_rankings[user]]

    all_items = sorted([(x,i) for i,x in enumerate(utils.rank_union(group_rankings))])
    all_items = defaultdict(int,all_items)

    recomm_matrix = np.empty((len(all_items),len(all_items)))
    recomm_matrix[:] = np.nan



'''
Constroi uma matrix user x item
cada linha representa as recomendacoes feitas para um usuario
o parametro with_one faz com que a matriz seja preenchida por 1s ao inves da posicao do item no ranking
#TODO tambem inserir na matrix os ratings do usuario que estao na base de treino
'''


def construct_user_item_matrix(group,users_rankings,with_ones=False,with_scores=False):
    
    group_rankings = [recomm_rank for user in group for recomm_rank in users_rankings[user]]

        

    all_items = sorted([(x,i) for i,x in enumerate(utils.rank_union(group_rankings))])
    all_items = defaultdict(int,all_items)

    recomm_matrix = np.empty((len(group),len(all_items)))
    recomm_matrix[:] = np.nan

    for useri,user in enumerate(group):
        for ranki,user_rank in enumerate(users_rankings[user]):
            for item_pos_rank,(item,score) in enumerate(user_rank):
                item_pos = all_items[item]
                if with_ones:
                    recomm_matrix[useri,item_pos] = 1
                elif with_scores:
                    recomm_matrix[useri,item_pos] = score
                else:
                    recomm_matrix[useri,item_pos] = 1-(item_pos_rank/len(user_rank))

    return recomm_matrix,all_items



def complete_user_item_matrix(user_item_matrix,method):

    #nusers,nitems = user_item_matrix.shape
    if method == BS:
        complete_matrix = method().fit_transform(user_item__matrix)
    else:
        complete_matrix = method().complete(user_item_matrix)

    return complete_matrix


def calc_items_attributes(group,user_item_matrix,items_positions,train_DF,method=SI):

    complete_matrix = complete_user_item_matrix(user_item_matrix,method)

    items_attributes = defaultdict(int)

    #get the scores computed by the matrix completion
    for item in items_positions:
        items_attributes[item] = complete_matrix[:,items_positions[item]]

    #remove the items already rated by any group member in the test set
    to_remove = []
    items_union_train = utils.group_union(group,train_DF)

    for item in items_positions:
        if item in items_union_train:
            items_attributes.pop(item)

    return items_attributes


'''
Verify if the item_id is relevant for the group given the dataframe.
To be relevant for a group an item should be relevant for all its members
'''
def is_relevant(group,item_id,dataframe):

    relevant = True
    for user in group:
        relevant = relevant and (not dataframe[(dataframe.user_id == user) &
                                              (dataframe.item_id == item_id)].empty)

    return relevant



'''
Receives a list of groups (each group contain a list of users),
the rankings used as input to users and the dataset used to construct
those rankings and returns a group dataset composed by the scores returned
to each items by a matrix completion algorithm.


output:
dataset =[
[<item_relevance>,user_id,att1,att2,...,attn,item_id]
[<item_relevance>,user_id,att1,att2,...,attn,item_id]
[<item_relevance>,user_id,att1,att2,...,attn,item_id]
...]

dataset_maping = [
user_id : [item1,item2,item3,...]
user_id : [item1,item2,item3,...]

'''
def create_letor_dataset(groups,users_rankings,train_dataframe, test_dataframe):

    dataset = []
    dataset_mapping = defaultdict(list)

    for group_id,group in enumerate(groups):
        matrix_scores,items_positions = construct_user_item_matrix(group,users_rankings)
        items_attributes = calc_items_attributes(group,matrix_scores,items_positions,train_dataframe)
        sorted_items = sorted(items_attributes.keys())
        
        for item_id in sorted_items:
            dataset_mapping[group_id].append(item_id)

            dataset.append([])
            if is_relevant(group,item_id,test_dataframe):
                dataset[-1].append(1)
            else:
                dataset[-1].append(0)

            dataset[-1].append(group_id)
            [dataset[-1].append(item_score) for item_score in items_attributes[item_id]]
            dataset[-1].append(item_id)

    return dataset,dataset_mapping


def save_letor_dataset(dataset,dataset_mapping,out_file,out_file_map):


    with open(out_file,'w') as out:
        for line in dataset:
            s = str(line[0])+' qid:'+str(line[1])
            for att_i,att in enumerate(line[2:-1]):
                s += ' '+str(att_i+1)+':'+str(att)
            s += ' #docid = '+str(line[-1])+' inc = 0.0 prob = 0.0\n'
            out.write(s)


    with open(out_file_map,'w') as out_map:
        sorted_users = sorted(dataset_mapping.keys())
        item_pos = 0
        for user in sorted_users:
            s = str(user)+':'
            for item_id in dataset_mapping[user][:-1]:
                s += '({},{});'.format(item_pos,item_id)
                item_pos += 1
            s += '({},{})\n'.format(item_pos,dataset_mapping[user][-1])
            item_pos += 1
            out_map.write(s)


def recommend_v2(group,users_rankings,recomm_matrix,items_positions,train_DF,method=SI):

    nusers,nitems = recomm_matrix.shape

    #actually, stores the average times the item was recommended
    #times_recommended = [sum(recomm_matrix[:,n])/float(nusers) for n in range(nitems)]    
    if method == BS:
        complete_matrix = method().fit_transform(recomm_matrix)
    else:
        complete_matrix = method().complete(recomm_matrix)

    items_scores = [np.average(complete_matrix[:,i]) for i in range(nitems)]
    
    rank = [(item,items_scores[items_positions[item]]) for item in items_positions]

    rank = sorted(rank,reverse=True,key = lambda tup: tup[1])

    rank = grs_rec.remove_items_in_train(train_DF,group,rank)

    return rank



def recommend_v1(group,users_rankings,recomm_matrix,items_positions,train_DF):

    nusers,nitems = recomm_matrix.shape
    recomm_matrix[recomm_matrix == np.nan] = 0.0

    #actually, stores the average times the item was recommended
    times_recommended = [sum(recomm_matrix[:,n])/float(nusers) for n in range(nitems)]    

    #factorize the matrix 
    P,S,Q = np.linalg.svd(recomm_matrix, full_matrices=False)

    #construct the items coeficients for each user j
    recomm_coeficients = np.array([np.dot(np.dot(P[j],S),Q[j]) for j in range(nusers)])

    #to each item sum the coeficients of all users
    new_coeficients = [sum(recomm_coeficients[:,i]) for i in range(nitems)]

    #add the times_recommended to the computed coeficients
    scores = [times_recommended[i]+new_coeficients[i] for i in range(nitems)]

    rank = [(item,scores[items_positions[item]]) for item in items_positions]
 
    rank = sorted(rank,reverse=True, key = lambda tup : tup[1])

    rank = grs_rec.remove_items_in_train(train_DF,group,rank)


    return rank









if __name__ == '__main__':

    args = parse_args()

    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)


    rec_partitions = [sorted(glob.glob(os.path.join(args.base_dir,"*"+recomm+".out"))) \
        for recomm in args.base_recs]
    
    rec_partitions_reeval = [sorted(glob.glob(os.path.join(args.base_dir,'reeval',"*"+recomm+".out"))) \
                for recomm in args.base_recs]

    groups = utils.read_groups(args.groups_file)

    num_partitions = len(rec_partitions[0])

    #base_recs_names = [os.path.basename(rec) for rec in rec_partitions]
    #for base_rec_partition in base_rec:
    for part in range(1,num_partitions+1):
        #args.base_rec = base_rec
        curr_part = 'u'+ str(part) #re.search('u[1-9]+',base_rec).group(0)
        #curr_part = base_rec[:2] #the two first caracthers are the partiton
        args.test_file = args.test_file.replace(args.part,curr_part)
        args.train_file = args.train_file.replace(args.part,curr_part)
        args.part = curr_part    
        #ipdb.set_trace()
        paths = [base_rec[part-1] for base_rec in rec_partitions]        

        users_rankings = utils.rankings_dict(paths,rank_size=args.i2use)

        test = utils.read_ratings_file(args.test_file)
        val = utils.read_ratings_file(args.val_file)
        train = utils.read_ratings_file(args.train_file)

        if args.save_dataset:
            if not os.path.isdir(os.path.join(args.out_dir,'classif/')):
                os.mkdir(os.path.join(args.out_dir,'classif/'))

            dataset,dataset_mapping = create_letor_dataset(groups,users_rankings,train,val)
            out_file = os.path.join(args.out_dir,'classif','u'+str(part)+'.train.letor')
            out_file_map = os.path.join(args.out_dir,'classif','u'+str(part)+'.train.map')
            save_letor_dataset(dataset,dataset_mapping,out_file,out_file_map)

           
            #Salving test partition
            test = utils.read_ratings_file(os.path.join(args.base_dir,'reeval','u'+str(part)+'.test'))
            train = utils.read_ratings_file(os.path.join(args.base_dir,'reeval','u'+str(part)+'.base'))

            if not os.path.isdir(os.path.join(args.out_dir,'reeval/')):
                os.mkdir(os.path.join(args.out_dir,'reeval/'))

            
            paths_reeval = [base_rec[part-1] for base_rec in rec_partitions_reeval]
            users_rankings_reeval = utils.rankings_dict(paths_reeval,rank_size=args.i2use)

            dataset_reeval,dataset_reeval_mapping = create_letor_dataset(groups,
                                                    users_rankings_reeval,
                                                    train,test)

            out_file_reeval = os.path.join(args.out_dir,'reeval',
                                'u'+str(part)+'.train.letor')
            out_file_reeval_map = os.path.join(args.out_dir,'reeval',
                                'u'+str(part)+'.train.map')

            save_letor_dataset(dataset_reeval,dataset_reeval_mapping,
                                out_file_reeval,out_file_reeval_map)


        else:

            svd_rankings = {}
            nnm_rankings = {}

            mcomp_rankings = {}

            METHODS = {'SI':SI}#,'ISVD':ISVD,'MF':MF,'BS':BS}#,'MICE':MICE}
            for method in METHODS:
                mcomp_rankings[method] = {}
                for groupi,group in enumerate(groups):
                    '''matrix,items_positions = construct_user_item_matrix(group,users_rankings,with_ones=True)
                    rank_v1 = recommend_v1(group,users_rankings,matrix,items_positions,train)
                    svd_rankings[groupi] = rank_v1'''

                    matrix_scores,items_positions = construct_user_item_matrix(group,users_rankings)
                    rank_v2 = recommend_v2(group,users_rankings,matrix_scores,items_positions,train,method=METHODS[method])
                    mcomp_rankings[method][groupi] = rank_v2


                out_file = os.path.join(args.out_dir,'u'+str(part))
                #grs_rec.save_group_rankings(svd_rankings,out_file+"-SVD.gout")
                grs_rec.save_group_rankings(mcomp_rankings[method],out_file+"-"+method+".gout")
            #run_grs_ranking(paths)







    


