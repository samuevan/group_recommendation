"""
This script receives the recommendations returned by a default recommendation
algorithm and a file containg groups of users and returns the recommendations
for each group.

"""


import re
import os
import sys
import groups_generator as gg
import ipdb
import argparse
import glob
from collections import defaultdict




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
verify if the items are present in the training file for any of the users
need to use average.keys() instead of just pass by the dictionary in order
to avoid the error : dictionary size changed size during iteration
'''
def remove_items_in_train(train_DF,group,rank):


    if rank.__class__ == dict:
        items_in_rank = rank.keys()
    else:
        items_in_rank = [x for x,_ in rank]


    items_to_remove = []

    items_union = gg.group_union(group,train_DF)

    for item in items_in_rank:
        if item in items_union:
            items_to_remove.append(item)


    for item in items_to_remove:
        if rank.__class__ == dict:
            rank.pop(item)
        else:
            #TODO too costy
            rank = [(x,y) for x,y in rank if x!= item]


    return rank


'''
input:
rec_rankings: reveives a set of rankings recommended to the users of a group
train_DF : Pandas Dataframe containing the train file used by base recommender

output: An aggregated ranking ordered by the least misery of the items scores
TODO: verify if we need to normalize the items scores
'''

def agg_ranking_least_misery(rec_rankings,group,train_DF,i2sug=10):
    averages = {}
    #Loop through users in the group and their recommend rankings
    for rank in rec_rankings:
        for item,item_score in rank:
            if item in averages:
                averages[item] = min(item_score, averages[item] )
            else:
                averages[item] = item_score

    #remove the items present in the train dataset for any of the users
    remove_items_in_train(train_DF,group,averages)

    group_size = len(rec_rankings)

    ranking = [(item,averages[item]) for item in averages]

    ranking = sorted(ranking,key = lambda tup : tup[1], reverse = True)[:i2sug]

    return ranking




'''
input: reveives a set of rankings recommended to the users of a group
output: An aggregated ranking ordered by the average of the items scores

TODO: verify if we need to normalize the items scores
'''
def agg_ranking_average(rec_rankings,group,train_DF,i2sug=10):
    averages = {}
    for rank in rec_rankings:
        for item,item_score in rank:
            if item in averages:
                averages[item] += item_score
            else:
                averages[item] = item_score

    #remove the items present in the train dataset for any of the users
    remove_items_in_train(train_DF,group,averages)

    group_size = len(rec_rankings)
    ranking = [(item,averages[item]/group_size) for item in averages]
    ranking = sorted(ranking,key = lambda tup : tup[1], reverse = True)[:i2sug]
    return ranking



def MRA_comb(rankings,group,train_DF,rank_size=10):
    scores = {}
    counting = {} #counts how many times the items apepars in the rankings
    if isinstance(rankings,dict):
        rankings_to_agg = list(rankings.values())
    else:
        rankings_to_agg = rankings

    #pega o tamanho do maior ranking para o caso de rankings de tamanhos diferentes
    input_size = max([len(rankings_to_agg[i]) for i in range(len(rankings_to_agg))])

    final_rank = []
    #percorre os rankings posicao a posicao
    for pos in range(input_size):
        for rank in rankings_to_agg:
            if pos < len(rank):
                #para cada ranking, conta o numero de vezes que o item na posicao
                #corrente já apareceu
                if rank[pos].__class__ == tuple:
                    elem = rank[pos][0]
                else:
                    elem = rank[pos]

                if elem in scores:
                    scores[elem] += 1
                    #caso o item tenha aparecido em mais de metada dos
                    #rankings (median) ele eh adicionado no ranking final
                    #e removido do dicionario de controle
                    if scores[elem] >= (len(rankings)/2):
                        final_rank.append((elem,scores[elem]))
                        scores.pop(elem)
                elif not elem in final_rank:
                    scores[elem] = 1

    #caso nao tenha preenchido o total de items necessarios para o tamanho do
    #raking de saida, adiciona os items ordenados pela frequencia dos mesmos nos
    #rankings de entrada. Note que essa frequencia pode ser inclusive maior que
    #a mediana, mas o que importa no metodo eh o instante em que um item alcanca
    #a mediana das contagems

    if len(final_rank) < rank_size:
        sorted_scores = sorted(list(scores.items()), key = lambda tup : tup[1], reverse = True)
        pos = 0
        #while len(final_rank) < rank_size and pos < len(scores):
        for elem in sorted_scores:
            final_rank.append(elem)
            pos += 1

    #remove the items that any user rated in train
    remove_items_in_train(train_DF,group,final_rank)

    return final_rank[:rank_size]


def RRF_func(pos,k):
    return 1.0/(k+pos)

def RRF_comb(rankings,group,train_DF,rank_size=10,k=60):
    scores = {}
    counting = {} #counts how many times the items apepars in the rankings
    if isinstance(rankings,dict):
        rankings_to_agg = list(rankings.values())
    else:
        rankings_to_agg = rankings

    for rank in rankings_to_agg:
        for pos,elem in enumerate(rank):
            if elem.__class__ == tuple:
                elem = elem[0]

            if elem in scores :
                scores[elem] += RRF_func(pos,k)
            else:
                scores[elem] = RRF_func(pos,k)

    final_rank = [(x,y) for x,y in sorted(scores.items(), key = lambda tup : tup[1],reverse=True)]
    remove_items_in_train(train_DF,group,final_rank)
    return final_rank[:rank_size]


def agg_ranking_borda(rec_rankings,group,train_DF,i2sug=10):

    borda_scores = {}

    for rank in rec_rankings:
        for item_pos,item in enumerate(rank):
            item_id,item_score = item
            if item_id in borda_scores:
                borda_scores[item_id] += len(rank)-item_pos
            else:
                borda_scores[item_id] = len(rank)-item_pos


    #remove the items present in the train dataset for any of the users
    remove_items_in_train(train_DF,group,borda_scores)

    final_ranking = [(item,borda_scores[item]) for item in borda_scores]
    final_ranking = sorted(final_ranking,key = lambda tup : tup[1], reverse = True)[:i2sug]

    return final_ranking


def oraculus(rec_rankings,group,train_DF,test_DF,i2sug=10):
    #group ground truth. i.e the items shared by the members of the group in the test set

    intersection_test = list(gg.group_intersection(group,test_DF))
    intersection_recomm = list(utils.rank_union(rec_rankings))
    #intersection_recomm = list(rank_intersection(rec_rankings))
    #ipdb.set_trace()
    group_gt = [(x,1.0) for x in intersection_recomm if x in intersection_test]

    #remove the items present in the train dataset for any of the users
    remove_items_in_train(train_DF,group,group_gt)

    for i in range(len(group_gt),i2sug):
        group_gt.append((9999999,0.0001))

    return group_gt[:i2sug]




def save_group_rankings(group_rankings,out_file_path):

    print(out_file_path)

    with (open(out_file_path,'w')) as out_file:
        print("Saving "+out_file_path)
        for group_id in group_rankings:
            str_aux = str(group_id) + '\t['
            str_aux += ','.join(
                        [str(item[0])+":"+ str(item[1]) for item in group_rankings[group_id] ])
            str_aux += ']\n'
            out_file.write(str_aux)


#def agg_rating_average(rec_ratings):

#def agg_average(recomendations,rank=True)


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
    p.add_argument('-o','--out_dir',type=str,default='',
        help='Output folder. The group recommendations will be saved in this foder')
    p.add_argument('--i2use',type=int,default=20,
        help='Size of the input rankings')
    p.add_argument('--i2sug',type=int,default=10,
        help='Size of the outrput rankings')

    parsed = p.parse_args()
    #ipdb.set_trace()
    if parsed.out_dir == '':
        parsed.out_dir = parsed.base_dir

    #base_rec_path = parsed.part + "-" + parsed.base_rec + ".out"
    #parsed.base_rec = os.path.join(parsed.base_dir,base_rec_path)

    parsed.test_file = os.path.join(parsed.base_dir,parsed.part+'.test')
    parsed.train_file = os.path.join(parsed.base_dir,parsed.part+'.base')

    #we use the same groups for all partitions
    #parsed.groups_file = parsed.groups_file.replace('u1',parsed.part)
     
    file_regex = r".*u[1-5]-([a-z A-Z]*)"
    if not parsed.base_recs:
        recs_in_dir = set().union([re.match(file_regex,rec).groups()[0] \
                        for rec in sorted(glob.glob(os.path.join(parsed.base_dir,"*.out")))])
        #ipdb.set_trace()
        parsed.base_recs = list(recs_in_dir)

	        


    return parsed



def run_grs_ranking(paths):

    groups = utils.read_groups(args.groups_file)

    users_rankings = utils.rankings_dict(paths,rank_size=args.i2use)

    test = gg.read_ratings_file(args.test_file)
    train = gg.read_ratings_file(args.train_file)

    avg_group_rec = {}
    LM_group_rec = {}
    borda_group_rec = {}
    oraculus_group_rec = {}
    MRA_group_rec = {}
    RRF_group_rec = {}
    for group_i, group in enumerate(groups):
        if group_i%300 == 0:
            print(group_i)
        rankings = [rank for user in group for rank in users_rankings[user]]
        avg_group_rec[group_i] = agg_ranking_average(rankings,group,train,args.i2sug)

        LM_group_rec[group_i] = agg_ranking_least_misery(rankings,group,train,args.i2sug)
        borda_group_rec[group_i] = agg_ranking_borda(rankings,group,train,args.i2sug)

        oraculus_group_rec[group_i] = oraculus(rankings,group,train,test,args.i2sug)
        MRA_group_rec[group_i] = MRA_comb(rankings,group,train,args.i2sug)
        RRF_group_rec[group_i] = RRF_comb(rankings,group,train,args.i2sug)
    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    print("here saving my favorite rankings")
    #ipdb.set_trace()
    out_file = str(args.part)+'-'
    out_file += '_'.join(args.base_recs)
    out_file += re.sub('.*groups','',os.path.basename(args.groups_file))
    #out_file += os.path.basename(args.groups_file).replace(args.part+'.groups','')
    out_file = os.path.join(args.out_dir,out_file)

    save_group_rankings(avg_group_rec,out_file+'_avg.gout')
    save_group_rankings(LM_group_rec,out_file+'_lm.gout')
    save_group_rankings(borda_group_rec,out_file+'_borda.gout')
    save_group_rankings(oraculus_group_rec,out_file+'_GT.gout')
    save_group_rankings(MRA_group_rec,out_file+'_MRA.gout')
    save_group_rankings(RRF_group_rec,out_file+'_RRF.gout')


if __name__ == '__main__':
    args = parse_args()
    #rec_partitions = sorted(glob.glob(os.path.join(args.base_dir,"*"+args.base_rec+".out")))

    rec_partitions = [sorted(glob.glob(os.path.join(args.base_dir,"*"+recomm+".out"))) \
        for recomm in args.base_recs]

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
        paths = [base_rec[part-1] for base_rec in rec_partitions]
        run_grs_ranking(paths)
