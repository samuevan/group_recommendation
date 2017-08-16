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
            rank = [(x,y) for x in rank if x!= item]



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

    group_gt = [(x,1.0) for x in intersection_test]

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
    p.add_argument('--base_rec',type=str,required=True,
        help='The file containing the recommendations performed by a base recommender')
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


    return parsed



def run_grs_ranking():

    groups = read_groups(args.groups_file)

    users_rankings = rankings_dict(args.base_rec,rank_size=args.i2use)

    #TODO preciso adicionar o arquivo de treino para conferir se o item não foi visto por nenhum usuario do grupo
    test = gg.read_ratings_file(args.test_file)
    train = gg.read_ratings_file(args.train_file)

    avg_group_rec = {}
    LM_group_rec = {}
    borda_group_rec = {}
    oraculus_group_rec = {}

    for group_i, group in enumerate(groups):
        if group_i%300 == 0:
            print(group_i)

        rankings = [users_rankings[user] for user in group]
        #group_ranking_avg = agg_average_ranking(rankings)
        avg_group_rec[group_i] = agg_ranking_average(rankings,group,train,args.i2sug)
        #group_ranking_LM = agg_least_misery_ranking(rankings)
        LM_group_rec[group_i] = agg_ranking_least_misery(rankings,group,train,args.i2sug)
        borda_group_rec[group_i] = agg_ranking_borda(rankings,group,train,args.i2sug)
        oraculus_group_rec[group_i] = oraculus(rankings,group,train,test,args.i2sug)


    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    print("here saving my favorite rankings")
    #ipdb.set_trace()
    out_file = os.path.basename(args.base_rec).replace('.out','')
    out_file += re.sub('.*groups','',os.path.basename(args.groups_file))
    #out_file += os.path.basename(args.groups_file).replace(args.part+'.groups','')
    out_file = os.path.join(args.out_dir,out_file)

    save_group_rankings(avg_group_rec,out_file+'_avg.gout')
    save_group_rankings(LM_group_rec,out_file+'_lm.gout')
    save_group_rankings(borda_group_rec,out_file+'_borda.gout')
    save_group_rankings(oraculus_group_rec,out_file+'_GT.gout')


if __name__ == '__main__':
    args = parse_args()
    rec_partitions = sorted(glob.glob(os.path.join(args.base_dir,"*"+args.base_rec+".out")))

    #base_recs_names = [os.path.basename(rec) for rec in rec_partitions]

    for base_rec in rec_partitions:
        args.base_rec = base_rec
        curr_part = re.search('u[1-9]+',base_rec).group(0)
        #curr_part = base_rec[:2] #the two first caracthers are the partiton
        args.test_file = args.test_file.replace(args.part,curr_part)
        args.train_file = args.train_file.replace(args.part,curr_part)
        args.part = curr_part

        run_grs_ranking()
