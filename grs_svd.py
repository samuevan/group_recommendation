import utils
import pandas as pd
import numpy as np




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
     
    if not parsed.base_recs:
        recs_in_dir = set().union([rec[3:-4] for rec in sorted(glob.glob(os.path.join(parsed.base_dir,"*.out")))])
        parsed.base_recs = list(recs_in_dir)    
	        


    return parsed





'''
Constroi uma matrix user x item
cada linha representa as recomendacoes feitas para um usuario
#TODO tambem inserir na matrix os ratings do usuario que estao na base de treino
'''


def construct_user_item_matrix(group,user_rankings):
    
    group_rankings = [user_ranking[user] for user in group]

    all_items = defaultdict([(x,i) for i,x in enumerate(utils.rank_union(group_rankings)))])

    recomm_matrix = np.zeros((len(group),len(all_items)))


    for useri,user in enumerate(user_rankings):
        for ranki,user_rank in enumerate(group_rankings):
            for item in ranki:
                item_pos = all_items[item]
                recomm_matrix[useri,item_pos] = 1

 



if __name__ == '__main__':

    args = arg_parse()

    groups = utils.read_groups(args.groups_file)

    users_rankings = utils.rankings_dict(paths,rank_size=args.i2use)

    test = gg.read_ratings_file(args.test_file)
    train = gg.read_ratings_file(args.train_file)

