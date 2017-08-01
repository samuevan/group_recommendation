import groups_generator as gg
import grs_recommendation_agg as grs
import argparse
import numpy as np
import os
import re
import ipdb
import sys
import glob
import pandas as pd

def compute_distances(dataset_path, first_user, last_user,out_dir, simi_func=gg.PCC):

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    out_path = os.path.join(out_dir,'distances_{}-{}'.format(first_user,last_user))
    dataset = gg.read_ratings_file(dataset_path)
    users = sorted(list(dataset.user_id.unique()))
    #we use +2 to compensate the index 0 and the diference between the positions
    distances = np.zeros((last_user-first_user+2,len(users)+1))
    distances.fill(np.nan)
    for u1_pos,u1 in enumerate(range(first_user,last_user+1)):
        u1_DF = dataset[dataset.user_id == u1]
        print('{}'.format(u1))

        for u2 in users[u1:]:
            if u2%100 == 0:
                print('{}->{}'.format(u1,u2))
            u2_DF = dataset[dataset.user_id == u2]            
            dist = gg.calc_similarity(u1_DF,u2_DF,simi_func=simi_func)
            if dist.__class__ == tuple:
                dist = dist[0]
            distances[u1_pos+1][u2] = dist

    np.save(out_path,distances,allow_pickle=False)



'''def concatenate_matrices(distances_path,to_save=False):
    matrices_paths =sorted(glob.glob(os.path.join(distances_path,'*.npy')))

    matrices = []
    matrices.append(np.load(matrices_paths[0]))

    for matrix_path in matrices_paths[1:]:
        matrices.append(np.load(matrix_path)[1:])

    return np.concatenate(matrices)'''



def concatenate_matrices(matrices_folder,to_save=False,out_file=''):
    matrices = glob.glob(os.path.join(matrices_folder,'*.npy'))
    splits = [re.findall('[0-9]*-[0-9]*',matrix)[0] for matrix in matrices] 

    pre = os.path.dirname(matrices[0])+'/distances_'
    
    sorted_splits = sorted(splits,key = lambda x : int(x.split('-')[0]))
    d = [np.load(pre+sorted_splits[0]+'.npy')] 

    for part in sorted_splits[1:]:                                                                          
        f = pre+part+'.npy'                                                                                                 
        #ipdb.set_trace()
        d.append(np.load(f)[1:])


    if to_save:
        final_matrix = np.concatenate(d)
        np.save(os.path.join(matrices_folder,'distances_complete'),final_matrix,allow_pickle=False)
    else:
        return np.concatenate(d)



FUNCTION_MAP = {'PCC':gg.PCC, 
                'intersection':gg.len_intersect,
                'cosine' : gg.cosine,
                'jaccard' : gg.jaccard}

def arg_parse():
    p = argparse.ArgumentParser()
    
    p.add_argument('--base',type=str,
        help="Folder containing the training instances")

    p.add_argument('--simi_func',choices=FUNCTION_MAP.keys(),default='PCC',
        help='Similarity function')

    p.add_argument('-o','--out_dir',nargs='?')
    
    p.add_argument('--fu',type=int,default=1,
        help = "The id of the first user which metric will be computed")
        
    p.add_argument('--lu',type=int,default=943,
        help = "The id of the last user which metric will be computed")

    parsed = p.parse_args()

    parsed.simi_func = FUNCTION_MAP[parsed.simi_func]

    if parsed.out_dir is None:
        parsed.out_dir = parsed.base

    return parsed




if __name__ == '__main__':
    '''dataset_path = sys.argv[1]
    fu = int(sys.argv[2])
    lu = int(sys.argv[3])
    out_dir = sys.argv[4]
    func = sys.argv[5]'''

    args = arg_parse()
    
    compute_distances(args.base,args.fu,args.lu,args.out_dir,simi_func = args.simi_func)
    

