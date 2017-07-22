import groups_generator as gg
import grs_recommendation_agg as grs
import numpy as np
import os
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
            dist,_ = gg.calc_similarity(u1_DF,u2_DF,simi_func=simi_func)
            distances[u1_pos+1][u2] = dist

    np.save(out_path,distances,allow_pickle=False)



def concatenate_matrices(distances_path,to_save=False):
    matrices_paths =sorted(glob.glob(os.path.join(distances_path,'*.npy')))

    matrices = []
    matrices.append(np.load(matrices_paths[0]))

    for matrix_path in matrices_paths[1:]:
        matrices.append(np.load(matrix_path)[1:])

    return np.concatenate(matrices)



if __name__ == '__main__':
    dataset_path = sys.argv[1]
    fu = int(sys.argv[2])
    lu = int(sys.argv[3])
    out_dir = sys.argv[4]

    compute_distances(dataset_path,fu,lu,out_dir)

