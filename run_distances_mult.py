import os
import sys
import groups_generator as gg


if __name__ == '__main__':
    dataset_f = sys.argv[1]
    out_dir = sys.argv[2]
    nump = int(sys.argv[3])
    dataset = gg.read_ratings_file(dataset_f)
    users = sorted(dataset.user_id.unique())

    step = int(len(users)/nump)
    
    for fu in range(0,len(users)+1,step):
        lu = min(fu + step,users[-1])
        #os.system('echo {0} {1}'.format(fu+1,lu))
        os.system('python save_distances.py {0} {1} {2} {3} &'.format(dataset_f,fu+1,lu,out_dir))
