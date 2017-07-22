import os
import sys
import groups_generator as gg


if __name__ == '__main__':
    dataset_f = sys.argv[1]
    out_dir = sys.argv[2]
    nump = int(sys.argv[3])

    dataset = gg.read_ratings_file(dataset_f)
    users = sorted(dataset.user_id.unique())

    halfp = int(nump*0.7)
    
    #run first half
    half_users = int(len(users)/2)
    step_half = int(half_users/halfp)
    for fu in range(1,half_users,step_half):
        lu = min(fu+step_half-1,users[half_users])
        print('{}->{}'.format(fu,lu))
        os.system('python save_distances.py {0} {1} {2} {3} &'.format(dataset_f,fu,lu,out_dir))

    print('half')
    first_2nd_half = lu+1
    step_2nd_half = int((len(users)-half_users)/(nump-halfp))
    for fu in range(first_2nd_half,len(users)+1,step_2nd_half):
        lu = min(fu+step_2nd_half-1,users[-1])
        print('{}->{}'.format(fu,lu))
        os.system('python save_distances.py {0} {1} {2} {3} &'.format(dataset_f,fu,lu,out_dir))



    '''step = int(len(users)/nump)
    
    for fu in range(0,len(users)+1,step):
        lu = min(fu + step,users[-1])
        #os.system('echo {0} {1}'.format(fu+1,lu))
        os.system('python save_distances.py {0} {1} {2} {3} &'.format(dataset_f,fu+1,lu,out_dir))'''
