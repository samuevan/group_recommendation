import os
import sys
import argparse
import groups_generator as gg



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


    p.add_argument('-p','--nump',type=int,default=1,
        help = "Number of process used to compute the function")

    parsed = p.parse_args()
    
    #we just need to set the correspondent function in the save_distances script
    #parsed.simi_func = FUNCTION_MAP[parsed.simi_func]

    if parsed.out_dir is None:
        parsed.out_dir = parsed.base

    return parsed



if __name__ == '__main__':


    '''dataset_f = sys.argv[1]
    out_dir = sys.argv[2]
    nump = int(sys.argv[3])'''

    args = arg_parse()

    dataset = gg.read_ratings_file(args.base)
    users = sorted(dataset.user_id.unique())

    halfp = int(args.nump*0.7)
    
    #run first half
    half_users = int(len(users)/2)
    step_half = int(half_users/halfp)
    for fu in range(1,half_users,step_half):
        lu = min(fu+step_half-1,users[half_users])
        args.fu = fu
        args.lu = lu
        print('{}->{}'.format(args.fu,args.lu))
        
        os.system('python save_distances.py --base {base} --fu {fu} --lu {lu} -o {out_dir} --simi_func {simi_func}&'.format(**vars(args)))

    print('half')
    first_2nd_half = lu+1
    step_2nd_half = int((len(users)-half_users)/(args.nump-halfp))
    for fu in range(first_2nd_half,len(users)+1,step_2nd_half):
        lu = min(fu+step_2nd_half-1,users[-1])
        
        args.fu = fu
        args.lu = lu
        print('{}->{}'.format(args.fu,args.lu))
        os.system('python save_distances.py --base {base} --fu {fu} --lu {lu} -o {out_dir} --simi_func {simi_func}&'.format(**vars(args)))

        #print('{}->{}'.format(fu,lu))
        #os.system('python save_distances.py {0} {1} {2} {3} &'.format(dataset_f,fu,lu,out_dir))



    '''step = int(len(users)/nump)
    
    for fu in range(0,len(users)+1,step):
        lu = min(fu + step,users[-1])
        #os.system('echo {0} {1}'.format(fu+1,lu))
        os.system('python save_distances.py {0} {1} {2} {3} &'.format(dataset_f,fu+1,lu,out_dir))'''
