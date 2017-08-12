import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import groups_generator as gg

'''
this function returns statistics for the groups
The number of unique users
The number of groups with items in the dataset
The number medium number of items for each group (for all groups and for the groups with items)
The average groups correlation

TODO add distance matrix as aditional parameter
'''
def compute_group_statistics(groups,dataset,compute_corr = False):


    intersection = [gg.group_intersection(g,dataset) for g in groups]

    items_per_group = []
    groups_with_items = 0
    unique_items = set()
    unique_groups = set()
    for group in intersection:
        items_per_group.append(len(group))
        if items_per_group[-1] > 0:
            groups_with_items += 1
        for item in group:
            unique_items.add(item)

    unique_users = set()
    for g in groups:
        str_group = str(sorted(g))
        unique_groups.add(str_group)
        for u in g:
            unique_users.add(u)

    avg_corr_total = 0.0

    if compute_corr:
        for g in groups:
            avg_corr_g = 0
            i = 0

            for u1,u2 in combinations(g,2):
                #x = calc_similarity(u1,u2,dataset)[0]
                x = gg.calc_similarity(dataset[dataset.user_id == u1],dataset[dataset.user_id == u2])[0]
                avg_corr_g += x
                i += 1

            avg_corr_total += avg_corr_g/i
            #print(avg_corr_total)
        avg_corr_total /= len(groups)

    out = 'Unique users: {0}\nUnique items: {1}\n'
    out += 'Unique groups {2}\nGroups with items: {3}\n'
    out +='Avg number of group items (total): {4}\n'
    out+='Avg number of group items (valid groups): {5}\nAvg. Group Correlation: {6}'

    #print(out.format(len(unique_users),len(unique_items),len(unique_groups),groups_with_items,sum(items_per_group)/len(groups),sum(items_per_group)/groups_with_items,avg_corr_total))
    return(out.format(len(unique_users),len(unique_items),len(unique_groups),groups_with_items,sum(items_per_group)/len(groups),sum(items_per_group)/groups_with_items,avg_corr_total))


def plot_distances_hist(distances,out_dir,out_file,threshold=0.27):

    distances_plain = []
    distance_bigger_thresh = []

    lines, rows = distances.shape

    for i in range(lines):
        for j in range(rows):
            if not np.isnan(distances[i][j]):
                if distances[i][j] < -1.0:
                    print('({},{}) : {}'.format(i,j,distances[i][j]))
                else:
                    distances_plain.append(distances[i][j])
                    if distances[i][j] >= threshold:
                        distance_bigger_thresh.append(distances[i][j])

    print(len(distance_bigger_thresh)/len(distances_plain))
    mean_distance = np.average(distances_plain)
    median_distance = np.median(distances_plain)

    fig,ax = plt.subplots()

    n,bins,patches = ax.hist(distances_plain,bins=50)


    ymin = min(n)
    ymax = max(n)

    ax.vlines(mean_distance,ymin=min(n),ymax=max(n)+0.1*ymax,colors='r',linestyles='-.')
    ax.vlines(0.03,ymin=min(n),ymax=max(n)+0.1*ymax,colors='r',linestyles='--')
    ax.vlines(threshold,ymin=min(n),ymax=max(n)+0.1*ymax,colors='r',linestyles=':')

    plt.ylim((min(n),max(n)+0.1*ymax))
    plt.xlim((-1.0,1.0))
    my_ticks = sorted([-1.0,0.03,mean_distance,threshold,1.0])


    ax.set_xlabel("Pearson Correlation Coeficient")
    ax.set_ylabel("# users pairs")
    plt.xticks(my_ticks,rotation='vertical')
    #plt.legend()
    #plt.show()
    plt.savefig(os.path.join(out_dir,out_file+'.pdf'))
    plt.close()
