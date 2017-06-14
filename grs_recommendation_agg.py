"""
This script receives the recommendations returned by a default recommendation
algorithm and a file containg groups of users and returns the recommendations
for each group.

"""


import re
import ipdb

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
input: reveives a set of rankings recommended to the users of a group
output: An aggregated ranking ordered by the least misery of the items scores

TODO: verify if we need to normalize the items scores
'''
def agg_ranking_least_misery(rec_rankings):
    averages = {}
    for rank in rec_rankings:
        for item,item_score in rank:
            if item in averages:
                averages[item] = min(item_score, averages[item] )
            else:
                averages[item] = item_score

    group_size = len(rec_rankings)
    ranking = sorted([(item,averages[item]/group_size) for item in averages], 
                key = lambda tup : tup[1], reverse = True)
    return ranking




'''
input: reveives a set of rankings recommended to the users of a group
output: An aggregated ranking ordered by the average of the items scores

TODO: verify if we need to normalize the items scores
'''
def agg_ranking_average(rec_rankings):
    averages = {}
    for rank in rec_rankings:
        for item,item_score in rank:
            if item in averages:
                averages[item] += item_score 
            else:
                averages[item] = item_score

    group_size = len(rec_rankings)
    ranking = sorted([(item,averages[item]/group_size) for item in averages], 
                key = lambda tup : tup[1], reverse = True)
    return ranking


#def agg_rating_average(rec_ratings):


#def agg_average(recomendations,rank=True)



def run_grs_ranking(recommendation_file,groups_file):    
    groups = read_groups(groups_file)
    users_rankings = rankings_dict(recommendation_file,rank_size=10)

    for group in groups:        
        rankings = [users_rankings[user] for user in group]
        ipdb.set_trace()
        agg_average_ranking(rankings)



#def run_grs(recommendation_file,groups_file,rank=True):








