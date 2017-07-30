'''
This version fixes the following issues:

'''


import numpy as np
import re
import math
import glob
import sys


'''rank_comm : Recommended list
test : list with the items the user already seen
'''
def weighted_precision_at(rank_recomm, test,num_items_to_eval):
    hits = 0.0
    avg_prec = 0.0
    #print rank_recomm
    #print test
    #Considerando somente os 100 primeiros
    for i in range(num_items_to_eval):
        kx = rank_recomm[i] #cada linha tem um item e o score da regressao
        kx_in_test = [True for item_rat in test if item_rat[0] == kx] 
        if len(kx_in_test) > 0:
            hits = hits+1;
            avg_prec += hits/(i + 1);

    #print "hits: " + str(hits) + "avg_prec: " +str(avg_prec)
    #a = input()
    if  hits != 0:
        prec = avg_prec/min(len(test),num_items_to_eval)                
        return prec
    else:

        return 0


def precision_at(rank_recomm, test,num_items_to_eval):
    hits = 0.0
    prec = 0.0
    #print rank_recomm
    #print test
    #Considerando somente os 100 primeiros
    for i in range(num_items_to_eval):
        kx = rank_recomm[i] #cada linha tem um item e o score da regressao
        kx_in_test = [True for item_rat in test if item_rat[0] == kx] 
        if len(kx_in_test) > 0:
            hits = hits+1;

    prec = hits/num_items_to_eval

    return prec

'''
rank : lista de ratings dos itens (nao os items propriamente ditos)
size_at : tamanho do k em DCG@k 
'''
def dcg(rank, size_at):
        res = 0.0;
        if len(rank) > 0: 
            #res = rank[0]

            #TODO alterar i < vet.size() por i < k

            for i in range(len(rank)):
                res += (2**rank[i] - 1) / math.log(i + 2) #estou usando +2 pois comeco de 0

        return res



def NDCG(agg_rankings, test_inputs,size_at=10):
    ndcg_value = 0.0
    for key in agg_rankings.keys():
        if agg_rankings.has_key(key) and test_inputs.has_key(key):
            #print "user: " + str(key)
            hits_and_ratings = get_hits_and_ratings(agg_rankings[key], test_inputs[key],size_at)

            ratings = [y for x,y in hits_and_ratings]
            ideal_rat_order = sorted(ratings,reverse=True)
            if len(ratings) > 0:
                dcg_recomm = dcg(ratings,size_at)
                idcg = dcg(ideal_rat_order,size_at)    
                #if len(ratings) > 1:
                    #print ratings
                    #print "%f - %f" %(dcg_recomm,idcg)
                ndcg_value += dcg(ratings,size_at)/dcg(ideal_rat_order,size_at)    


    ndcg_value = ndcg_value/len(agg_rankings)

    return ndcg_value   


def MAP(agg_rankings, test_inputs,size_at=10):
    
    map_value = 0.0
    num_users_test_has_elem = 0
    for key in agg_rankings.keys():
        if agg_rankings.has_key(key) and test_inputs.has_key(key):
            #print "user: " + str(key)
            map_value += weighted_precision_at(agg_rankings[key],test_inputs[key],size_at)
            num_users_test_has_elem += 1


    

    #print "Num users test "+str(num_users_test_has_elem) +" - "+ str(map_value/num_users_test_has_elem)
    #print "Num users " + str(len(agg_rankings))

    map_value = map_value/len(agg_rankings)

    return map_value   


def get_hits_and_ratings(rank_recomm, test,size_at):
    hits_and_ratings = []
    for i in range(size_at):
        kx = rank_recomm[i] #cada linha tem um item e o score da regressao
        for item_rat in test:
            if item_rat[0] == kx:
                hits_and_ratings.append(item_rat)

    return hits_and_ratings


def hits_statistics(agg_rankings, test_inputs,size_at=10):
    hits_total = 0.0
    hits_avg = 0.0
    for key in agg_rankings.keys():
        if agg_rankings.has_key(key) and test_inputs.has_key(key):

            for i in range(size_at):
                kx = agg_rankings[key][i] #cada linha tem um item e o score da regressao
                kx_in_test = [True for item_rat in test_inputs[key] if item_rat[0] == kx] 
                if len(kx_in_test) > 0:
                    hits_total = hits_total+1;
            

    hits_avg = hits_total /len(agg_rankings)
    return hits_total,hits_avg



def avg_precision_at(agg_rankings, test_inputs,size_at=10):

    prec_value = 0.0
    for key in agg_rankings.keys():
        if agg_rankings.has_key(key) and test_inputs.has_key(key):
            #print "user: " + str(key)
            prec_value += precision_at(agg_rankings[key],test_inputs[key],size_at)


    prec_value = prec_value/len(agg_rankings)

    return prec_value   


def read_ranking(ranking_file):

    users_rankings = {}
    ranking_handler = open(ranking_file)
    for line in ranking_handler:
        tokens = line.strip().split('\t')
        user = int(tokens[0])
        items = tokens[1].strip().replace('[','').replace(']','').split(',')
        users_rankings[user] = []
        for item_score in items:
            item = int(item_score.split(':')[0])
            users_rankings[user].append(item)

    return users_rankings            



def read_test(test_file):

    data = open(test_file,'r')
    past_usr,past_mov,past_rat = data.readline().strip().split('\t')
    line_usr = past_mov

    users_test = {int(past_usr):[(int(past_mov),float(past_rat))]}


    nusers = 1;
    for line in data:
        usr,mov,rat = line.strip().split('\t')
        
        if usr == past_usr:

            users_test[int(usr)].append((int(mov),float(rat)))
            #line_usr += ' '+mov
            past_usr,past_mov = usr,mov
        else:
            users_test[int(usr)] = [(int(mov),float(rat))]
            nusers += 1
            past_usr,past_mov = usr,mov

    return users_test


def run(basedir):

    for part in range(1,6):
        partition = "u"+str(part)
        files = sorted(glob.glob(basedir+partition+"*.out"))    
        test = read_test(basedir+partition+'.test')

       # print files
        print "\t\t\tmap@10\tp@1\tp@5\tp@10\tNDCG\thits\tavg hits"
        for f in files:
            data = read_ranking(f)
            map10 = MAP(data,test,10)
            p1 = avg_precision_at(data,test,1)
            p5 = avg_precision_at(data,test,5)
            p10 = avg_precision_at(data,test,10)
            ndcg10 = NDCG(data,test,10)
            total_hits,avg_hits = hits_statistics(data,test,10)
            print str(f)+"\t\t\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%d\t%.4f" %(map10,p1,p5,p10,ndcg10,total_hits,avg_hits)


        



if __name__ == "__main__":
    
    basedir = sys.argv[1]
    run(basedir)
