#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""site
@author: manuelalautizi
"""
import os
os.chdir("/home/manuela/Desktop/master_thesis/")
# =============================================================================
# IMPORT PACKAGES
# =============================================================================
import pandas as pd
import numpy as np
import networkx as nx
from operator import itemgetter
import matplotlib
from itertools import groupby
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from time import time
from toolz import unique
import itertools as it
import random
from sklearn.decomposition import PCA
import cpplogrank

random.seed(87654)#(30)
# =============================================================================
# FUNCTIONS
# =============================================================================

# =============================================================================
# ObjectiveFunction (computed by using PCA)
# =============================================================================
import itertools
import pandas as pd
import networkx as nx
from sklearn.decomposition import PCA
import cpplogrank

# Adjacency matrix simulated network
G = pd.read_csv("biogrid.human.entrez.tsv", sep="\t")
# Matrix with simulated gene expression, survival time and censor data
patients = pd.read_csv("GSE30219.tumor.os.entrez.csv", index_col=0)
patients.reset_index(drop=True, inplace=True)

G.columns = ["A", "B"]
G.head()

# CREATION OF THE GRAPH
tuples = [tuple(G.iloc[i]) for i in range(G.shape[0])]
G = nx.Graph(tuples)

# REMOVE NODES WITHOUT THE GE DATA
Vertices = list(G.nodes())
V = list(set(Vertices).intersection([i for i in map(int, patients.columns.tolist()[3:])])) #3
# i remove from the graph the nodes not in that list (V)
G.remove_nodes_from(list(set(G.nodes())^set(V)))
# remove nodes with degree 0
G.remove_nodes_from(list(nx.isolates(G)))
# remove self loops
self_loops = [(n, n) for n, nbrs in G.adj.items() if n in nbrs]
G.remove_edges_from(self_loops)
# remove components made of 2 nodes
G.remove_nodes_from(list(itertools.chain(*[i for i in nx.connected_components(G) if len(i)==2])))
    
def comp_lrt(df, fts):
    fts = [i for i in map(str, fts)]
    if len(fts) == 1:
        df = df[fts + ['os_time', 'os_event']]
        P0 = df[df.loc[:, fts[0]] >= np.median(df[fts])]
        P1 = df[df.loc[:, fts[0]] < np.median(df[fts])]
        
        lrt = cpplogrank.logrank_test(P0['os_time'], P1['os_time'], P0['os_event'], P1['os_event'])
        return(lrt)
        
    else:
        pca = PCA(n_components = 2, random_state=1234)
        
        principalComponents = pca.fit_transform(df[fts])
        principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
        
        new = pd.concat([df[fts + ['os_time', 'os_event']], principalDf.iloc[:,0]], axis=1)
        new = new.sort_values('principal component 1')
               
        P0 = new[new['principal component 1'] > 0]
        P1 = new[new['principal component 1'] <= 0]
        
        lrt = cpplogrank.logrank_test(P0['os_time'], P1['os_time'], P0['os_event'], P1['os_event'])
        
        return(lrt)

# =============================================================================
# Computation Best Theoretical Solution & Greedy Solution
# =============================================================================
def best_theo(G, df, K):    
    
    all_paths = []   
    for s in G.nodes():
        for t in G.nodes():
            if s != t:
                for p in nx.all_simple_paths(G, s, t, K):
                    if len(p) == K:
                        all_paths.append(p)

    for i in G.nodes():
        if len(list(G.neighbors(i))) > (K-1):
            a = list(it.combinations(G.neighbors(i), K-1))
            for j in a:
                all_paths.append([i, j[0], j[1], j[2]])

    all_paths  = list(map(list, unique(map(tuple, all_paths))))
        
    scores_theo = sorted([(p,-np.log10(comp_lrt(df, p))) for p in all_paths], key=lambda x:x[1], reverse=True)
    best_theo = [scores_theo[i] for i in range(len(scores_theo)) if scores_theo[i][1] == scores_theo[0][1]]
        
    return (best_theo)

# =============================================================================
#  Greedy
# =============================================================================    
def greedy_algorithm(df, G, K):

    all_paths = []        
    for n in G.nodes():        
        path = []
        path.append(n)    
        
        for i in range(K-1):
        
            best_score = 0
            best_n = 0   
            for k, v in G[n].items():
                if v['weight'] > best_score:
                    best_score = v['weight']
                    best_n = k
    
            if best_n not in path:
                path.append(best_n)       
                n = best_n 
                
        if len(path) == K:   
            all_paths.append((path, -np.log10(comp_lrt(df, path))))

    path_greedy_best = max(all_paths, key = lambda x:x[1])   
    return(path_greedy_best)


# =============================================================================
# Jaccard similarity    
# =============================================================================
def jaccard(x,y):
    inter = set.intersection(*[set(x), set(y)])
    union = set.union(*[set(x), set(y)])
    return len(inter)/len(union)

# =============================================================================
# Pathway from node list to edge list (and viceversa)
# =============================================================================
def path_n2e(path):
# e.g. from [6122, 1017, 6164] to [(6122, 1017), (6122, 6164)]     
    return [(path[i],path[i+1]) for i in range(len(path)-1)]

def path_e2n(epath):
# e.g. from [(6122, 1017), (6122, 6164)] to [6122, 1017, 6164]    
    if len(epath) == 0:
        return []
    
    npath = list(epath[0])
    for i in range(1, len(epath)):
        npath.append(epath[i][1])
    return npath

# =============================================================================
# Heuristic information computation (p-gain)
# =============================================================================
       
def heuristic(i, j, gx, net, path, corr_mat, dict_pval):

    path_n = path_e2n(path)
    
    if len(path_n) == 0:    
        if tuple([i]) in dict_pval.keys():
            a = dict_pval[tuple([i])]
        else:
            a = comp_lrt(gx, [i])
            dict_pval[tuple([i])] = a
        if tuple([j]) in dict_pval.keys():
            b = dict_pval[tuple([j])]
        else:
            b = comp_lrt(gx, [j])
            dict_pval[tuple([j])] = b
        if tuple(sorted(tuple([i,j]))) in dict_pval.keys():
            c = dict_pval[tuple(sorted(tuple([i,j])))]
        else:
            c = comp_lrt(gx, [i,j])
            dict_pval[tuple(sorted(tuple([i,j])))] = c                
            
        weight = [a, b, c]
        pgain = np.min(weight[:2]) / weight[2]
        net[i][j]['weight'] = (1/abs(corr_mat.loc[str(i), str(j)]))*pgain
        
    else:
        if tuple(sorted(tuple(path_n))) in dict_pval.keys():
            pval1 = dict_pval[tuple(sorted(tuple(path_n)))]
        else:
            pval1 = comp_lrt(gx, path_n)
            dict_pval[tuple(sorted(tuple(path_n)))] = pval1
            
        if tuple(sorted(tuple(path_n+[j]))) in dict_pval.keys():
            pval2 = dict_pval[tuple(sorted(tuple(path_n+[j])))]
            
        else:
            pval2 = comp_lrt(gx, path_n+[j])   
            dict_pval[tuple(sorted(tuple(path_n+[j])))] = pval2
        
        combs = list(it.product(path_n, [j]))
        corr_max = max(abs(np.asarray([corr_mat.loc[str(combs[i][0]), str(combs[i][1])] for i in range(len(combs))])))
        net[i][j]['weight']  = (pval1/pval2)*(1/corr_max)  
        
# =============================================================================
# Jumping ants
# =============================================================================
def jump(alpha, beta, net, gx, path, corr_mat, dict_pval):
    neigh = []
    npath = path_e2n(path)

    for n in npath: # i'm going to consider all the nodes of such path and not just the last node
                    # Yes BUT JUST IF I DONT HAVE ANYMORE NEIGHS! (->wrong)
        #neigh += [e for e in net.edges(n) if not e[1] in npath]
        neigh += [e for e in net.edges(n) if ((e[1],e[0]) or (e[0], e[1])) not in path]
        
        for p in net.edges(n):            
            # heuristic information computation
            if "weight" not in net.get_edge_data(p[0],p[1]):
                heuristic(p[0], p[1], gx, net, path, corr_mat, dict_pval)
    
    scores = {e: pow(net.get_edge_data(*e)['pheromone'], alpha)*pow(net.get_edge_data(*e)['weight'], beta) for e in neigh}
    denom = sum(scores.values())
    probs = np.array([scores[e]/denom for e in neigh])
    
    try:
        t = neigh[np.random.choice(len(neigh), 1, p=probs)[0]]
    except ValueError:  # due to a python bug probabilities might not sum to 1 exactly
        t = neigh[np.random.choice(len(neigh), 1)[0]]
    return t
    

# =============================================================================
# Tour costruction
# =============================================================================
def walk(alpha, beta, net, gx, K, corr_mat, dict_pval):
    
    #scores = {n:np.sum([w["pheromone"] for u,v,w in net.edges(n, data=True)]) for n in net.nodes()}
    #p = (1-np.array(probs))*np.array(list(scores.values()))
    scores = {n: net.degree(n, 'pheromone')/net.degree(n) for n in net.nodes()} #sum of the pheromone on the edges of that node/degree
        
    denom = sum(scores.values())
    p = np.array([scores[n]/denom for n in net.nodes()])

    try:
        start = np.random.choice(net.nodes(), 1, p=p)[0]
    except ValueError:  # due to a python bug probabilities might not sum to 1 exactly
        start = np.random.choice(net.nodes(), 1)[0]
    
    
    path = []
    #for j in range(K-1):
    while len(path_e2n(path))<K:
        if len(path) == 0:
            neigh = [e for e in net.edges(start)]
        else:
            neigh = [e for i in path_e2n(path) for e in net.edges(i) if tuple((e[0],e[1])) not in path and tuple((e[1],e[0])) not in path]
        
        if len(neigh) > 0:
            for n in neigh:
                # Heuristic information computation
                if "weight" not in net.get_edge_data(n[0],n[1]):
                                   
                    #tra n[1] e tutti gli altri nel path
                    heuristic(n[0], n[1], gx, net, path, corr_mat, dict_pval)
                
            # Computation of transition probability and selection of the next node
            scores = {e: pow(net[e[0]][e[1]]['pheromone'], alpha) * pow(net[e[0]][e[1]]['weight'], beta) for e in neigh}
            denom = sum(scores.values())
            probs = np.array([scores[e]/denom for e in neigh])
            
            try:
                append_ = neigh[np.random.choice(len(neigh), 1, p=probs)[0]]
                path.append(append_)               
            except ValueError:
                append_ = neigh[np.random.choice(len(neigh), 1)[0]]
                path.append(append_)
        else:
            # nel caso in cui ho già esplorato tutti i neighbors
            append_ = jump(alpha, beta, net, gx, path, corr_mat, dict_pval)
            path.append(append_)
    return (path)

# =============================================================================
#  ––––––––––––––––––––––––––– EXTRACT SUBNET FROM REAL DATA ––––––––––––––––––
# =============================================================================

# Extraction subnetwork
def extract(G, start, size):
    
    size_sample = int(size/2)
    nodes = []
    
    for i in range(size_sample+1):
        next_ = random.choice(list(G.neighbors(start)))
        nodes.append(next_)
        start = next_
    alll = []
    for i in nodes:
        
        alll.append(list(G.neighbors(i)))
    
    alll = sum(alll, [])

    neigh = random.sample(alll, size_sample)
    for n in neigh:
        nodes.append(n)    
    return(list(set(nodes)))

def check(list_, same):     
    if len(list_) < (same) or len(list_) == (same):
        return True
    elif len(list_) > (same) and len(set([l for l in list_[-same:]])) != 1:
        return True
    elif len(list_) > (same) and len(set([l for l in list_[-same:]])) == 1:
        return False

#%%
# =============================================================================
#  Max-MinAntSystem
# =============================================================================
def MaxMin_AS(params, datatype, save):
    
    start_time = time()
    
    if datatype == "simulated":
        alpha, beta, rho, gx, net, greedy, best_theo, ants, K, corr_mat = params
    elif datatype == "real":
        alpha, beta, rho, gx, net, ants, K, corr_mat = params
        
    # =========================================================================#
    # alpha:     regulate the importance of the pheromone                      #
    # beta:      regulate the importance of the heuristic information          #
    #                                                                          #
    # ph_min:    minimum value that pheromone can assume                       #
    # ph_max:    maximum value that pheromone can assume                       #
    #                                                                          #
    # gx:        gene expression matrix                                        #
    #             (with survival time and censor data columns)                 #
    # net:       PPI network                                                   #
    #                                                                          #
    # greedy:    greedy solution                                               #
    # best_theo: best theoretical solution                                     #
    #                                                                          #
    # ant:       number of ants in the colony                                  #
    # K:         desired subnetwork size                                       #
    # trials:    maximum number of trials                                      #
    #                                                                          #  
    # sim:       current simulation                                            #
    # =========================================================================#
    dict_pval = {}
    for n, w in net.nodes(data=True):
        dict_pval[tuple([n])]=w["weight"]
        
    run=0    
    best_per_run = []
    
    max_allowed = 20 # 30        #max number of best rturned allowed 
    same_in_a_row = 15 # 10     #20 #number of identical consecutive values before to stop (it means that after 20 iter is considered stucked in local optima)
    
    antRes_per_run = {}

    while check(sorted(best_per_run, reverse=True), max_allowed):   #quando il best non cambia più per 5 volte, mi fermo

        iter_ = 0        
        best_per_iter = []
        info_per_iter = []        
        antRes_per_iter = {}
        
        # RESET PARAMETERS ================================================
        # Pheromone initialization     
        for u, v, w in net.edges(data=True):
            w['pheromone'] = 1
        P = {0: nx.attr_matrix(net, 'pheromone')[0]}        
        best_global = 1 
        ph_max=1
        ph_min=0

        while check(best_per_iter, same_in_a_row):   

            # CALCOLO PATHs + UPDATE VALUES + COMPUTE VALUES ==================
            # computing all the pathways
            antPathway = {i:walk(alpha, beta, net, gx, K, corr_mat, dict_pval) for i in range(ants)}
        
            # Decay pheromone over ALL the edges of the network
            for e in net.edges():                
                net[e[0]][e[1]]['pheromone'] = (1-rho)*net[e[0]][e[1]]['pheromone']
                
            # computing the pval of the pathway crossed by the ants
            pathways = []
            for k,v in antPathway.items():
                
                if tuple(sorted(tuple(path_e2n(v)))) in dict_pval.keys():
                    pathways.append((v, dict_pval[tuple(sorted(tuple(path_e2n(v))))]))
                else:
                    compute = comp_lrt(gx, path_e2n(v))
                    dict_pval[tuple(sorted(tuple(path_e2n(v))))] = compute
                    pathways.append((v, compute))
                    
            """
            # NON MMAS =================================================================================
            # update pheromone over all crossed pathways
            for p, s in pathways:
                upd = np.sum([net[i[0]][i[1]]['pheromone']  for i in p])
                for i in p:
                    net[i[0]][i[1]]['pheromone'] = net[i[0]][i[1]]['pheromone'] + upd #-np.log10(s)/len(p)

            """
            # MMAS =================================================================================
            # update pheromone only over best pathways
            sort_by_score = sorted(pathways, key=lambda x:x[1])                   
            # best = [(bestpath1, score), (bestpath2, score)...]
            best = [sort_by_score[i] for i in range(len(sort_by_score)) if sort_by_score[i][1] == sort_by_score[0][1]]
            
            # note: there are duplicate pathways, since might happen that more that 1 ant go through the same path ->this means updating ph on the
            # same path more than once
            best_pval = best[0][1]
            just_paths = sorted(best, key=lambda x:x[1])
            just_best_paths = [just_paths[i][0] for i in range(len(just_paths)) if just_paths[i][1] == just_paths[0][1]]
            set_paths=[sorted(tuple(sorted(i)) for i in just_best_paths[j]) for j in range(len(just_best_paths))]
            path_to_update = [list(i) for i in set(map(tuple, set_paths))]

            # Update ph_max (maximum limit pheromone) -> alzo la soglia con il miglior pval      
            if best_pval < best_global:                                           
                best_global = best_pval
                ph_max = -np.log10(best_global)
                
            for p in path_to_update:
                #upd = np.sum([net[i[0]][i[1]]['pheromone']  for i in p])
                for i in p:
                     new_ph = net[i[0]][i[1]]['pheromone'] + -np.log10(best_pval)#/len(p)  #?? upd/len(p) 
                     
                     # apply Max/Min
                     if new_ph > ph_max:
                         net[i[0]][i[1]]['pheromone'] = ph_max
                     elif new_ph < ph_min:
                         net[i[0]][i[1]]['pheromone'] = ph_min
                     else:    
                         net[i[0]][i[1]]['pheromone'] = new_ph
            # =================================================================================

            P[k+1] = nx.attr_matrix(net, 'pheromone')[0]

            # SAVING THE NEEDED INFO ==========================================
            
            antRes_per_iter[iter_] = {i:pathways[i] for i in range(len(pathways))}
            
            # best pval among all ants for that iteration
            best_pval_by_now = min(list(antRes_per_iter[iter_].values()), key=itemgetter(1))[1]
            best_path_by_now = min(list(antRes_per_iter[iter_].values()), key=itemgetter(1))[0]
            
            best_per_iter.append(best_pval_by_now)   # e.g.:([(64, 7), (7, 47), (7, 5)], 0.0182, 13)
            info_per_iter.append((best_pval_by_now, antRes_per_iter[iter_], best_path_by_now))
            
            #print("allora-------------------------------\n")
            #print(best_per_iter)
            iter_ += 1
            
        antRes_per_run[run] = antRes_per_iter
        
        
        save1 = [(k,min(list(v2.values()), key=itemgetter(1))[1]) for k,v in antRes_per_run.items() for k2,v2 in v.items()]
        best_per_run.append(min(save1, key=itemgetter(1))[1])
        
        # list of best values of each restart
        # save more info related to such best
#        info_per_run.append((min(info_per_iter, key=itemgetter(0))[0], min(info_per_iter, key=itemgetter(0))[1], min(info_per_iter, key=itemgetter(0))[2]))      

        run += 1
    # from the best solution i get: num of iterations done, pvalue, and list of best value at each iteration (for plotting)

    save2 = [(k,min(list(v2.values()), key=itemgetter(1))[1]) for k,v in antRes_per_run.items() for k2,v2 in v.items()]
    solution = antRes_per_run[min(save2, key=itemgetter(1))[0]]
    
    scores = [[-np.log10(solution[i][j][1]) for j in range(ants)] for i in range(len(solution.keys()))]
    max_ = [np.max(s) for s in scores]
    avg = [np.mean(s) for s in scores]
 
    plt.figure(figsize = (15,13))
    fig, ax = plt.subplots()    
    ax.axhline(y=np.max(max_), label='best returned', c = "lightgreen", linewidth=2, ls="-.")
    plt.plot(max_, ls='--', label='best per iter', linewidth=2, c = "dodgerblue")  
    plt.plot(avg, ls='--', label='avg per iter', linewidth=2, c="hotpink")  
    ax.axhline(y=best_theo[0][1], c='k', label='theoretical best', linewidth=2)
    ax.axhline(y=greedy[1], label='greedy', c = "darkorange", linewidth=2)
    plt.ylabel("-log10(P-value)", fontsize = 12)
    plt.xlabel("Iterations", fontsize = 12)
    plt.grid(c = "lightgrey")
    plt.title("Algorithm Convergence", fontsize = 15)
    plt.legend(bbox_to_anchor=(1,1), ncol = 1, prop={'size': 13}, fontsize = 12)
    plt.show()
        
    extract_bests = sorted([v2 for v in solution.values() for v2 in v.values()], key=lambda x:x[1])
    # list of best paths with corresponding pval (but a same path can be repeated)
    all_bests = [extract_bests[i] for i in range(len(extract_bests)) if extract_bests[i][1] == extract_bests[0][1]]
    
    print("\n=== Results ===")
    print("*SOLUTION:", all_bests[0][0])
    print("*[-log10]:", -np.log10(all_bests[0][1]))    
    
    end = time()
    time_ = end - start_time
    print("*Time:    ", round(time_, 3))
    #return(all_bests[0][1])
    return(max_)
    
    
    
    
