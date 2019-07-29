#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: manuelalautizi
"""

# =============================================================================
# IMPORT PACKAGES
# =============================================================================
import pandas as pd
import numpy as np
from lifelines.statistics import logrank_test
from sklearn.cluster import KMeans
import networkx as nx
#import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt
#plt.use('tkagg')
from matplotlib.backends.backend_pdf import PdfPages
from time import time
from itertools import chain
from toolz import unique
import itertools as it
import random
from sklearn.decomposition import PCA

# =============================================================================
# FUNCTIONS
# =============================================================================

# =============================================================================
# ObjectiveFunction (computed by using K-Means/PCA)
# =============================================================================
def comp_lrt_kmeans(df, fts):
    fts = [i for i in map(str, fts)]
    km = KMeans(2, random_state=1234) 
    km.fit(df[fts])
    pop = df[['os_time', 'os_event'] + list(fts)].copy()
    pop['y'] = km.predict(df[fts])
    pop = pop.groupby('y')
    lrt = logrank_test(pop.get_group(0)['os_time'], pop.get_group(1)['os_time'], 
                       pop.get_group(0)['os_event'], pop.get_group(1)['os_event'])
    return(lrt.p_value)

def comp_lrt_pca(df, fts):
    fts = [i for i in map(str, fts)]
    if len(fts) == 1:
        df = df[fts + ['os_time', 'os_event']]
        P0 = df[df.loc[:, fts[0]] >= np.median(df[fts])]
        P1 = df[df.loc[:, fts[0]] < np.median(df[fts])]
        
        lrt = logrank_test(P0['os_time'], P1['os_time'], P0['os_event'], P1['os_event'])
        return(lrt.p_value)
        
    else:
        pca = PCA(n_components = 2, random_state=1234)
        
        principalComponents = pca.fit_transform(df[fts])
        principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
        
        new = pd.concat([df[fts + ['os_time', 'os_event']], principalDf.iloc[:,0]], axis=1)
        new = new.sort_values('principal component 1')
        
        
        P0 = new[new['principal component 1'] > 0]
        P1 = new[new['principal component 1'] <= 0]
        
        lrt = logrank_test(P0['os_time'], P1['os_time'], P0['os_event'], P1['os_event'])
        
        return(lrt.p_value)

# =============================================================================
# Computation Best Theoretical Solution & Greedy Solution
# =============================================================================
def best_theo(G, df, K, alg):    
    
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

    #[i.sort() for i in all_paths]
    all_paths  = list(map(list, unique(map(tuple, all_paths))))
     
    if alg == "kmeans":
        scores_theo = sorted([(p,-np.log10(comp_lrt_kmeans(df, p))) for p in all_paths], key=lambda x:x[1], reverse=True)
        best_theo = [scores_theo[i] for i in range(len(scores_theo)) if scores_theo[i][1] == scores_theo[0][1]]
        
    elif alg == "pca":
        scores_theo = sorted([(p,-np.log10(comp_lrt_pca(df, p))) for p in all_paths], key=lambda x:x[1], reverse=True)
        best_theo = [scores_theo[i] for i in range(len(scores_theo)) if scores_theo[i][1] == scores_theo[0][1]]
        
    else:
        print("Chose an algorithm among pca or kmeans")
        
    return (best_theo)

# =============================================================================
#  Greedy
# =============================================================================    
def greedy_algorithm(df, G, K, alg):

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
            if alg == "pca":
                all_paths.append((path, -np.log10(comp_lrt_pca(df, path))))
            elif alg == "kmeans":
                all_paths.append((path, -np.log10(comp_lrt_kmeans(df, path))))
            else:
                print("Chose an algorithm typing pca or kmeans")

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
def heuristic(i, j, gx, net, alg, path, corr_mat):
    if alg == "pca":
        weight = [comp_lrt_pca(gx, fts) for fts in [[i], [j], [i, j]]]
        if len(path) == 0:
            net[i][j]['weight']  = (np.min(weight[:2]) / weight[2])/corr_mat.loc[str(i), str(j)]            
        else:
            combs = list(it.product(path_e2n(path), [j]))
            corr_max = max(abs(np.asarray([corr_mat.loc[str(combs[i][0]), str(combs[i][1])] for i in range(len(combs))])))
            net[i][j]['weight']  = (np.min(weight[:2]) / weight[2])/corr_max
    
    if alg == "kmeans":
        weight = [comp_lrt_kmeans(gx, fts) for fts in [[i], [j], [i, j]]]
        net[i][j]['weight']  = np.min(weight[:2]) / weight[2] 
        
# =============================================================================
# Jumping ants
# =============================================================================
def jump(alpha, beta, net, gx, path, alg, corr_mat):
    neigh = []
    npath = path_e2n(path)
    
    for n in npath: # i'm going to consider all the nodes of such path and not just the last node
        neigh += [e for e in net.edges(n) if not e[1] in npath]
        
        for p in net.edges(n):            
            # heuristic information computation
            if "weight" not in net.get_edge_data(p[0],p[1]):
                heuristic(p[0], p[1], gx, net, alg, path, corr_mat)
    
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
def walk(alpha, beta, net, gx, K, alg, corr_mat):
    # Selection of the starting node
    scores = {n: net.degree(n, 'pheromone')/net.degree(n) for n in net.nodes()} #sum of the pheromone on the edges of that node/degree
    denom = sum(scores.values())
    probs = np.array([scores[n]/denom for n in net.nodes()])
    
    try:
        s = np.random.choice(net.nodes(), 1, p=probs)[0]
    except ValueError:  # due to a python bug probabilities might not sum to 1 exactly
        s = np.random.choice(net.nodes(), 1)[0]
    path = []
    
    for j in range(K-1):
        neigh = [e for e in net.edges(s) if not e[1] in path_e2n(path)]

        if len(neigh) > 0:
            for n in neigh:
                # Heuristic information computation
                if "weight" not in net.get_edge_data(n[0],n[1]):
                                   
                    #tra n[1] e tutti gli altri nel path
                    heuristic(n[0], n[1], gx, net, alg, path_e2n(path), corr_mat)
    
            # Computation of transition probability and selection of the next node
            scores = {e: pow(net[e[0]][e[1]]['pheromone'], alpha) * pow(net[e[0]][e[1]]['weight'], beta) for e in neigh}
            denom = sum(scores.values())
            probs = np.array([scores[e]/denom for e in neigh])
            
            try:
                path.append(neigh[np.random.choice(len(neigh), 1, p=probs)[0]])
            except ValueError:
                path.append(neigh[np.random.choice(len(neigh), 1)[0]])
        else:
            # nel caso in cui ho già esplorato tutti i neighbors
            path.append(jump(alpha, beta, net, gx, path, alg, corr_mat))
    return (path)

# =============================================================================
#  Max-MinAntSystem for simulated data 
# =============================================================================
def MaxMin_AS_sim(params, alg):
    start = time()
    
    alpha, beta, rho, ph_min, ph_max, gx, net, greedy, best_theo, ants, K, trials, sim, corr_mat = params
    
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
  
    W = nx.attr_matrix(net, 'weight')[0]   

    # Pheromone initialization     
    for u, v, w in net.edges(data=True):
        w['pheromone'] = 1
        
    P = {0: nx.attr_matrix(net, 'pheromone')[0]}
    
    antRes = {i:{} for i in range(trials)}
    trials_done = 0
    best_global = 0
    
    all_mean = []   
    for trial in range(trials):
        antPathway = {i:walk(alpha, beta, net, gx, K, alg, corr_mat) for i in range(ants)}
    
        # Decay pheromone 
        for e in net.edges():                
            net[e[0]][e[1]]['pheromone'] = (1-rho)*net[e[0]][e[1]]['pheromone']
            
        # Update pheromone:
        best_pval = 0
        best_path = []
        for ant in range(ants):
            
            path = list(set(list(chain.from_iterable(antPathway[ant]))))  
            path2 = antPathway[ant]
            
            if alg == "pca":
                pval = -np.log10(comp_lrt_pca(gx, path))
                
            elif alg == "kmeans":
                pval = -np.log10(comp_lrt_kmeans(gx, path))
                
            if pval> best_pval:
                best_pval = pval
                best_path = path2
                
            antRes[trial][ant] = (antPathway[ant], pval)
        
        # Update maximum limit pheromone
        if trial == 0:
            best_global = best_pval
        else:
            if best_pval > best_global:
                best_global = best_pval
                ph_max = best_pval

        # Stopping Criteria
        for i in best_path:
             new_ph = net[i[0]][i[1]]['pheromone'] + best_pval/len(best_path)
             if new_ph > ph_max:
                 net[i[0]][i[1]]['pheromone'] = ph_max
             elif new_ph < ph_min:
                 net[i[0]][i[1]]['pheromone'] = ph_min
             else:
                 net[i[0]][i[1]]['pheromone'] = new_ph
        P[trial+1] = nx.attr_matrix(net, 'pheromone')[0]
        
        media = np.mean([antRes[trial][j][1] for j in range(ants)])
        all_mean.append(media)       
        
        trials_done += 1
        
        if len(all_mean) > 10 and abs(np.mean(all_mean[-5:])- np.mean(all_mean[-10:-5]))<0.03: 
            break   

    end = time()
    time_ = end - start
    print("time in seconds", time_)
        
    trials = trials_done    
    antRes = {k:v for k,v in list(antRes.items())[:trials_done]}
    
    # =============================================================================
    # Plotting and Saving results
    # =============================================================================
    # os.chdir("...")       387032
    
    fig, ax = plt.subplots()
    scores = [[antRes[i][j][1] for j in range(ants)] for i in range(trials)]
    best = [np.max(s) for s in scores]
    avg = [np.mean(s) for s in scores]
       
    ax.axhline(y=best_theo[0][1], c='k', label='theoretical best', linewidth=2)
    ax.plot(range(trials), best, ls='--', label='best -log10(p-value)', c = "hotpink", linewidth=2)
    ax.plot(range(trials), avg, ls='--', label='avg -log10(p-value)', c = "dodgerblue", linewidth=2)
    ax.axhline(y=greedy, label='greedy', c = "darkorange", linewidth=2)
    
    plt.ylabel("-log10(P-value)", fontsize = 12)
    plt.xlabel("Iterations", fontsize = 12)
    plt.xlim(0,trials)
    plt.grid(c = "lightgrey")
    plt.title("Algorithm Convergence", fontsize = 15)
    plt.legend(bbox_to_anchor=(1,1), ncol = 1, prop={'size': 13}, fontsize = 12)
    plt.show()
    
    
    with PdfPages('2_Convergence_%i_%i_%i.pdf'%(K, trials, sim)) as pdf:
        fig, ax = plt.subplots()
        scores = [[antRes[i][j][1] for j in range(ants)] for i in range(trials)]
        best = [np.max(s) for s in scores]
        avg = [np.mean(s) for s in scores]
           
        ax.axhline(y=best_theo[0][1], c='k', label='best theoretical', linewidth=2)
        ax.plot(range(trials), best, ls='--', label='best -log10(p-value)', c = "hotpink", linewidth=2)
        ax.plot(range(trials), avg, ls='--', label='avg -log10(p-value)', c = "dodgerblue", linewidth=2)
        ax.axhline(y=greedy, label='greedy', c = "darkorange", linewidth=2)
        
        plt.ylabel("-log10(P-value)", fontsize = 12)
        plt.xlabel("Iterations", fontsize = 12)
        plt.xlim(0,trials)
        plt.grid(c = "lightgrey")
        plt.title("Algorithm Convergence (PCA)", fontsize = 15)
        plt.legend(bbox_to_anchor=(1,1), ncol = 1, prop={'size': 13}, fontsize = 12)
        pdf.savefig()
        plt.close()

        res_df = pd.DataFrame(columns=[e for sl in [['ant%i_path'%i,'ant%i_score'%i] for i in range(ants)] for e in sl])
        for i in range(trials):
            res_df.loc[i,:] = [e for sl in [[antRes[i][j][0],antRes[i][j][1]] for j in range(ants)] for e in sl]
            
        res_df.to_csv('2_antpaths_%i_%i_%i.txt'%(K, trials, sim), sep='\t')
        theo_df = pd.DataFrame(columns=['path', 'score'])
        for i in range(len(best_theo)):
            theo_df.loc[i,:] = best_theo[i]
        theo_df.to_csv('2-best_%i_%i_%i.txt'%(K, trials, sim), sep='\t')
        pd.DataFrame(W).to_csv('2_netEdges_%i_%i_%i.txt'%(K, trials, sim), sep='\t')
        
        
        best_solution = max(list(antRes[trials-1].values()), key =  lambda x:x[1])
    return(antRes, time_, trials, best_solution)


# =============================================================================
#  ––––––––––––––––––––––––––– FUNCTIONS FOR REAL DATA ––––––––––––––––––––––––
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

    
# =============================================================================
#  Max-MinAntSystem for real data
# =============================================================================
def MaxMin_AS(params, alg):
    start = time()
    
    alpha, beta, rho, ph_min, ph_max, gx, net, ants, K, trials, sim, corr_mat = params
    
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
    # ant:       number of ants in the colony                                  #
    # K:         desired subnetwork size                                       #
    # trials:    maximum number of trials                                      #
    #                                                                          #  
    # sim:       current simulation                                            #
    # =========================================================================#
  
    # Pheromone initialization     
    for u, v, w in net.edges(data=True):
        w['pheromone'] = 1
        
    P = {0: nx.attr_matrix(net, 'pheromone')[0]}
    
    antRes = {i:{} for i in range(trials)}
    trials_done = 0
    best_global = 0
    
    all_mean = []   
    for trial in range(trials):
        antPathway = {i:walk(alpha, beta, net, gx, K, alg, corr_mat) for i in range(ants)}
        # Decay pheromone 
        for e in net.edges():                
            net[e[0]][e[1]]['pheromone'] = (1-rho)*net[e[0]][e[1]]['pheromone']
            
        # Update pheromone:
        best_pval = 0
        best_path = []
        for ant in range(ants):
            
            path = list(set(list(chain.from_iterable(antPathway[ant]))))  
            path2 = antPathway[ant]
            
            if alg == "pca":
                pval = -np.log10(comp_lrt_pca(gx, path))
                
            elif alg == "kmeans":
                pval = -np.log10(comp_lrt_kmeans(gx, path))
                
            if pval> best_pval:
                best_pval = pval
                best_path = path2
                
            antRes[trial][ant] = (antPathway[ant], pval)
        
        # Update maximum limit pheromone
        if trial == 0:
            best_global = best_pval
        else:
            if best_pval > best_global:
                best_global = best_pval
                ph_max = best_pval

        # Stopping Criteria
        for i in best_path:
             new_ph = net[i[0]][i[1]]['pheromone'] + best_pval/len(best_path)
             if new_ph > ph_max:
                 net[i[0]][i[1]]['pheromone'] = ph_max
             elif new_ph < ph_min:
                 net[i[0]][i[1]]['pheromone'] = ph_min
             else:
                 net[i[0]][i[1]]['pheromone'] = new_ph
        P[trial+1] = nx.attr_matrix(net, 'pheromone')[0]
        
        media = np.mean([antRes[trial][j][1] for j in range(ants)])
        all_mean.append(media)           
        
        trials_done += 1

        
        if len(all_mean) > 10 and abs(np.mean(all_mean[-5:])- np.mean(all_mean[-10:-5]))<0.03: 
            break   
    
    end = time()
    time_ = end - start
    print("time in seconds", time_)
        
    trials = trials_done    
    antRes = {k:v for k,v in list(antRes.items())[:trials_done]}
    
    # =============================================================================
    # Plotting and Saving results
    # =============================================================================
    # os.chdir("...")       
    
    fig, ax = plt.subplots()
    scores = [[antRes[i][j][1] for j in range(ants)] for i in range(trials)]
    best = [np.max(s) for s in scores]
    avg = [np.mean(s) for s in scores]
       
    ax.plot(range(trials), best, ls='--', label='best -log10(p-value)', c = "hotpink", linewidth=2)
    ax.plot(range(trials), avg, ls='--', label='avg -log10(p-value)', c = "dodgerblue", linewidth=2)
    
    plt.ylabel("-log10(P-value)", fontsize = 12)
    plt.xlabel("Iterations", fontsize = 12)
    plt.xlim(0,trials)
    plt.grid(c = "lightgrey")
    plt.title("Algorithm Convergence", fontsize = 15)
    plt.legend(bbox_to_anchor=(1,1), ncol = 1, prop={'size': 13}, fontsize = 12)
    plt.show()
    
    
    with PdfPages('2_Convergence_%i_%i_%i.pdf'%(K, trials, sim)) as pdf:
        fig, ax = plt.subplots()
        scores = [[antRes[i][j][1] for j in range(ants)] for i in range(trials)]
        best = [np.max(s) for s in scores]
        avg = [np.mean(s) for s in scores]
           
        ax.plot(range(trials), best, ls='--', label='best -log10(p-value)', c = "hotpink", linewidth=2)
        ax.plot(range(trials), avg, ls='--', label='avg -log10(p-value)', c = "dodgerblue", linewidth=2)
        
        plt.ylabel("-log10(P-value)", fontsize = 12)
        plt.xlabel("Iterations", fontsize = 12)
        plt.xlim(0,trials)
        plt.grid(c = "lightgrey")
        plt.title("Algorithm Convergence (PCA)", fontsize = 15)
        plt.legend(bbox_to_anchor=(1,1), ncol = 1, prop={'size': 13}, fontsize = 12)
        pdf.savefig()
        plt.close()

        res_df = pd.DataFrame(columns=[e for sl in [['ant%i_path'%i,'ant%i_score'%i] for i in range(ants)] for e in sl])
        for i in range(trials):
            res_df.loc[i,:] = [e for sl in [[antRes[i][j][0],antRes[i][j][1]] for j in range(ants)] for e in sl]
            
        res_df.to_csv('2_antpaths_%i_%i_%i.txt'%(K, trials, sim), sep='\t')
        
        best_solution = max(list(antRes[trials-1].values()), key =  lambda x:x[1])
    return(antRes, time_, trials, best_solution)
