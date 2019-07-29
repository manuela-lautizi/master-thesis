#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: manuelalautizi
profiler
"""
import os
os.chdir('/home/manuela/Desktop/master_thesis')

import pandas as pd
import numpy as np
import networkx as nx
import random
import library_MMAS as MMAS
import library_MMAS_2 as MMAS_corrMat 
from importlib import reload
reload(MMAS)

# =============================================================================
# Simulated data
# =============================================================================
# Adjacency matrix simulated network
sim_net = pd.read_csv("net_sim.csv", index_col = 0)
# Matrix with simulated gene expression, survival time and censor data
sim_gx = pd.read_csv("gx_sim.csv")

sim_net.columns = [int(i)+1 for i in sim_net.columns]
sim_net.index = [int(i)+1 for i in sim_net.index]
sim = nx.Graph()

for i in sim_net.columns:
    for j in sim_net.columns:
        if sim_net.loc[i, j] > 0:
            sim.add_edge(int(i),int(j))

# Parameters
ants = 30
trials = 30 
K = 4
# Hyperparameters:
alpha  = 1 
beta   = 1
rho    = 0.3   
ph_min = 0
ph_max = 1 

# correlation matrix
corr2 = sim_gx.iloc[:,:-2].corr()

# =============================================================================
# Assignment heuristic information to edges for the computation 
# of best theoretical solution
# =============================================================================
# PCA
# =============================================================================
net_pca = sim.copy()
for u, v, w in net_pca.edges(data=True):        
    pvals = [MMAS.comp_lrt_pca(sim_gx, fts) for fts in [[u], [v], [u,v]]]
    w['weight']  = np.min(pvals[:2]) / pvals[2] 


best_theo_pca = MMAS.best_theo(net_pca, sim_gx, K, alg = "pca")     
greedy_pca = MMAS.greedy_algorithm(sim_gx, net_pca, K, alg = "pca")
   
print("Best theo (PCA)", best_theo_pca)
print("Best greedy (PCA)", greedy_pca)

#best theo[([28, 1, 3, 23, 24], 3.122319911789083), ([24, 23, 3, 1, 28], 3.122319911789083)]

best_theo_pca = [([1, 14, 30, 41], 2.947679414568686)]
greedy_pca = ([17, 5, 3, 44], 0.9388180674292859)

# =============================================================================
# K-Means
# =============================================================================
net_kmeans = sim.copy()
for u, v, w in net_kmeans.edges(data=True):        
    pvals = [MMAS.comp_lrt_kmeans(sim_gx, fts) for fts in [[u], [v], [u,v]]]
    w['weight']  = np.min(pvals[:2]) / pvals[2] 

best_theo_kmeans = MMAS.best_theo(net_kmeans, sim_gx, K, alg = "kmeans")     
greedy_kmeans = MMAS.greedy_algorithm(sim_gx, net_kmeans, K, alg = "kmeans")
   
print("Best theo (kmeams)", best_theo_kmeans)
print("Best greedy (kmeans)", greedy_kmeans)


# =============================================================================
# Application
# =============================================================================
pca_time = []
kmeans_time = []

pca_iter = []
kmeans_iter = []

simulations = 10
for i in range(simulations):  
    
    sim_G = sim.copy()
    print("--------PCA--------")      
    par_pca = alpha, beta, rho, ph_min, ph_max, sim_gx, sim_G, greedy_pca[1], best_theo_pca, ants, K, trials, i, corr2
    antRes_pca, time_pca, trials_pca, best_pca = MMAS_corrMat.MaxMin_AS_sim(par_pca, "pca")
    print("best pca", best_pca)
    '''
    print("-------KMEANS-------")
    par_kmeans = alpha, beta, rho, ph_min, ph_max, sim_gx, sim_G, greedy_pca[1], best_theo_kmeans, ants, K, trials, i
    antRes_kmeans, time_kmeans, trials_kmeans, best_kmeans = MMAS.MaxMin_AS_sim(par_kmeans, "kmeans")
    '''
    pca_time.append(time_pca)
    pca_iter.append(trials_pca)    
    
    # Jaccard
    l = [best_pca[0][0][0]]
    for i in range(len(best_pca[0])):
        l.append(best_pca[0][i][1])        

    print("JACCARD", MMAS.jaccard(best_theo_pca[0][0], l)*100)
    
# =============================================================================
# Real Data
# =============================================================================
patients = pd.read_csv("GSE30219.tumor.os.entrez.csv", header = 0)
G = pd.read_csv("biogrid.human.entrez.tsv", sep = "\t", header = None)

#correlation matrix
corr_matrix = pd.read_csv("corr_matrix.csv", header = 0, index_col = 0)

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
# =============================================================================
# Application
# =============================================================================

#Parameters
ants = 60
trials = 10 
K = 5
# Hyperparameters:
alpha  = 1 
beta   = 1
rho    = 0.3   
ph_min = 0
ph_max = 1 

simulations = 1
for i in range(simulations):      
    Graph = G.copy()
    param = alpha, beta, rho, ph_min, ph_max, patients, Graph, ants, K, trials, i, corr_matrix
    antRes_pca, time_pca, trials_pca, best_pca  = MMAS.MaxMin_AS(param, "pca")


# =============================================================================
# If evaluation on a sample of the real data:
# Extraction subnetwork
# =============================================================================
size = 100
start = random.choice(list(G.nodes()))

nodes = MMAS.extract(G, start, size)
subG = G.subgraph(nodes)
nx.draw_networkx(subG)

subPatients = patients[[i for i in map(str, nodes)] + ['os_time', 'os_event']]

for i in range(simulations):      
    Graph = subG.copy()
    param = alpha, beta, rho, ph_min, ph_max, subPatients, Graph, ants, K, trials, i
    antRes_pca, time_pca, trials_pca, best_pca  = MMAS.MaxMin_AS(param, "pca")




