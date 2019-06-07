#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: manuelalautizi
"""
import os
os.chdir('...')

import pandas as pd
import numpy as np
import networkx as nx
import random
import Library_MMAS as MMAS 
from importlib import reload
reload(MMAS)

# =============================================================================
# Simulated data
# =============================================================================
# Adjacency matrix simulated network
sim_net = pd.read_csv(".../netSIM.csv")
# Matrix with simulated gene expression, survival time and censor data
sim_gx = pd.read_csv(".../gxSIM.csv")

sim = nx.Graph()

for i in sim_net.columns.tolist():
    for j in sim_net.columns.tolist():
        if sim_net.loc[int(i), j] > 0.00000:
            sim.add_edge(int(i),int(j))

#Parameters
ants = 30
trials = 30 
K = 4
# Hyperparameters:
alpha  = 1 
beta   = 1
rho    = 0.3   
ph_min = 0
ph_max = 1 

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
    par_pca = alpha, beta, rho, ph_min, ph_max, sim_gx, sim_G, greedy_pca[1], best_theo_pca, ants, K, trials, i
    antRes_pca, time_pca, trials_pca, best_pca  = MMAS.MaxMin_AS_sim(par_pca, "pca")
    
    print("-------KMEANS-------")
    par_kmeans = alpha, beta, rho, ph_min, ph_max, sim_gx, sim_G, greedy_pca[1], best_theo_kmeans, ants, K, trials, i
    antRes_kmeans, time_kmeans, trials_kmeans, best_kmeans = MMAS.MaxMin_AS_sim(par_kmeans, "kmeans")
    
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
patients = pd.read_csv("...")
G = pd.read_csv("...")

# =============================================================================
# Application
# =============================================================================

#Parameters
ants = 60
trials = 60 
K = 10
# Hyperparameters:
alpha  = 1 
beta   = 1
rho    = 0.3   
ph_min = 0
ph_max = 1 

simulations = 10
for i in range(simulations):      
    Graph = G.copy()
    param = alpha, beta, rho, ph_min, ph_max, patients, Graph, ants, K, trials, i
    antRes_pca, time_pca, trials_pca, best_pca  = MMAS.MaxMin_AS(param, "pca")


# =============================================================================
# If evaluation on a sample of the real data:
# Extraction subnetwork
# =============================================================================
size = 500
start = random.choice(G.nodes())

nodes = MMAS.extract(G, start, size)
subG = G.subgraph(nodes)
nx.draw(subG)

subPatients = patients[[i for i in map(str, nodes)] + ['os_time', 'os_event']]


