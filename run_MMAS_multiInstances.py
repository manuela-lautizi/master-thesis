#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: manuelalautizi
profiler
"""

import os
os.chdir("/home/manuela/Desktop/master_thesis/")

import pandas as pd
import networkx as nx
import numpy as np
import library_MMAS_4 as MMAS_4  # ultimo
from time import time


instances = 1
to_plot = []

M = 300 # 50 genes
N = 20  # 20 patients
K = 5
np.random.seed(1234)

for i in range(instances):

    # 1) Gene Expression
    sim_gx = pd.DataFrame({i: np.random.gamma(5, 1, size=N) for i in range(M)})
    
    # 2) Survival time, Censor data
    sim_gx['os_time'] =  np.random.exponential(1.5, size = N) 
    sim_gx['os_event'] = np.random.choice([0,1], N, p=[.3,.7])
    
    sim_gx.columns = [str(sim_gx.columns.tolist()[i]) for i in range(len(sim_gx.columns)-2)]+["os_time", "os_event"]
    
    # 2) PPI Network
    sim_net = nx.Graph()
    sim_net.add_edges_from(nx.scale_free_graph(M, alpha=0.41, beta=0.54, gamma=0.05, delta_in=0.2).to_undirected().edges())
    self_loops = [(n, n) for n, nbrs in sim_net.adj.items() if n in nbrs]
    sim_net.remove_edges_from(self_loops)
      
    corr_mat = sim_gx.iloc[:,:-2].corr()
    # compute best theo and greedy:
    # assegno i pesi per calcolare i best theo e greedy
    net_pca = sim_net.copy()
    
    # assign weights to nodes
    for n,w in net_pca.nodes(data=True): 
        w['weight']  = MMAS_4.comp_lrt(sim_gx, [n])
    # assign weights to edges
    for u, v, w in net_pca.edges(data=True):                
        weight = MMAS_4.comp_lrt(sim_gx, [u,v])
        pgain = np.min([net_pca.nodes[u]['weight'], net_pca.nodes[v]['weight']]) / weight
        w['weight']  = (1/abs(corr_mat.loc[str(u), str(v)]))*pgain       

    best_theo = MMAS_4.best_theo(net_pca, sim_gx, K)     
    greedy = MMAS_4.greedy_algorithm(sim_gx, net_pca, K)
    print("best", best_theo)
    print("greedy", greedy)
    
#best [([130, 0, 106, 82, 85], 5.011918810938159), ([85, 82, 106, 0, 130], 5.011918810938159), ([0, 76, 82, 130], 5.011918810938159)]
#greedy ([85, 82, 106, 2, 182], 1.719407509657186)
    
#%%
import library_MMAS_4 as MMAS_4
ants = 30  
# Hyperparameters:
alpha  = 2 #1
beta   = 1
rho    = 0.4 #0.4 #0.3 
K=5
# provare co rho 0.6
toplot2=[]
times2=[]
for i in range(1):
    start_time = time()
    sim_G = sim_net.copy()
    param = alpha, beta, rho, sim_gx, sim_G, greedy, best_theo, ants, K, corr_mat
    best = MMAS_4.MaxMin_AS(param, datatype = "simulated", save = "no")        #best_sim, t, best
    toplot2.append(best)
    #print("best theo:", best_theo[0][1], "-returned:", np.max(best))
    end = time()
    time_ = end - start_time
    times2.append(time_)
#%%

import matplotlib
matplotlib.use('Agg')
%matplotlib inline
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

plt.figure(figsize = (15,13))
fig, ax = plt.subplots()
ax.axhline(y=best_theo[0][1], c='k', label='theoretical best', linewidth=2)
ax.axhline(y=greedy[1], label='greedy', c = "darkorange", linewidth=2)
plt.ylabel("-log10(P-value)", fontsize = 12)
plt.xlabel("Iterations", fontsize = 12)
plt.grid(c = "lightgrey")
plt.title("Algorithm Convergence", fontsize = 15)
plt.legend(bbox_to_anchor=(1,1), ncol = 1, prop={'size': 13}, fontsize = 12)

for sol in toplot2:#to_plot:    
    #ax.axhline(y=np.max(best), label='best returned', c = "lightgreen", linewidth=2, ls="-.")
    plt.plot(sol, ls='--', label='best per iter', linewidth=2)  

pd.DataFrame(nx.adjacency_matrix(sim_net).todense()).to_csv("/home/manuela/Desktop/master_thesis/sim_net_300N_5K.csv")
sim_gx.to_csv("/home/manuela/Desktop/master_thesis/sim_gx_300N_5K.csv")
#best_theo.to_csv("/home/manuela/Desktop/master_thesis/best_theo_instance_"+i+".txt")
#greedy.to_csv("/home/manuela/Desktop/master_thesis/greedy_instance_"+i+".txt")
#%%
# =============================================================================
# GRAPH
# =============================================================================

k = sim_net.subgraph([2,3,140,173])  
plt.figure(figsize=(6,6))
nx.draw_networkx(k, pos=nx.random_layout(net_pca))
plt.show()
#%%
import collections
degree_sequence = sorted([d for n, d in net_pca.degree()], reverse=True)  # degree sequence
# print "Degree sequence", degree_sequence
degreeCount = collections.Counter(degree_sequence)
deg, cnt = zip(*degreeCount.items())

plt.figure(figsize=(10,10))
plt.bar(deg, cnt, width=0.80, color='b')

plt.title("Degree Histogram")
plt.ylabel("Count")
plt.xlabel("Degree")

#%%
# =============================================================================================================================================================
# REAL DATA
# =============================================================================================================================================================
import itertools
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
    
corr_mat=pd.read_csv("corr_matrix.csv", index_col=0)
corr_mat_real = corr_mat #pd.read_csv("corr_matrix.csv", index_col=0)
corr_mat_real.index = [str(i) for i in corr_mat_real.index]


#%%
# =============================================================================
# RUN + PLOT
# =============================================================================
import library_MMAS_4 as MMAS_4
ants = 50  
# Hyperparameters:
alpha  = 3
beta   = 2
rho    = 0.5 
K=10

# preassegno dei valori
start_time = time()
net_ = G.copy()       
for n,w in net_.nodes(data=True): 
    w['pval']  = MMAS_4.comp_lrt(patients, [n])    
print(time() - start_time)

start_time = time()
param = alpha, beta, rho, patients, net_, ants, K, corr_mat_real
best = MMAS_4.MaxMin_AS(param, datatype = "real", save = "yes")        # best_sim, t, best
print(time() - start_time)






