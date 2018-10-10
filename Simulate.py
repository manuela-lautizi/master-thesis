#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: manuelalautizi
"""

import pandas as pd
import numpy as np
import networkx as nx

import os
os.chdir('...')

# =============================================================================
# SIMULATED DATA: (gx, net)
# 1) Gene Expression, Survival data, censor data
# 2) PPI Network
# =============================================================================
M = 50
N = 20
np.random.seed(1234)

# 1) Gene Expression
gx = pd.DataFrame({i: np.random.gamma(5, 1, size=N) for i in range(M)})

# 2) Survival time, Censor data
gx['os_time'] =  np.random.exponential(1.5, size = N) 
gx['os_event'] = np.random.choice([0,1], N, p=[.3,.7])

# 2) PPI Network
net = nx.Graph()
net.add_edges_from(nx.scale_free_graph(M, alpha=0.41, beta=0.54, gamma=0.05, delta_in=0.2).to_undirected().edges())
net.remove_edges_from(net.selfloop_edges())

