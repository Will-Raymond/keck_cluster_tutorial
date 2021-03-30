# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 16:41:29 2021

@author: willi
"""

import sys
import rsnapsim as rss
import numpy as np
import os


# This file calls the rsnapsim solver with some commandline inputs:
    # gene_file - string of gene to use
    # linspace args seperated by -, example: 0-1000-1001
    # ki - initation rate
    # n_traj - number of trajectories
    # save_dir - output dir for intensity files
    # save_name - name to save


gene_file = sys.argv[1]  
lin = [float(x) for x  in sys.argv[2].split('-')]
lin[-1] = int(lin[-1])
time = np.linspace(*lin)
ki = np.float(sys.argv[3])
n_traj = np.int(sys.argv[4])
save_dir = sys.argv[5]
save_name = sys.argv[6]


p_str, p_obj, p_tagged, seq = rss.seqmanip.open_seq_file(gene_file) #open up the gene file and make a protein obj
poi = p_tagged['1'][0]  #protein object
solver = rss.solver  #solver class
solver.protein=poi #set the solver protein object

ssa_soln_wt = solver.solve_ssa(poi.kelong ,time,ki=ki,n_traj=n_traj, low_memory=True, record_stats=True) #solve 

np.save(os.path.join(save_dir, save_name), ssa_soln_wt.I) #save intensity