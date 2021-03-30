# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 16:12:33 2021

@author: willi
"""

import numpy as np
import os
import rsnapsim as rss
import time
print('utilizing rsnapsim version:')
print(rss.__version__)

p_str, p_obj, p_tagged, seq = rss.seqmanip.open_seq_file('./Bactin_withTags.txt')

poi = p_tagged['1'][0]  #protein object
solver = rss.solver  #solver class
solver.protein=poi


strGeneCopy = {'TTT': 17.6, 'TCT': 15.2, 'TAT': 12.2, 'TGT': 10.6, 'TTC': 20.3,
                            'TCC': 17.7, 'TAC': 15.3, 'TGC': 12.6, 'TTA': 7.7, 'TCA': 12.2,
                            'TAA': 1.0, 'TGA': 1.6, 'TTG': 12.9, 'TCG':  4.4, 'TAG': 0.8,
                            'TGG': 13.2, 'CTT': 13.2, 'CCT': 17.5, 'CAT': 10.9, 'CGT': 4.5,
                            'CTC': 19.6, 'CCC': 19.8, 'CAC': 15.1, 'CGC': 10.4, 'CTA':  7.2,
                            'CCA': 16.9, 'CAA': 12.3, 'CGA':  6.2, 'CTG': 39.6, 'CCG':  6.9,
                            'CAG': 34.2, 'CGG': 11.4, 'ATT': 16.0, 'ACT': 13.1, 'AAT': 17.0,
                            'AGT': 12.1, 'ATC': 20.8, 'ACC': 18.9, 'AAC': 19.1, 'AGC': 19.5,
                            'ATA':  7.5, 'ACA': 15.1, 'AAA': 24.4, 'AGA': 12.2, 'ATG': 22.0,
                            'ACG': 6.1, 'AAG': 31.9, 'AGG': 12.0, 'GTT': 11.0, 'GCT': 18.4,
                            'GAT': 21.8, 'GGT': 10.8, 'GTC': 14.5, 'GCC': 27.7, 'GAC': 25.1,
                            'GGC': 22.2, 'GTA':  7.1, 'GCA': 15.8, 'GAA': 29.0, 'GGA': 16.5,
                            'GTG': 28.1, 'GCG': 7.4, 'GAG': 39.6, 'GGG': 16.5}

mean_tRNA_copynumber = np.mean(list(strGeneCopy.values()))
strGeneCopy.pop('TAG')
strGeneCopy.pop('TAA')
strGeneCopy.pop('TGA')


k_diff = 10
k_trna = np.array([strGeneCopy[rss.cdict.trna_ids[x]] for x in range(0,61)])*k_diff
k_bind = .033
kelong = 1
k_compl = 10
k_index = np.array(poi.ktrna_id)

t = np.linspace(0,500,501)
st = time.time()
ssa_soln3 = solver.solve_ssa_trna(k_index,  k_diff,k_bind,kelong,k_compl ,t,n_traj=1, k_trna=k_trna)
print('time for {0} trajectories {1}'.format(1,time.time()-st))