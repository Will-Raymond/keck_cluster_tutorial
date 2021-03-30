# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 16:53:04 2021

@author: willi
"""

import subprocess
import time


gene_file = './Bactin_withTags.txt'
timevec_str = '0-1000-1001'
n_traj=15
save_dir = './Results/'

#######
# Lets get some explainations for what qsub args are doing here
#  -q munsky.q@node* | This says to use a particular node tag? idk the name
#                      the * at the end means any node
#  -cwd | use the current working directory from where this is run
#  -o /dev/null | this is the out file, here we are using /dev/null to 
#       autodelete this out file as its made, but it can be a regular file too.
#  -e Errs/ | Where to place error files if they are made
# -v | Here we get to enviroment variables, -v ki=.01 sets a local variable
#      accessed by the shell script called at the end, the variables
#      in the shell script must be identical, $ki is set by ki=.01
# python string parsing: '%f'%(.01) set a float into a string
#                        '%s'%(string) set a string in a string
#                        '%d'%(1) set an integer in a string
# all together these make the following command sent to subprocess.call:
# qsub -q munsky.q@node* -cwd -o /dev/null -e Errs/ -v  ki=0.010000 -v  ki=./Bactin_withTags.txt -v  time=0,1000,1001 -v  n_traj=15 -v  save_dir=./Results/ -v  save_name=Bactin_intensity
# this command is sent 3 times to different nodes with different ki.


for ki in [.01,.033,.08]:
    save_name=('Bactin_intensity_' + str(ki))
    
    cmd = ' '.join( ['qsub','-q munsky.q@node* -cwd -o /dev/null -e Errs/','-v','','ki=%f' % (ki),
                     '-v','','genefile=%s' % (gene_file),
                     '-v','','time=%s' % (timevec_str),
                     '-v','','n_traj=%d' % (n_traj),
                     '-v','','save_dir=%s' % (save_dir),
                     '-v','','save_name=%s' % (save_name),
                     'bash_call_rss_solver.sh'] )
    print(cmd)
    subprocess.call( cmd, shell=True )
    time.sleep(0.1) # delay for 0.1 seconds.	
