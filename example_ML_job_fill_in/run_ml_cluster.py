import subprocess
import time

<var1> = 'ffnn'
<var2> = 'example_ffnn'
<var4> = './ML_models/'

import os   #make a directory if its not there
if not os.path.exists(<var4>):
    os.makedirs(<var4>)

layers = [ [100,50,20,3], \  #the FFNN architectures to try!
           [100,200,100,25,3], \
           [40,200,300,100,3], \
           [500,300,200,100,50,3]]
           
for <var3> in layers:
    cmd = ' '.join( ['qsub','-q munsky.q@node* -cwd -o /dev/null -e Errs/',
                     '-v','','<var1>=%s' % (<var1>),
                     '-v','','<var2>=%s' % (<var2>),
                     '-v','','<var3>=%s' % (str(<var3>)),
                     '-v','','<var4>=%s' % (<var4>),
                     'bash_call_rss_solver.sh'] )
    print(cmd)
    subprocess.call( cmd, shell=True )
    time.sleep(0.1) # delay for 0.1 seconds.	
