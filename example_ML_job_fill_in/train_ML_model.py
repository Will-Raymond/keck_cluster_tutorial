# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 16:12:33 2021

@author: willi
"""

import numpy as np
import os
from . import GenericML
import time
import sys

gnml = GenericML()

##### Inputs here that are called from the bash script
neuronstr = sys.argv[3].remove(' ').remove('[').remove(']').split(',')#split the input string by ,
neurons = [int(layer_size) for layer_size in neuronstr]  #convert the input string into a list of ints
save_name = str(sys.argv[2])
model_type = str(sys.argv[1])
save_dir = str(sys.argv[4])

NN2_args = gnml.default_model_args[model_type]
old_neurons = gnml.default_model_args['ffnn']['neurons']
new_neurons = neurons #new layer neuron sizes
NN2_args['neurons'] = new_neurons

# lets load the classic iris example
from sklearn.datasets import load_iris
iris_data, iris_labels = load_iris(return_X_y=True)

gnml.training_data = iris_data
iris_labels_3d = gnml.convert_labels_to_onehot(iris_labels) # so its in format [0, 1, 0] N x 3, not Nx1 [1 or 2 or 3] 
gnml.training_labels  = iris_labels_3d


print('training %s'%(model_type))
print('data loaded......')

gnml.train_model('ffnn', autosize_input=True, modelargs = NN2_args) #train the model

print('Model trained......')
gnml.save_model(save_name )

#how did we do?
print('Validation Accuracy:')
print(gnml.model.metrics['validation_accuracy'])

conf = np.array(gnml.model.metrics['conf'])
print('Confidence Matrix:')
print(conf)
classes = [0,1,2]





