# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 12:26:39 2020

@author: willi
"""


import numpy as np
import itertools as it
import re
import os

import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split
import itertools
import warnings
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve, f1_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix

from tensorflow import keras

import joblib 
import pickle
import json
from matplotlib import cm

# define layer types here

Sequential =  keras.models.Sequential  
Dense = keras.layers.Dense
LSTM = keras.layers.LSTM 
Conv2D = keras.layers.Conv2D
Conv1D = keras.layers.Conv1D
Embedding =  keras.layers.Embedding 
sequence =  keras.preprocessing.sequence
Flatten = keras.layers.Flatten
MaxPooling2D = keras.layers.MaxPooling2D
MaxPooling1D = keras.layers.MaxPooling1D
LSTM = keras.layers.LSTM 
Embedding =  keras.layers.Embedding 
sequence =  keras.preprocessing.sequence


'''

What has to be user defined:
    
    MLfun label thresholds
    GenericML.load_training_data
    Network architectures past defaults
    
'''



class MLfun():
    '''
    shared functions for all machine learning methods, ML models have access to these
    '''
    def __init__(self):
        pass
    
    def get_metrics(self,xtest,ytest,xlabel,ylabel):
        
        thresh = .5 #DEFINE YOUR PROBABILITY THRESHOLDS HERE FOR LABELS
        
        ypred = self.model.predict(ytest)
        yclass = np.array(ypred >thresh ).astype(int)

        #acc = np.sum(np.abs((yclass.flatten() - ylabel.flatten())))/len(yclass)
        

        
        if yclass.shape[1] < 3:
            try:
                tmpylabel = np.argmax(ylabel,axis=1)
                tmpyclass = np.argmax(yclass,axis=1)
            except:
                tmpylabel = ylabel
                tmpyclass = yclass
                

            conf = confusion_matrix(tmpylabel,tmpyclass)
            acc = np.sum(np.diag(conf))/ np.sum(conf)
            model_fpr, model_tpr = self.get_roc(ypred,ylabel)
            f1 = f1_score(ylabel,yclass)
            return {'acc':acc, 'fpr':model_fpr.tolist(),'tpr':model_tpr.tolist(),'f1':f1,'conf':conf.tolist(), 'validation_accuracy': self.scores[-1], 'ytrue':tmpylabel, 'ypred': tmpyclass, 'yprob':ypred }   
        
        else:
            
            
            f1 = f1_score(ylabel,yclass, average='macro')
            
            try:
                tmpylabel = np.argmax(ylabel,axis=1)
                tmpyclass = np.argmax(yclass,axis=1)
            except:
                tmpylabel = ylabel
                tmpyclass = yclass
                

            conf = confusion_matrix(tmpylabel,tmpyclass)
                
            acc = np.sum(np.diag(conf))/ np.sum(conf)
            roc_auc = roc_auc_score(ylabel, ypred, average='macro')
            
            return {'acc':acc, 'roc_auc':roc_auc,'f1':f1,'conf':conf.tolist(),'validation_accuracy': self.scores[-1], 'ytrue':tmpylabel, 'ypred': tmpyclass, 'yprob':ypred }   
        
    
    def get_roc(self,probs, test_labels):
        """Compare machine learning model to baseline performance.
        Computes statistics and shows ROC curve."""
        if type(probs) != list:
            probs = list(probs)
    
        if type(test_labels) != list:
            test_labels = list(test_labels)
        
        # Calculate false positive rates and true positive rates
        base_fpr, base_tpr, _ = roc_curve(test_labels, [1 for _ in range(len(test_labels))])
        model_fpr, model_tpr, _ = roc_curve(test_labels, probs)
        return model_fpr, model_tpr


class KNN(MLfun):
    '''
    K nearest neighbors wrapper
    
    https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm
    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    
    **arguments**
    
    - n_neighbors, number of neighbors for the classifier
    
    ::WARNING:: using a large amount of neighbors for a large dataset will generate a LARGE and slow model file.
    recommended to keep data below 20,000 samples
    
    '''
    
    def __init__(self, n_neighbors,):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
    def fit(self,x_training, x_test, labels_training, labels_test ):
        self.model.fit(x_training, labels_training)
        self.scores = np.array([1- np.sum ( np.abs(labels_test - self.model.predict(x_test)) )/len(labels_test)])
        self.metrics = self.get_metrics( x_training, x_test, labels_training, labels_test, )
        
    def load_model(self, filename):
        self.model = pickle.load(open(filename,'rb'))
    


class RF(MLfun):
    '''
    Random Forest Wrapper 
    
    https://en.wikipedia.org/wiki/Random_forest
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    
    **arguments**
    
    - n_trees, number of random trees
    - seed, random seed to use
    
    '''
    
    def __init__(self, n_trees, seed):
        
        self.model = RandomForestClassifier(n_estimators=n_trees, 
                               bootstrap = True,
                               max_features = 'sqrt',
                               random_state = seed,
                               verbose=1)
        self.seed = seed
        
    def fit(self,x_training, x_test, labels_training, labels_test):
        self.model.fit(x_training, labels_training)
        
        self.scores = np.array(  [1- np.sum ( np.abs(labels_test - self.model.predict(x_test)) )/len(labels_test) ])
        self.metrics = self.get_metrics( x_training, x_test, labels_training, labels_test, )

    def load_model(self, filename):
        self.model = None
        self.model = joblib.load(filename)
        


class LSTMNN(MLfun):
    '''
    LSTM-RNN wrapper
    
    LSTM-RNNs are good for text/ linear pattern classification but are slow to train
    Recommended to keep nepochs low
    
    **arguments**
    
    input_length, how large the input is
    n_neurons, number of neurons
    activation, the final layer activation function
    vectors_per_char, the maximum encoding
    

    
    '''
    
    
    def __init__(self, input_length, n_neurons, activation,vectors_per_char,  ):

        #Defaults
            
        # input_length = 300

        # n_neruons = 100
        # activation = 'sigmoid'
        # vectors_per_char = 32
    
        
        self.model = Sequential()
        self.model.add(Embedding(1024, vectors_per_char, input_length=input_length)) 
        self.model.add(LSTM(n_neurons))   
     
        self.model.add(Dense(3, activation = activation ))            
            
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        
    
    def fit(self,x_training, x_test, labels_training, labels_test,   n_epochs = 20, batch_size = 200, normalize = False, verbose=1):
        print('------------------------------------------------------')
        print('Running LSTM training with the following conditions: ')
        print('n epochs: %i    batches: %i '%(n_epochs,batch_size) )
        print('total testing data: %i    total training: %i '%(int(len(labels_training)), int(len(labels_test))  ))
        print('------------------------------------------------------')
        print(self.model.summary())
        if normalize == True:
            x_training = x_training/max([x_training.max(), x_test.max()])
            x_test = x_test/max([x_training.max(), x_test.max()])

        
        self.model.fit(x_training, labels_training, epochs=n_epochs, batch_size=batch_size, verbose=verbose,validation_data=(x_test, labels_test)) 
        self.scores = self.model.evaluate(x_test, labels_test, verbose=0 )

        self.metrics = self.get_metrics( x_training, x_test, labels_training, labels_test, )
        
    def load_model(self,file):
        self.model.load_weights(file)       
        


class CNN2D(MLfun):
    '''
    2d Convolutional NN wrapper
    
    Good for image and time series data
    
    **arguments**
    
    input_shape, how large the input is, should be something like (1,n_samples,X,Y) in shape
    conv_layers, how many convultional layers
    conv_kernels, the size and shape of kernels for the convolution, ex: (2,2) (3,3)
    dense_layers, the number of fully connected layers after the convolutions
    neurons, number of neurons in the dense layers
    activations, activation functions for the dense layers
    max_pool, use average pooling? boolean
    pool_sizes, the sizes of the pools to average
    strides, the size of strides for pooling
    padding, type of padding
    
    '''    
    def __init__(self, input_shape, conv_layers, conv_sizes, conv_kernels,  dense_layers, neurons, activations, max_pool, pool_sizes, strides, padding ):
        #Defaults
            
        # input_dim = 8x8x1

        # conv layers = 4
        # conv_sizes = [50,46,30,16]
        # conv_kernels = [(2,2),(2,2),(2,2),(2,2) ]
        
        
        # dense_layers = 4
        # neurons = [256,128,64,32,2]
        # activations = ['relu','relu','relu','relu','relu','relu','relu','softmax']
        
        
        self.model = Sequential()  #convolutions
        for i in range(conv_layers):
            if i == 0:
                self.model.add(Conv2D (conv_sizes[i],conv_kernels[i],  input_shape=input_shape, activation=activations[i]))
                
                if max_pool:
                    self.model.add(MaxPooling2D(pool_size = pool_sizes[i], strides =  strides[i], padding= padding ))
            else:
                
                self.model.add(Conv2D( conv_sizes[i],conv_kernels[i] , activation=activations[i]))
                if max_pool:
                    self.model.add(MaxPooling2D( pool_size = pool_sizes[i], strides =  strides[i], padding= padding ))
                    
        self.model.add(Flatten())  #flatten 
        
        for i in range(dense_layers):
                self.model.add(Dense(neurons[i], activations[i]))
                
        self.model.add(Dense(neurons[-1], activation = activations[-1] ))            
            
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
    
    def fit(self,x_training, x_test, labels_training, labels_test,   n_epochs = 20, batch_size = 200, normalize = False, verbose=1):
        print('------------------------------------------------------')
        print('Running CNN training with the following conditions: ')
        print('n epochs: %i    batches: %i    normalized to 0-1:  %i'%(n_epochs,batch_size,normalize) )
        print('total testing data: %i    total training: %i '%(int(len(labels_training)), int(len(labels_test))  ))
        print('------------------------------------------------------')
        print(self.model.summary())
        if normalize == True:
            x_training = x_training/max([x_training.max(), x_test.max()])
            x_test = x_test/max([x_training.max(), x_test.max()])

        
        self.model.fit(x_training, labels_training, epochs=n_epochs, batch_size=batch_size, verbose=verbose,validation_data=(x_test, labels_test)) 
        self.scores = self.model.evaluate(x_test, labels_test, verbose=0 )
        self.metrics = self.get_metrics( x_training, x_test, labels_training, labels_test, )
        
    def load_model(self,file):
        self.model.load_weights(file)       

class CNN(MLfun):
    '''
    1d Convolutional NN wrapper
    
    Good for image and time series data
    
    **arguments**
    
    input_shape, how large the input is, should be something like (1,n_samples,X) in shape
    conv_layers, how many convultional layers
    conv_kernels, the size and shape of kernels for the convolution, ex: (2) (3)
    dense_layers, the number of fully connected layers after the convolutions
    neurons, number of neurons in the dense layers
    activations, activation functions for the dense layers
    max_pool, use average pooling? boolean
    pool_sizes, the sizes of the pools to average
    strides, the size of strides for pooling
    padding, type of padding
    
    '''        
    def __init__(self, input_shape, conv_layers, conv_sizes, conv_kernels,  dense_layers, neurons, activations,max_pool, pool_sizes, strides, padding  ):
        #Defaults
            
        # input_dim = 8x8x1

        # conv layers = 4
        # conv_sizes = [50,46,30,16]
        # conv_kernels = [(2,2),(2,2),(2,2),(2,2) ]
        
        
        # dense_layers = 4
        # neurons = [256,128,64,32,2]
        # activations = ['relu','relu','relu','relu','relu','relu','relu','softmax']
        
        
        self.model = Sequential()  #convolutions
        for i in range(conv_layers):
            if i == 0:
                self.model.add(Conv1D (conv_sizes[i],conv_kernels[i],  input_shape=input_shape, activation=activations[i]))
                if max_pool:
                    self.model.add(MaxPooling1D(pool_size = pool_sizes[i], strides =  strides[i], padding= padding ))
            else:
                
                self.model.add(Conv1D( conv_sizes[i],conv_kernels[i] , activation=activations[i]))
                if max_pool:
                    self.model.add(MaxPooling1D(pool_size = pool_sizes[i], strides =  strides[i], padding= padding))
        self.model.add(Flatten())  #flatten 
        
        for i in range(dense_layers):
                self.model.add(Dense(neurons[i], activations[i]))
                
        self.model.add(Dense(neurons[-1], activation = activations[-1] ))            
            
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
    
    def fit(self,x_training, x_test, labels_training, labels_test,   n_epochs = 20, batch_size = 200, normalize = False, verbose=1):
        print('------------------------------------------------------')
        print('Running CNN training with the following conditions: ')
        print('n epochs: %i    batches: %i    normalized to 0-1:  %i'%(n_epochs,batch_size,normalize) )
        print('total testing data: %i    total training: %i '%(int(len(labels_training)), int(len(labels_test))  ))
        print('------------------------------------------------------')
        print(self.model.summary())
        if normalize == True:
            x_training = x_training/max([x_training.max(), x_test.max()])
            x_test = x_test/max([x_training.max(), x_test.max()])

        
        self.model.fit(x_training, labels_training, epochs=n_epochs, batch_size=batch_size, verbose=verbose,validation_data=(x_test, labels_test)) 
        self.scores = self.model.evaluate(x_test, labels_test, verbose=0 )
        self.metrics = self.get_metrics( x_training, x_test, labels_training, labels_test, )
    def load_model(self,file):
        self.model.load_weights(file)       
        
    




class ffNN(MLfun):
    '''
    Feed Forward NN wrapper

    **arguments**
    
    input_dim, how large the input is, float, ex: 64
    layers, how many layers?, float
    neurons, how many neurons per layer, list of ints ex: [100,10,2]
    activations, activation functions for the dense layers

    
    '''            
    def __init__(self, input_dim, layers, neurons, activations ):
        #Defaults
            
        # input_dim = 64
        # layers = 6
        # neurons = [100,50,25,10,4,1]
        # activations = ['relu','relu','relu','relu','relu','sigmoid']
            
        
        self.model = Sequential()
        for i in range(layers-1):
            if i == 0:
                self.model.add(Dense(neurons[i], input_dim=input_dim, activation=activations[i]))
            else:
                self.model.add(Dense(neurons[i], activations[i]))
        self.model.add(Dense(neurons[-1], activation = activations[-1] ))
        
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
    
    def fit(self,x_training, x_test, labels_training, labels_test,   n_epochs = 20, batch_size = 200, normalize = True, verbose = 1):
        print('------------------------------------------------------')
        print('Running FFNN training with the following conditions: ')
        print('n epochs: %i    batches: %i    normalized to 0-1:  %i'%(n_epochs,batch_size,normalize) )
        print('total testing data: %i    total training: %i '%(int(len(labels_training)), int(len(labels_test))  ))
        print('------------------------------------------------------')
        print(self.model.summary())
        if normalize == True:
            x_training = x_training/max([x_training.max(), x_test.max()])
            x_test = x_test/max([x_training.max(), x_test.max()])

        
        self.model.fit(x_training, labels_training, epochs=n_epochs, batch_size=batch_size, verbose=1,validation_data=(x_test, labels_test)) 
        self.scores = self.model.evaluate(x_test, labels_test, verbose=0 )
        self.metrics = self.get_metrics( x_training, x_test, labels_training, labels_test, )
        
        
    def load_model(self,file):
        self.model.load_weights(file)
        
    def metrics(self):
        x=1
        
        
        
        
class GenericML():
    '''
    Generic class to provide access to the ML wrappers
    
    '''
    
    def __init__(self):
        
        
        self.testing_set = None
        self.model = None
        
        self.training_path = '.'
        self.save_dir = '.'
        
        self.default_model_args = {}
        self.default_model_args['ffnn'] = {'input_dim': 64,
                                           'layers': 6,
                                           'neurons': [200, 100, 50, 25, 10, 3],
                                           'activations': ['relu', 'relu', 'relu', 'relu', 'relu', 'softmax']}
        
        self.default_model_args['cnn'] = {'input_shape': (8,8,1),
                                          'conv_layers':4,
                                          'conv_kernels':  [(5),(4),(3),(2) ],
                                          'conv_sizes': [128,64,32,16],
                                          'max_pool': False,
                                          'pool_sizes': [(2),(2),(2),(2)],
                                          'strides': [(1),(1),(1),(1)],
                                          'padding': 'valid',
                                          'dense_layers': 4,
                                          'neurons': [256,128,64,32,3],
                                          'activations': ['relu', 'relu', 'relu', 'relu', 'relu','relu','relu', 'softmax']}        
        
        
        self.default_model_args['lstm'] = {'input_length': 300,
                                           'n_neurons': 100,
                                           'vectors_per_char':32,
                                           'activation': 'softmax'}
                                           
        
        self.default_model_args['cnn2d'] = {'input_shape': (8,8,1),
                                          'conv_layers':4,
                                          'conv_kernels':  [(5,5),(4,4),(3,3),(2,2) ],
                                          'conv_sizes': [128,64,32,16],
                                          'max_pool': False,
                                          'pool_sizes': [(2,2),(2,2),(2,2),(2,2)],
                                          'strides': [(1,1),(1,1),(1,1),(1,1)],
                                          'padding': 'valid',
                                          'dense_layers': 4,
                                          'neurons': [256,128,64,32,3],
                                          'activations': ['relu', 'relu', 'relu', 'relu', 'relu','relu','relu', 'softmax']}        
        
        self.default_model_args['rf'] = {'n_trees':50, 'seed':10}
        self.default_model_args['knn'] = {'n_neighbors':2}
        
        self.default_data_args = {'witheld_percentage':.2,'seed':10 }
        
        self.default_training_args = {}
        self.default_training_args['ffnn'] = {'n_epochs': 20, 'batch_size': 200, 'normalize': False}
        self.default_training_args['cnn'] = {'n_epochs': 20, 'batch_size': 500, 'normalize': False}
        self.default_training_args['cnn2d'] = {'n_epochs': 20, 'batch_size': 100, 'normalize': False}

        
        self.default_training_args['lstm'] = {'n_epochs': 5, 'batch_size': 500, 'normalize': False}
        self.default_training_args['knn'] = {}
        
        self.default_training_args['rf'] = {}
    
        
    
    
    def load_training_data(self):
        '''
        Define a function here to load the training data in your format
        '''
        #load training data here
        self.training_data = 1
        self.training_labels = 1
        self.withheld_data = 1
        
        
        
    def __calculate_cnn_flat_layer_dim(self,model_type='cnn' ):
        '''
        Returns the size of the flattened dimension of a CNN
        
        * Arguments: model_type = 'cnn' or 'cnn2d'
        '''
        modelargs = self.default_model_args[model_type]
        slen = len(self.training_data[0,:])
        kernels = modelargs['conv_kernels']
        filters = modelargs['conv_sizes']
        dense_neuron_1 = filters[-1]* (np.array([slen - x  for x in np.cumsum(kernels)]) + np.array([x+1 for x in range(0,len(kernels))]) )[-1]
        return dense_neuron_1
        
    def train_model(self, model_type='cnn', modelargs = None, dataargs = None, trainingargs = None, autosize_input = False):
        '''
        Trains the model type desired for whatever argument dictionaries are passed
        
        modelargs is a dictionary of terms to be passed to the model classes
        dataargs goes to the data formatting function, sklearn.test_train_split
        trainingargs goes to training methods for tf.model.fit()
        
        autosize input, boolean, if true will automatically resize the input layers to match your self.training_data
        
        '''

        if modelargs == None:
            modelargs = self.default_model_args[model_type.lower()]
            
            if autosize_input:
                if model_type == 'cnn':
                
                    if modelargs['input_shape'] != self.training_data[0,:].shape:
                        modelargs['input_shape'] = self.training_data[0,:].shape 
                        # slen = len(self.training_data[0,:])
                        # kernels = modelargs['conv_kernels']
                        # filters = modelargs['conv_sizes']
                        # dense_neuron_1 = filters[-1]* (np.array([slen - x  for x in np.cumsum(kernels)]) + np.array([x+1 for x in range(0,len(kernels))]) )[-1]
                        # print('calculated flatten size:')
                        # print(dense_neuron_1)
                        # modelargs['neurons'][0] = dense_neuron_1
        
                if model_type == 'cnn2d':
                
                    if modelargs['input_shape'] != self.training_data[0,:].shape :
                        modelargs['input_shape'] = self.training_data[0,:].shape 
                        # slen = len(self.training_data[0,:])
                        # kernels = modelargs['conv_kernels']
                        # filters = modelargs['conv_sizes']
                        # dense_neuron_1 = filters[-1]* (np.array([slen - x  for x in np.cumsum(kernels)]) + np.array([x+1 for x in range(0,len(kernels))]) )[-1]
                        # print('calculated flatten size:')
                        # print(dense_neuron_1)
                        # modelargs['neurons'][0] = dense_neuron_1  
                        
                if model_type == 'ffnn':
                    
                    if modelargs['input_dim'] != len(self.training_data[0,:]):
                        modelargs['input_dim'] = len(self.training_data[0,:])
                        
                if model_type == 'lstm':
                    
                    if modelargs['input_length'] != len(self.training_data[0,:]):
                        modelargs['input_length'] = len(self.training_data[0,:])        
        
        if dataargs == None:
            dataargs = self.default_data_args
            
        if trainingargs == None:
            trainingargs = self.default_training_args[model_type.lower()]
            
        if model_type.lower() in ['ffnn','knn','rf']:
            self.model_type = model_type.lower()
            if model_type.lower() == 'ffnn':
                self.model = ffNN( **modelargs)
            if model_type.lower() == 'knn':
                self.model = KNN( **modelargs)
            if model_type.lower() == 'rf':
                self.model = RF( **modelargs)       
                
            self.model.fit( *self.format_training_data(self.training_data,self.training_labels,**dataargs),**trainingargs  )
            
        if model_type.lower() == 'cnn':
            self.model_type = 'cnn'
      
            self.model = CNN( **modelargs)
               
            self.model.fit( *self.format_training_data(self.training_data,self.training_labels,**dataargs),**trainingargs  )

        if model_type.lower() == 'cnn2d':
            self.model_type = 'cnn2d'
      
            self.model = CNN2D( **modelargs)
            
            self.model.fit( *self.format_training_data(self.training_data,self.training_labels,**dataargs),**trainingargs  )

                        
        if model_type.lower() in ['lstm']:
            self.model_type = model_type.lower()
          
            self.model = LSTMNN( **modelargs)

            self.model.fit( *self.format_training_data(self.training_data,self.training_labels,**dataargs),**trainingargs  )
                
        self.model_trained = True
        
    def format_training_data(self,data,labels, witheld_percentage =.2, seed = 10 ):
        x_train, x_test, label_train, label_test = train_test_split(data, labels, test_size=witheld_percentage,random_state=seed)
        return x_train, x_test, label_train, label_test
    


    def test_witheld_data(self):
        self.withheld_data = 1
        labels = self.model.model.predict( self.withheld_data  )
        acc = 1
        return acc
        
    
    def convert_labels_to_onehot(self,labels):
        '''
        converts labels in the format 1xN, [0,0,1,2,3,...] to onehot encoding,
        ie: N_classes x N,  [[1,0,0,0],[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]     
    
        '''
        
        onehotlabels = np.zeros((labels.shape[0],len(np.unique(labels))))
        for i in range(len(onehotlabels)):
            
            onehotlabels[i,labels[i]] = 1
        return onehotlabels
        
        
    
    def save_model(self,filename):
        '''
        Will save a model file + metrics .json if available
        
        '''
        
        if self.model_type in ['ffnn','cnn', 'lstm','cnn2d']:
            self.model.model.save((self.save_dir + filename))
            try: 
                self.model.model.metrics
                with open((filename + '.json'), 'w') as fp:
                    json.dump(self.model.model.metrics, fp)
            except:
                pass
            
            
            
        if self.model_type == 'rf':
            joblib.dump(self.model.model, filename) 
            try: 
                self.model.model.metrics
                with open((filename + '.json'), 'w') as fp:
                    json.dump(self.model.model.metrics, fp)
            except:
                pass
            
                
        if self.model_type == 'knn':
            with open(filename, 'wb') as f:
                pickle.dump(self.model.model, f)   
            try: 
                self.model.model.metrics
                with open((filename + '.json'), 'w') as fp:
                    json.dump(self.model.model.metrics, fp)
            except:
                pass
            
          
    def load_model(self,model_type, model_file, modelargs = None):
        
        
        if modelargs == None:
            modelargs = self.default_model_args[model_type.lower()]
 
        if model_type.lower() == 'ffnn':
            self.model_type = 'ffnn'
            self.model = ffNN( **modelargs)
            self.model.load_model(model_file)
            
        if model_type.lower() == 'cnn':
            self.model_type = 'cnn'
            
            
            if modelargs['input_shape'] != self.training_data[0,:].shape:
                modelargs['input_shape'] = self.training_data[0,:].shape 
                    
            self.model = CNN( **modelargs)
            self.model.load_model(model_file)         

        if model_type.lower() == 'cnn2d':
            self.model_type = 'cnn2d'
            
            
            if modelargs['input_shape'] != self.training_data[0,:].shape:
                modelargs['input_shape'] = self.training_data[0,:].shape 
                    
            self.model = CNN2D( **modelargs)
            self.model.load_model(model_file)   
        
        if model_type.lower() == 'lstm':
            self.model_type = 'lstm'
            self.model = LSTMNN( **modelargs)
            self.model.load_model(model_file) 
            
        if model_type.lower() == 'knn':
            self.model_type = 'knn'
            self.model = KNN( **modelargs)
            self.model.load_model(model_file) 
            
        if model_type.lower() == 'rf':
            self.model_type = 'rf'
            self.model = RF( **modelargs)
            self.model.load_model(model_file) 
            
        self.model_trained = True
        try:  #reload metrics if available
            fname = '.'.join(model_file.split('.')[:-1])
            with open((fname + '.json'), 'r') as fp:
                data = json.load(fp)
            self.model.metrics = data
        except:
            pass       

if __name__ == "__main__":
    #example usage
    
    gnml = GenericML()
    
    
    # lets load the classic iris example
    from sklearn.datasets import load_iris
    iris_data, iris_labels = load_iris(return_X_y=True)
    
    c = cm.Spectral(iris_labels*(256/2))
    plt.scatter(iris_data[:,0],iris_data[:,1],c=c )
    
    
    '''
    we can set the training data / labels directly 
    '''
    
    gnml.training_data = iris_data
    
    iris_labels_3d = gnml.convert_labels_to_onehot(iris_labels) # so its in format [0, 1, 0] N x 3, not Nx1 [1 or 2 or 3] 
    gnml.training_labels  = iris_labels_3d
    
    
    gnml.train_model('knn')
    
    print('data loaded......')
    gnml.train_model('ffnn', autosize_input=True)
    print('Model trained......')
    
    gnml.save_model('testmodel1.h5' )
    
    #how did we do?
    print('Validation Accuracy:')
    print(gnml.model.metrics['validation_accuracy'])
    
    
    conf = np.array(gnml.model.metrics['conf'])
    
    classes = [0,1,2]
    
    fig, ax = plt.subplots()
    ax.matshow(conf, cmap = cm.summer_r )
    for i in range(len(classes)):
        for j in range(len(classes)):
            text = ax.text(j, i, conf[i, j],
                           ha="center", va="center", color="k")
    
    fig,ax = plt.subplots(2,2,dpi=300)
    c = cm.Dark2(iris_labels)
    ax[0,0].scatter(iris_data[:,0],iris_data[:,1],c=c )
    fig.show()
    ax[0,0].set_title('True classes')
    

    ytest = gnml.model.model.predict(iris_data)
    yclass = (ytest > .5)
    yclass = np.argmax(yclass.astype(int),axis=1)
    
    c = cm.Dark2(yclass)
    ax[0,1].scatter(iris_data[:,0],iris_data[:,1],c=c )
    fig.show()
    ax[0,1].set_title('Predicted classes')
    
    c = cm.Dark2(iris_labels)
    ax[1,0].scatter(iris_data[:,2],iris_data[:,1],c=c )
    fig.show()
    

    ytest = gnml.model.model.predict(iris_data)
    yclass = (ytest > .5)
    yclass = np.argmax(yclass.astype(int),axis=1)
    
    c = cm.Dark2(yclass)
    ax[1,1].scatter(iris_data[:,2],iris_data[:,1],c=c )
    
    fig.show()
    
    
    
    #pretty terrible, lets add some more neurons
    
    print(gnml.default_model_args['ffnn']['neurons'])
    
    
    NN2_args = gnml.default_model_args['ffnn']
    old_neurons = gnml.default_model_args['ffnn']['neurons']
    new_neurons = [300,600,300,100,50,3] #new layer neuron sizes
    NN2_args['neurons'] = new_neurons
    
    gnml.train_model('ffnn', autosize_input=True, modelargs = NN2_args)
    
    
    print('Validation Accuracy:')
    print(gnml.model.metrics['validation_accuracy'])
    
    
    conf = np.array(gnml.model.metrics['conf'])
    
    classes = [0,1,2]
    
    fig, ax = plt.subplots()
    ax.matshow(conf, cmap = cm.summer_r )
    for i in range(len(classes)):
        for j in range(len(classes)):
            text = ax.text(j, i, conf[i, j],
                           ha="center", va="center", color="k")
            
    gnml.save_model('testmodel2.h5' )
    
    
    fig,ax = plt.subplots(2,2,dpi=300)
    c = cm.Dark2(iris_labels)
    ax[0,0].scatter(iris_data[:,0],iris_data[:,1],c=c )
    fig.show()
    ax[0,0].set_title('True classes')
    

    ytest = gnml.model.model.predict(iris_data)
    yclass = (ytest > .5)
    yclass = np.argmax(yclass.astype(int),axis=1)
    
    c = cm.Dark2(yclass)
    ax[0,1].scatter(iris_data[:,0],iris_data[:,1],c=c )
    fig.show()
    ax[0,1].set_title('Predicted classes')
    
    c = cm.Dark2(iris_labels)
    ax[1,0].scatter(iris_data[:,2],iris_data[:,1],c=c )
    fig.show()
    

    ytest = gnml.model.model.predict(iris_data)
    yclass = (ytest > .5)
    yclass = np.argmax(yclass.astype(int),axis=1)
    
    c = cm.Dark2(yclass)
    ax[1,1].scatter(iris_data[:,2],iris_data[:,1],c=c )
    
    fig.show()
    
    
    
    