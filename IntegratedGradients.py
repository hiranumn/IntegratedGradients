################################################################
# Implemented by Naozumi Hiranuma (hiranumn@uw.edu)            #
#                                                              #
# Kears-compatible implmentation of Integrated Gradients       # 
# proposed in "Axiomatic attribution for deep neuron networks" #
# (https://arxiv.org/abs/1703.01365).                          #
#                                                              #
# Keywords: Shapley values, interpretable machine learning     #
################################################################
import numpy as np
from time import sleep
import sys
import keras.backend as K

from keras.models import Model, Sequential

'''
Integrated gradients approximates Shapley values by integrating partial
gradients with respect to input features from reference input to the
actual input. The following class implements this concept.
'''
class integrated_gradients:
    # model: Keras model that you wish to explain.
    # outchannels: In case the model are multi tasking, you can specify which channels you want.
    def __init__(self, model, outchannels=[], verbose=1):
    
        # Bacnend: either tensorflow or theano)
        self.backend = K.backend()
        
        #load model supports keras.Model and keras.Sequential
        if isinstance(model, Sequential):
            self.model = model.model
        elif isinstance(model, Model):
            self.model = model
        else:
            print "Invalid input model"
            return -1
        
        #load input tensors
        self.input_tensors = [
                 # input data place holder
                 self.model.inputs[0],             
                 # how much to weight each sample by
                 self.model.sample_weights[0],
                 # The learning phase flag is a bool tensor (0 = test, 1 = train)
                 # to be passed as input to any Keras function that uses 
                 # a different behavior at train time and test time.
                 K.learning_phase() 
                 ]
        
        #If outputchanel is specified, use it.
        #Otherwise evalueate all outputs.
        self.outchannels = outchannels
        if len(self.outchannels) == 0: 
            if verbose: print "Evaluated output channel (0-based index): All"
            if K.backend() == "tensorflow":
                self.outchannels = range(self.model.output.shape[1]._value)
            elif K.backend() == "theano":
                self.outchannels = range(model1.output._keras_shape[1])
        else:
            if verbose: 
                print "Evaluated output channels (0-based index):", 
                for i in self.outchannels: print i
                print
                
        #Build gradient functions for desired output channels.
        self.get_gradients = {}
        if verbose: print "Building gradient functions"
        
            # Evaluate over all channels.
        for c in self.outchannels:
            # Get tensor that calcuates gradient
            if K.backend() == "tensorflow":
                gradients = self.model.optimizer.get_gradients(self.model.output[:, c], self.model.input)
            if K.backend() == "theano":
                gradients = self.model.optimizer.get_gradients(self.model.output[:, c].sum(), self.model.input)
                
            # Build computational graph that calculates the tensfor given inputs
            self.get_gradients[c] = K.function(inputs=self.input_tensors, outputs=gradients)
            
            # This takes a lot of time for a big model with many tasks.
            # So lets pring the progress.
            if verbose:
                sys.stdout.write('\r')
                sys.stdout.write("Progress: "+str(int((c+1)*1.0/len(self.outchannels)*1000)*1.0/10)+"%")
                sys.stdout.flush()
        # Done
        if verbose: print "\nDone."
            
                
    '''
    Input: sample to explain, channel to explain
    Optional inputs:
        - reference: reference values (defaulted to 0s).
        - steps: # steps from reference values to the actual sample.
    Output: list of numpy arrays to integrated over.
    '''
    def explain(self, sample, outc=0, reference=False, num_steps=50, verbose=0):
        samples, num_steps, step_sizes = integrated_gradients.linearly_interpolate(sample, reference, num_steps)
        
        # Desired channel must be in the list of outputchannels
        assert outc in self.outchannels
        if verbose: print "Explaning the "+str(self.outchannels[outc])+"th output."
            
        # For tensorflow backend
        _input = [samples, # X
          np.ones(num_steps+1), # sample weights
          0 # learning phase in TEST mode
          ]
        
        if K.backend() == "tensorflow": 
            gradients = self.get_gradients[outc](_input)[0]
        elif K.backend() == "theano":
            gradients = self.get_gradients[outc](_input)
        gradients = np.sum(gradients, axis=0)
        explanation = np.multiply(gradients, step_sizes)
        return explanation

    
    '''
    Input: numpy array of a sample
    Optional inputs:
        - reference: reference values (defaulted to 0s).
        - steps: # steps from reference values to the actual sample.
    Output: list of numpy arrays to integrated over.
    '''
    @staticmethod
    def linearly_interpolate(sample, reference=False, num_steps=50):
        # Use default reference values if reference is not specified
        if reference is False: reference = np.zeros(sample.shape);

        # Reference and sample shape needs to match exactly
        assert sample.shape == reference.shape

        # Calcuated stepwise difference from reference to the actual sample.
        ret = np.zeros(tuple([num_steps] +[i for i in sample.shape]))
        for s in range(num_steps):
            ret[s] = reference+(sample-reference)*(s*1.0/num_steps)

        return ret, num_steps, (sample-reference)*(1.0/num_steps)