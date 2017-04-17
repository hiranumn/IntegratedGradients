################################################################
# Implemented by Naozumi Hiranuma (hiranumn@uw.edu)            #
#                                                              #
# Kears-compatible implmentation of Integrated Gradients       # 
# proposed in "Axiomatic attribution for deep neuron networks" #
# (https://arxiv.org/abs/1703.01365).                          #
#                                                              #
# Keywords: Shapley values, interpretable machine learning     #
################################################################

# TODOs
# - See it this works with Tensorflow backend?
# - Make it compatible with non-sequential model.

import numpy as np
from time import sleep
import sys
import keras.backend as K

from keras.models import Model, Sequential

'''
Input: numpy array of a sample
Optional inputs:
    - reference: reference values (defaulted to 0s).
    - steps: # steps from reference values to the actual sample.
Output: list of numpy arrays to integrated over.
'''
def linear_ip(sample, reference=False, steps=50):
    # Use default reference values if reference is not specified
    if not(reference): reference = np.zeros(sample.shape);
        
    # Calcuated stepwise difference from reference to the actual sample.
    ret = np.zeros(tuple([steps] +[i for i in sample.shape]))
    for s in range(steps):
        ret[s] = reference+(sample-reference)*(s*1.0/steps)
        
    return ret, steps, (sample-reference)*(1.0/steps)

'''
Integrated gradients approximates Shapley values by integrating partial
gradients with respect to input features from reference input to the
actual input. The following class implements this concept.
'''
class integrated_gradients:
    
    # model: Keras model that you wish to explain.
    # outchannels: In case the model has multiple outputs, you can specify 
    def __init__(self, model, outchannels=False, verbose=1):
        #load model
        if isinstance(model, Sequential):
            self.model = model.model
        elif isinstance(model, Model):
            self.model = model
        else:
            print "Invalid input model"
            return -1
        
        #load input tensors
        self.input_tensors = [self.model.inputs[0], # input data
                 self.model.sample_weights[0], # how much to weight each sample by
                 K.learning_phase()
                 ]
        
        if not outchannels: 
            outchannels = range(self.model.output._keras_shape[1])
            if verbose:
                print "Evaluated output channels (0-based index): All"
        else:
            if verbose:
                print "Evaluated output channels (0-based index):",
                for i in outchannels:
                    print i,
                print
        self.outchannels = outchannels
                
        #Build gradient functions for desired output channels.
        self.get_gradients = {}
        index = 0
        if verbose: print "Building gradient functions"
        for f in range(len(outchannels)):
            gradients = model.optimizer.get_gradients(model.layers[-1].output.flatten()[outchannels[f]], model.inputs)
            self.get_gradients[f] = K.function(inputs=self.input_tensors, outputs=gradients)
            
            if verbose:
                sys.stdout.write('\r')
                sys.stdout.write("Progress: "+str(int((f+1)*1.0/len(outchannels)*1000)*1.0/10)+"%")
                sys.stdout.flush()
        if verbose: print "\nDone."

    def explain(self, sample, outc=0, reference=False, steps=50, verbose=0):
        linear_interpolated_samples, steps, stepsize = linear_ip(sample, reference, steps)
        if verbose: print "Explaning the "+str(self.outchannels[outc])+"th output."
        
        explanation = np.zeros(sample.shape) 
        
        #TODO: Figure out a way to do this in batch
        for i in range(steps):
            _input = [linear_interpolated_samples[i:i+1,:], # X
              [1], # sample weights
              0 # learning phase in TEST mode
              ]
            insta_gradients = self.get_gradients[outc](_input)[0]
            explanation += np.multiply(insta_gradients[0,:], stepsize)
        return explanation