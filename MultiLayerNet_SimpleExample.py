#!/usr/bin/env python
# Visualize the output of a multilayer network
# Input is 2 neurons (and taken to be the x,y coordinates in an image)
# Output is 1 neuron (and taken to be the color at that pixel)
#
# This demonstrates how to set up the feedforward through
# a neural network, as well as how to apply batch processing
#
# by FM, May 2017

from numpy import * # get the "numpy" library for linear algebra

import matplotlib
import matplotlib.pyplot as plt # for plotting
#%matplotlib inline

# how to apply a single layer:
def apply_layer_new(y_in,w,b): # a function that applies a layer, w=weights, b=biases    
    z=dot(y_in,w)+b # note different order in matrix product!
    return(1/(1+exp(-z))) # sigmoid function

# setting up all the network, with random weighst & biases:
Nlayers=20 # not counting the input layer & the output layer
LayerSize=100  # 100 neurons

# all the weights in the intermediate layers:
Weights=random.uniform(low=-3,high=3,size=[Nlayers,LayerSize,LayerSize])
#this is an array of dim Nlayers*LayerSize*LayerSize
Biases=random.uniform(low=-1,high=1,size=[Nlayers,LayerSize])
# numpy.random.uniform(low=0.0, high=1.0, size=None): Draw samples from a uniform distribution.
#Samples are uniformly distributed over the half-open interval [low, high) (includes low, but excludes high)

# for the first hidden layer (coming in from the input layer)
WeightsFirst=random.uniform(low=-1,high=1,size=[2,LayerSize]) 
# Input is 2-dimensional (and taken to be the x,y coordinates in an image)
BiasesFirst=random.uniform(low=-1,high=1,size=LayerSize)

# for the final layer (i.e. the output neuron)
WeightsFinal=random.uniform(low=-1,high=1,size=[LayerSize,1])
BiasesFinal=random.uniform(low=-1,high=1,size=1)

# how to apply the full network:
def apply_multi_net(y_in):
    global Weights, Biases, WeightsFinal, BiasesFinal, Nlayers
    
    y=apply_layer_new(y_in,WeightsFirst,BiasesFirst)    
    for j in range(Nlayers):
        y=apply_layer_new(y,Weights[j,:,:],Biases[j,:])
    output=apply_layer_new(y,WeightsFinal,BiasesFinal)
    return(output)

# Now plot the result, as a picture

M=100 # size of image
# Generate a 'mesh grid', i.e. x,y values in an image
v0,v1=meshgrid(linspace(-0.5,0.5,M),linspace(-0.5,0.5,M))
batchsize=M**2 # number of samples = number of pixels = M^2
y_in=zeros([batchsize,2])
y_in[:,0]=v0.flatten() # fill first component (index 0)
y_in[:,1]=v1.flatten() # fill second component

# use the MxM input grid that we generated above 
y_out=apply_multi_net(y_in) # apply net to all these samples!

y_2D=reshape(y_out[:,0],[M,M]) # back to 2D image

plt.figure(figsize=[10,10])
plt.axes([0,0,1,1]) # fill all of the picture with the image
plt.imshow(y_2D,origin='lower',extent=[-0.5,0.5,-0.5,0.5],interpolation='nearest')
plt.axis('off') # no axes
plt.show()
matplotlib.pyplot.savefig('out.png')
