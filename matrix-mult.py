from numpy import *

LayerSize=10

def apply_layer_new(y_in,w,b): # a function that applies a layer, w=weights, b=biases    
        z=dot(y_in,w)+b # note different order in matrix product!
        return(1/(1+exp(-z))) # sigmoid function
# for the first hidden layer (coming in from the input layer)
WeightsFirst=random.uniform(low=-1,high=1,size=[2,LayerSize]) 
# Input is 2-dimensional (and taken to be the x,y coordinates in an image)
BiasesFirst=random.uniform(low=-1,high=1,size=LayerSize)
M=10 # size of image
# Generate a 'mesh grid', i.e. x,y values in an image
v0,v1=meshgrid(linspace(-0.5,0.5,M),linspace(-0.5,0.5,M))
batchsize=M**2 # number of samples = number of pixels = M^2
y_in=zeros([batchsize,2])
y_in[:,0]=v0.flatten() # fill first component (index 0)
y_in[:,1]=v1.flatten() # fill second component
print(shape(y_in))
y=apply_layer_new(y_in,WeightsFirst,BiasesFirst)

print(shape(y))

#print(y)


