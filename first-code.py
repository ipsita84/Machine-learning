# Neural network without hidden layer

from numpy import * # get numpy library for linear algebra
n0=3 # input layer size
n1=2 # output layer size

w=random.uniform(low=-1,high=+1,size=(n1,n0)) # assigning randowm n08n1 no. of weighs
b=random.uniform(low=-1,high=+1,size=n1) # biases which forms a column vector of size n1

y_in=array([0.2,0.4,-0.1]) # these are input values
z=dot(w,y_in)+b # output vector of size n1

print(z)
