import numpy as np
from perceptron import Perceptron
from matplotlib import pyplot as plt

def data_generate(n):
         a = (2-1)*np.random.random_sample((n+40,2))+1
         b = (5-4)*np.random.random_sample((n+40,2))+4
         bias = np.ones((n+40,1))
         a = np.concatenate((bias,a),axis = 1)
         b = np.concatenate((bias,b),axis = 1)
         return a,b
def show_data(v1,v2): # function to plot data
         plt.plot(v1[:,1],v1[:,2],'r.',v2[:,1],v2[:,2],'g*')
         plt.xlabel('x-axis')
         plt.ylabel('y-axis')
         plt.axis([-2,7,-2,7])
         plt.show()
def labels(n,v1,v2):
         l = np.ones((n+40,1))
         a = np.concatenate((v1,l),axis = 1)
         b = np.concatenate((v2,-1*l),axis = 1)
# concatenate the data to form a single training set
         train_data = np.concatenate((a,b),axis =0)
         return train_data
# Instantiate the class Perceptron
c,d = data_generate(60) # Generate test data
show_data(c,d)
# Add labels +1 to 'a' and -1 to 'b'
Training = labels(60,c,d)
w = Perceptron() # Instantiate the Perceptron class
# Now pass the training data for the perceptron to learn the weight vectors 
# using the Rosenblatt learning algorithm
weight_vector = w.learning(Training)

t1,t2 = data_generate(10)
Test = labels(10,t1,t2)
correct = 0
incorrect = 0
for x in Test:
    d =	x[3]
    x = np.delete(x,3)
    y = w.response(np.vdot(weight_vector,x))
    if y == d:
       correct+=1
    else:
       incorrect+=1

print 'Number of correctly classified points is' +' '+str(correct)
print 'Number of incorrectly classified points is' +' '+str(incorrect)

