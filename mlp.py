 	# Build a neural network with one hidden layer to learn XOR gate
import numpy as np
from matplotlib import pyplot as plt

class MultiLayerPerceptron:
      def __init__(self):
          self.learning_rate = 0.5
          self.W1 = np.random.random((2,3)) # Weight matrix and bias initialization from input to hidden layer
          self.W2 = np.random.random((1,3)) # Weight matrix and bias initialization from hidden layer to output
# Create matrices to store the gradient for the weight matrices
          self.delta_W1 = np.zeros(np.shape(self.W1))
          self.delta_W2 = np.zeros(np.shape(self.W2))

      def sigmoid(self,inp):
          return 1/(1+np.exp(-inp))
 
      def derivative(self,inp):
          return ((1-inp)*(inp))        
  
      def forward_pass(self,x):
          x = np.transpose(x)
          x = np.hstack((1,x))
          v1 = np.dot(self.W1,x)
          activated_hidden = self.sigmoid(v1)
          activated_hidden = np.hstack((1,activated_hidden))
          v2 = np.dot(self.W2,activated_hidden)
          y = self.sigmoid(v2)
	  return (activated_hidden,y)

      def backward_pass(self,d,activated_hidden,y):
	  sq_error = 0
          error = d-y
	  sq_error+= error**2
          delta_output = error*self.derivative(y)
          delta_hidden = self.derivative(activated_hidden)*delta_output*self.W2
          self.delta_W2= self.learning_rate*delta_output*y
	  self.W2+=self.delta_W2
	  self.delta_W1= self.learning_rate*activated_hidden*delta_hidden
          self.W1+=self.delta_W1	
          return sq_error

      def return_weights(self):
          return (self.W1,self.W2)

X = np.array([[0,0],[1,1],[0,1],[1,0]])
d = [0,0,1,1]
epoch = 100000
MLP = MultiLayerPerceptron()
for i in xrange(epoch):
    for x in X:
        ctr = 0
        (AH,Y) = MLP.forward_pass(x)
        SE = MLP.backward_pass(d[ctr],AH,Y)
        ctr+=1
T1,T2 = MLP.return_weights()


