import numpy as np

class Perceptron:
      def __init__(self):
          self.w = np.random.sample((1,3))
          self.lr = 0.20
      def response(self,n):
          if n>=1:
             n = 1
          else:
             n = -1
          return n
      def learning(self,data):
          epoch_num = 0
          while epoch_num<=10:
		net_error = 0
		print 'Epcoh number'+ ' '+str(epoch_num+1)
                for n in data:
                    d = n[3]
                    n = np.delete(n,3)
                    y = self.response(np.vdot(self.w,n))
                    error = d-y # Error 
                    self.w+= self.lr*error*n # Update the weights
                    net_error+=abs(error)
		    #plt.plot(net_error,'r.')
           	    #plt.xlabel('Epoch number')
          	    #plt.ylabel('Error')      
                epoch_num+=1          
	  #plt.show()	
          print net_error	
          return self.w
    
     


