import numpy as np

def xor(x1, x2):
	if x1 == x2:
	  return 0
	else:
	  return 1

def gradient(f, x):
	h = 1e-4
	grad = np.zeros_like(x)
	
	for idx in range(x.size):
		tmp_val = x[idx]
		x[idx] = tmp_val + h
		fxhl = f(x)
		
		x[idx] = tmp_val - h
		fx2 = f(x)

		grad[idx] = (fxh1 - fxh2) / (2*h)
		x[idx] = tmp_val
	return grad


class simpleNet:
	def __init__(self,A):
	   self.X = None
	   self.A = A
	   self.params = {}
	   self.params['W'] = np.random.rand(2)
	   self.params['B'] = np.random.rand(1)

	def predict(self, x):
	   W = self.params['W'] 
	   B = self.params['B']
	   self.X = x
	   return np.sum(x * W.T) + B

	def backward(self, X, y):
	   W, b = self.params.values()
	   #dW, db = gradient(dout, np.array(W, b))
	   self.params['W'] = W - self.A * X * (y - np.sum(X*W.T) - b )
	   self.params['B'] = b - self.A * (y - np.sum(X*W.T - b))  
def train(EPOCH):
   X_train = np.array([ [0,0],[0,1],[1,0],[1,1] ])
   Y_train = np.array([ 0,1,1,0 ])
   net = simpleNet(A=0.01)
   for epoch in range(1, EPOCH+1):
      for x, y in zip(X_train, Y_train):
        net.backward(x,y)
      print("EPOCH [%d/%d]" % (epoch, EPOCH))
   
   for x, y in zip(X_train, Y_train):
      output = net.predict(x)
      print(x,y)
   print("parameters")
   print(net.params['W'])
   print(net.params['B'])
if __name__ == '__main__':

 def f(X):
  return X[0]**2 + X[1]**2

 train(10)
 
