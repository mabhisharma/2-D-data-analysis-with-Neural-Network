#########################
#Author : Abhishek Sharma
#########################
import numpy as np

class NeuralNet(object):
	"""docstring for NeuralNet"""
	def __init__(self, numberOfLayers, numberOfUnits):
		super(NeuralNet, self).__init__()
		self.numberOfLayers = numberOfLayers
		self.parameters = {}
		self.Z = {}
		self.A = {}
		self.grads = {}
		self.initialiseParameters(numberOfUnits)


	def initialiseParameters(self, L):
		np.random.seed(3)
		for i in range(1, self.numberOfLayers+1):
			self.parameters["W" + str(i)] = np.random.randn(L[i], L[i-1])*0.1
			self.parameters["b" + str(i)] = np.zeros((L[i], 1))

			
	def feedForward(self, X):
		self.A[0] = X
		for i in range(1, self.numberOfLayers):
			self.Z[i] = np.dot( self.parameters[ "W" + str(i) ],self.A[i-1])\
											+ self.parameters[ "b" + str(i) ]
			
			self.A[i] = self.activation(self.Z[i], type="tanh") 
		
		
		self.Z[i+1] = np.dot( self.parameters[ "W" + str(i+1) ],self.A[i])\
										 + self.parameters[ "b" + str(i+1) ]
		
		self.A[i+1] = self.activation(self.Z[i+1], type="sigmoid")

	def activation(self, Z, type):
		if type == "tanh":
			return np.tanh(Z)

		elif type == "relu":
			return np.maximum(0,Z)
		
		else:
			return 1./(1.+np.exp(-Z))

	def calculateCost(self, Y):
		m = Y.shape[1]
		
		logprobs = np.multiply(np.log(self.A[self.numberOfLayers]),Y)\
				 + np.multiply(np.log(1-self.A[self.numberOfLayers]),1-Y)
		
		self.cost = (-1./m)*np.sum(logprobs)


	def backactivation(self, m, W, A_prev, dA, Z, type):
		
		if type == "tanh":
			dZ = dA*(1- np.power(self.activation(Z, type="tanh"),2))

		else :
			s = 1/(1+np.exp(-Z))
			dZ = dA * s * (1-s)

		dW = 1./m*np.dot(dZ, A_prev.T)
		db = 1./m*np.sum(dZ, axis =1 , keepdims=True)
		dA_prev = np.dot(W.T,dZ)

		return dW, db, dA_prev

	def backpropogation(self,Y, AL):
		L = self.numberOfLayers
		m = Y.shape[1]
		dAL = - (np.divide(Y, self.A[L]) - np.divide(1 - Y, 1 - self.A[L]))
		
		dW, db , dA_prev =self.backactivation(m,self.parameters['W'+str(L)], 
												self.A[L-1], dAL, self.Z[L], 
															type='sigmoid')

		self.grads["dW" + str(L)] ,self.grads["db" + str(L)] = dW, db

		for i in reversed(range(1, self.numberOfLayers)):
			dW,db , dA_prev = self.backactivation(m, 
												self.parameters['W'+str(i)], 
												self.A[i-1], dA_prev, 
												self.Z[i],type='tanh')

			self.grads["dW" + str(i)] ,self.grads["db" + str(i)] = dW, db


	def updateParameters(self, learning_rate):
		L = self.numberOfLayers

		for l in range(0,L):
			self.parameters['W'+str(l+1)] -= learning_rate*self.grads['dW'+str(l+1)]
			self.parameters['b'+str(l+1)] -= learning_rate*self.grads['db'+str(l+1)]



	def gradientDescent(self, X, noOfIterations, learningRate, Y, printCost=False):
		AL = []
		WL = []
		bL = []
		for i in range(noOfIterations):
			self.feedForward(X)
			self.calculateCost(Y)

			if printCost and i % 100 == 0:
				print ("Cost after iteration %i: %f" %(i, self.cost))
				AL.append(self.A[self.numberOfLayers-1])
				WL.append(self.parameters["W"+str(self.numberOfLayers)])
				bL.append(self.parameters["b"+str(self.numberOfLayers)])
			
			self.backpropogation(Y, self.A[self.numberOfLayers])
			self.updateParameters(learningRate)

		return AL, WL, bL

	def predict(self, X):
		self.feedForward(X)
		predictions = (self.A[self.numberOfLayers] > 0.5)
		return predictions