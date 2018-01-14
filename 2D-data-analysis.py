import os
import numpy as np 
import imageio
import sklearn.datasets
import matplotlib.pyplot as plt
from neuralnet import NeuralNet
from mpl_toolkits.mplot3d import Axes3D
import imageio


images = []
def create2DData():
	np.random.seed(3)
	X, Y = sklearn.datasets.make_circles(n_samples=200, factor=.5, noise=0)
	X, Y = X.T, Y.reshape(1, Y.shape[0])
	return X, Y

def plot2DData(X, Y):
	A = X[0, :].reshape(1,-1)
	B = X[1, :].reshape(1,-1)
	plt.scatter(A[Y==0], B[Y==0], c='r', label="Class A")
	plt.scatter(A[Y==1], B[Y==1], c='b', label="Class B")
	plt.legend()
	plt.xlabel('Feature Vector 1')
	plt.ylabel('Feature Vector 2')
	plt.title('Two Dimensional data')
	plt.savefig(os.path.join('Plots','Two-Dimensional-data.png'))
	plt.show(block=False)

def plot3DData(data, w, b, count, Y):
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	A = data[0, :].reshape(1, -1)
	B = data[1, :].reshape(1, -1)
	C = data[2, :].reshape(1, -1)
	ax.plot(A[Y==0], B[Y==0], C[Y==0], '*', label="Class A")
	ax.plot(A[Y==1], B[Y==1], C[Y==1], '.', label="Class B")
	ax.legend()
	ax.set_xlabel("Output from 1st unit of Hidden Layer")
	ax.set_ylabel("Output from 2nd unit of Hidden Layer")
	ax.set_zlabel("Output from 3rd unit of Hidden Layer")
	ax.set_title("Two Two-Dimensional-data")
	x = np.linspace(-1, 1, 100)
	y = np.linspace(-1, 1, 100)
	X1, Y1 = np.meshgrid(x, y)
	Z = (-b -w[0,0]*X1 -w[0,1]*Y1)/w[0,2]
	surf = ax.plot_surface(X1, Y1, Z, label="Decision Plane",linewidth=0, antialiased=False)
	for angle in range(0, 360, 15):
		ax.view_init(30, angle)
		plt.draw()
		plt.savefig(os.path.join('Plots','gifs',str(count)+str(angle)+'.png'))
		plt.pause(.01)

def plot3D(data, w, b, count, Y):
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	A = data[0, :].reshape(1, -1)
	B = data[1, :].reshape(1, -1)
	C = data[2, :].reshape(1, -1)
	ax.plot(A[Y==0], B[Y==0], C[Y==0], '*', label="Class A")
	ax.plot(A[Y==1], B[Y==1], C[Y==1], '.', label="Class B")
	ax.set_xlabel("Output from 1st unit of Hidden Layer")
	ax.set_ylabel("Output from 2nd unit of Hidden Layer")
	ax.set_zlabel("Output from 3rd unit of Hidden Layer")
	ax.set_title("Two Two-Dimensional-data")
	ax.legend()
	x = np.linspace(-1, 1, 100)
	y = np.linspace(-1, 1, 100)
	X1, Y1 = np.meshgrid(x, y)
	Z = (-b -w[0,0]*X1 -w[0,1]*Y1)/w[0,2]
	surf = ax.plot_surface(X1, Y1, Z, label="Decision Plane",linewidth=0, antialiased=False)
	ax.view_init(30, 30)
	plt.show()
	plt.savefig(os.path.join('Plots',str(count)+'.png'))
	images.append(imageio.imread(os.path.join('Plots',str(count)+'.png')))



def plotTransformedData(A, W, b, Y):
	for count,(w,b, data) in enumerate(zip(W,b, A)):
		plot3D(data, w, b, count ,Y)
	
	imageio.mimsave('3-ddata-tranformation.gif', images, duration=0.3)

def main():
	X, Y = create2DData()
	plot2DData(X,Y)
	noOfLayers = 2 # Hidden and Output layer (Excluding the input layer)
	layerDimensions = [2, 3, 1] # No of units in Input, Hidden, Output layer
	noOfIterations = 6000
	learningRate = 0.6
	N = NeuralNet(noOfLayers, layerDimensions) # Create a object of Neural Net
	AL, WL, bL = N.gradientDescent(X, noOfIterations, learningRate, Y, printCost=True)
	# plotTransformedData(AL, WL, bL, Y)

if __name__ == '__main__':
	main()