import numpy as np
import math
import scipy.misc as scp
from sklearn.metrics import confusion_matrix

def fileToMatrix(fileName):
	return np.loadtxt(fileName, delimiter=',')



def setRange(A):
	A = A/255
	return A


#rectifies the input matrix
def relu(A):
	A[A < 0] = 0
	return A



#finds the sigmoid of the layers of the vector
def sigmoid(A):
	for i in range(A.size):
		A[i] = 1/(1+(math.exp(-A[i]))) 
	return A



#recieves array of size 10 and returns the index of the larges value
def findMaxIndex(A):
	return A.argmax(1)


def mismatches(o, y):
	mismatches = np.array([])
	i = 0
	for out, actual in zip(o,y):
		if not (out == actual):
			mismatches = np.append(mismatches, i)
		i += 1
	return mismatches
		
def displayMismatch(img):
	img = img.reshape(28,28)
	img.shape
	img = img * 255
	scp.imshow(img)


def main():
	xinput = fileToMatrix("xtest.txt")
	xinput = setRange(xinput)
	
	try:
		xinput.shape[1]
	except:
		xinput = xinput.reshape([1,xinput.size])
		
	
	W0 = fileToMatrix("W0.txt")
	B0 = fileToMatrix("B0.txt")
	W1 = fileToMatrix("W1.txt")
	B1 = fileToMatrix("B1.txt")
	W2 = fileToMatrix("W2.txt")
	B2 = fileToMatrix("B2.txt")
	
	
	h0 = np.matmul(xinput, W0)
	h0 += B0
	h0 = relu(h0)
	
	
	
	h1 = np.matmul(h0,W1)
	h1 += B1
	h1 = relu(h1)
	

	out = np.matmul(h1, W2)
	out += B2
	for i, o in enumerate(out):
		out[i] = sigmoid(o)
	
	actualOut = findMaxIndex(out)


	yOutput = fileToMatrix("ytest.txt")
	mismatchedList = mismatches(actualOut, yOutput)
	
	print(mismatchedList)
	print(mismatchedList.size)	 
	confMat = confusion_matrix(yOutput,actualOut)
	print(confMat)
	
main()





	