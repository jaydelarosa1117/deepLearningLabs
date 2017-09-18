import numpy as np
import random


def fileToMatrix(fileName):
    return np.loadtxt(fileName, delimiter=',')


def sumListConts(L1, L2):
	L = []
	L.append(L1[0]+L2[0])
	L.append(L1[1]+L2[1])
	L.append(L1[2]+L2[2])
	L.append(L1[3]+L2[3])
	L.append(L1[4]+L2[4])
	L.append(L1[5]+L2[5])
	return L

def randWeights(s):
	L = []
	L.append((np.random.rand(784,512)-.5)*s)
	L.append((np.random.rand(512,512)-.5)*s)
	L.append((np.random.rand(512,10)-.5)*s)
	L.append((np.random.rand(512)-.5)*s)
	L.append((np.random.rand(512)-.5)*s)
	L.append((np.random.rand(10)-.5)*s)
	return L	

def classify(W, subsetX):
	h0 = np.matmul(subsetX, W[0])
	h0 += W[3]
	h0 = relu(h0)

	h1 = np.matmul(h0,W[1])
	h1 += W[4]
	h1 = relu(h1)

	out = np.matmul(h1, W[2])
	out += W[5]
	out = sigmoid(out)
	
	return out

#rectifies the input matrix
def relu(A):
    A[A < 0] = 0
    return A

def sigmoid(A):
	return  1/(1+(np.exp(-A))) 

def findMaxIndex(A):
    return A.argmax(1)

def logit(A):
	return np.log(A) - np.log(1-A)
		
def hotEncoder(yOut):
	enc = np.zeros(shape=(yOut.size,10))
	for i in range(yOut.size):
		enc[i][int(yOut[i])] = 1
	return enc




def findWeightsRandom(X, Y):
	Y = hotEncoder(Y) 
	wBest = randWeights(0.2)
	for i in range(5000):
		print(i)
		subsetSize = random.randint(50,1000)
		randRows= np.random.randint(X.shape[0], size=subsetSize)
		subsetX = X[randRows,:]/255.0
		subsetY = Y[randRows,:]

		W = sumListConts(wBest,randWeights(.01))

		e0mat = (subsetY - classify(W,subsetX))
		e0 = np.sum(e0mat**2)

		e1mat = (subsetY - classify(wBest,subsetX))
		e1 = np.sum(e1mat**2)
		
		if e0 < e1:
			wBest[0] = W[0]
			wBest[1] = W[1]
			wBest[2] = W[2]
			wBest[3] = W[3]
			wBest[4] = W[4]
			wBest[5] = W[5]
	return wBest




def findWeightsPseudoinverse(X, Y):
	Y=hotEncoder(Y)*.9+.05
	W = randWeights(.2)
	
	h0 = relu(np.matmul(X,W[0])+W[3])
	h1 = relu(np.matmul(h0, W[1])+W[4])
	
	Yn = logit(Y)
	
	W[5] = np.mean(Yn,axis=0)
	W[2] = np.matmul(np.linalg.pinv(h1),(Yn-W[5]))
	return W




def findWeightsBackprop(X,Y):
	W = randWeights(0.2)
	Y = hotEncoder(Y)
	for i in range(1000):
		print(i)
		subsetSize = random.randint(50,1000)
		randRows= np.random.randint(X.shape[0], size=subsetSize)
		subsetX = X[randRows,:]/255.0
		subsetY = Y[randRows]
		
		h0 = relu(np.matmul(subsetX,W[0])+W[3])
		h1 = relu(np.matmul(h0,W[1])+W[4])
		P = sigmoid(np.matmul(h1, W[2]) + W[5])
		
		dP = (P-subsetY)*P*(1-P)
		
		dH1 = (np.matmul(dP,np.transpose(W[2]))) * np.sign(h1)
		dH0 = (np.matmul(dH1,np.transpose(W[1]))) * np.sign(h0)
		
		lda = 0.001
		W[2] = W[2] - lda*(np.matmul(np.transpose(h1),dP))
		W[1] = W[1] - lda*(np.matmul(np.transpose(h0),dH1))
		W[0] = W[0] - lda*(np.matmul(np.transpose(subsetX),dH0))
		W[5] = W[5] - lda*(np.sum(dP,axis=0))
		W[4] = W[4] - lda*(np.sum(dH1,axis=0))
		W[3] = W[3] - lda*(np.sum(dH0,axis=0))
	return W
		
		
		

def alg4(X, Y):
	
	W = findWeightsPseudoinverse(X,Y)
#	W = randWeights(.2)
	Y = hotEncoder(Y)
	for i in range(1000):
		print(i)
		subsetSize = random.randint(50,1000)
		randRows= np.random.randint(X.shape[0], size=subsetSize)
		subsetX = X[randRows,:]/255.0
		subsetY = Y[randRows]
		
		h0 = relu(np.matmul(subsetX,W[0])+W[3])
		h1 = relu(np.matmul(h0,W[1])+W[4])
		P = sigmoid(np.matmul(h1, W[2]) + W[5])
		
		dP = (P-subsetY)*P*(1-P)
		
		dH1 = (np.matmul(dP,np.transpose(W[2]))) * np.sign(h1)
		dH0 = (np.matmul(dH1,np.transpose(W[1]))) * np.sign(h0)
		
		lda = 0.01
		W[2] = W[2] - (np.matmul(lda*np.transpose(h1),dP))
		W[1] = W[1] - (np.matmul(lda*np.transpose(h0),dH1))
		W[0] = W[0] - (np.matmul(lda*np.transpose(subsetX),dH0))
		W[5] = W[5] - lda*(np.sum(dP,axis=0))
		W[4] = W[4] - lda*(np.sum(dH1,axis=0))
		W[3] = W[3] - lda*(np.sum(dH0,axis=0))
	return W




def noHLayerTrain(X,Y):
	Y = hotEncoder(Y)
	W = ((np.random.rand(784,10)-.5)*.2)
	B = ((np.random.rand(10)-.5)*.2)
	
	lda = .00001
	for i in range(300):
		subsetSize = random.randint(50,1000)
		randRows= np.random.randint(X.shape[0], size=subsetSize)
		subsetX = X[randRows,:]/255.0
		subsetY = Y[randRows ]
		
		
		P = np.matmul(subsetX,W) + B
		
		W = W - (lda * (np.matmul(np.transpose(subsetX),(P-subsetY))))
		B = B - np.sum(lda * ((P-subsetY)),axis=0)
		print(W)
		print(B)
		
	X = np.load("xtest.npy")
	Y = np.load("ytest.npy")
	P = np.matmul(X,W) + B
	print(P)
	out = findMaxIndex(P)
	count = 0
	for i in range(out.size):
		if out[i] == Y[i]:
			count += 1
	print("the output is ",count)
	
	
	
	
	return W

def main():
#	X = fileToMatrix("xtrain.txt")
	Y = np.load("ytrain.npy")
	X = np.load("xtrain.npy")
#	Y = np.load("ytrain.npy")

	
	print("done Reading ")
	W =	 findWeightsRandom(X,Y)
#	W = findWeightsPseudoinverse(X,Y)
#	W = findWeightsBackprop(X,Y)
#	W = alg4(X,Y)
#	noHLayerTrain(X, Y)
	X = np.load("xtest.npy")
	Y = np.load("ytest.npy")
	count = 0
	out = findMaxIndex(classify(W,X))
	for i in range(out.size):
		if out[i] == Y[i]:
			count += 1
	print("the output is ",count*100/10000)
	
main()
	


	
