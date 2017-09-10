import random as rand
import numpy as np

def fileToMatrix(fileName):
	return np.loadtxt(fileName,delimiter=',')

def main():
	x = fileToMatrix("xtrain.txt")
	print(x.shape)
main()