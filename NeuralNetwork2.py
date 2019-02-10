"""
This is my first try at creating a simple feed-forward neural network!
I used it to solve a simple problem : XOR:
		a xor b = True only if a != b

finished : 06:01:2019
created by: khaled tayachi

"""
import pickle
import numpy as np 
import random
import math
import time

def sigmoid(x):
	return 1 / (1 + math.exp(-x))

def dSigmoid(x):
	return x * (1 - x)

def tanh(x):
	return math.tanh(x)
def dtanh(x):
	return 1 - x**2


class NeuralNetwork:
	def __init__(self,  inputLayers, hiddenLayers, outputLayers, activation_Func, x_Train=None, y_Train=None, x_Test=None, y_Test=None, lr=None, derivative_activation_Func=None):
		self.x_Train = x_Train
		self.y_Train = y_Train
		self.x_Test = x_Test
		self.y_Test = y_Test
		self.nIn = inputLayers
		self.nH = hiddenLayers
		self.nO = outputLayers
		self.activation_Func = np.vectorize(activation_Func)
		self.derivative_activation_Func = np.vectorize(derivative_activation_Func)
		# Learning rate
		self.lr = lr
		self.Ws = []
		self.Bs = []
		# initializing the weights and biases and filling them with random numbers 
		self.Ws.append(np.random.rand(self.nH[0], self.nIn))
		self.Bs.append(np.random.rand(self.nH[0], 1))
		for i in range(len(self.nH) - 1):
			self.Ws.append(np.random.rand(self.nH[i + 1], self.nH[i]))
			self.Bs.append(np.random.rand(self.nH[i + 1], 1))
		self.Ws.append(np.random.rand(self.nO, self.nH[-1]))
		self.Bs.append(np.random.rand(self.nO, 1))

	def feed_forward(self, inputs):
		layers = [inputs]
		# calculating the output of each layer : y = w*x + b
		hidden = np.dot(self.Ws[0], inputs) + self.Bs[0]
		#print(hidden)
		hidden = self.activation_Func(hidden)
		layers.append(hidden)
		if len(self.Ws) > 2:
			for i in range(1, len(self.Ws) - 1):
				hidden = np.dot(self.Ws[i], hidden) + self.Bs[i]
				hidden = self.activation_Func(hidden)
				layers.append(hidden)
		outputs = np.dot(self.Ws[-1], hidden) + self.Bs[-1]
		outputs = self.activation_Func(outputs)
		layers.append(outputs)
		return layers

	def back_propagate(self, inputs, labels):
		layers = self.feed_forward(inputs)
		error = None
		#calculating the error and the derivative and using them to tweak the weights and biases
		for i in range(len(layers) - 1, 0, -1):
			if i == len(layers) - 1:
				error = labels - layers[i]
			else:
				error = np.dot(self.Ws[i].T, error)
			gradiant = error * self.derivative_activation_Func(layers[i])
			deltaWs = np.dot(gradiant, layers[i - 1].T)
			self.Ws[i - 1] += deltaWs
			self.Bs[i - 1] += gradiant

	def cost(self, inputs, labels):
		#calculating the squared error for each sample
		outputs = self.feed_forward(inputs)[-1]
		cost =  0.5 * np.sum((outputs - labels)**2)
		return cost

	def train(self, times):
		# the total error of all the samples
		totalCost = 0
		for i in range(1, times + 1):
			#randomizing the training data
			choice = random.choice(list(range(len(self.x_Train))))
			self.back_propagate(self.x_Train[choice], self.y_Train[choice])
			totalCost = ((totalCost * (i-1)) + self.cost(self.x_Train[choice], self.y_Train[choice])) / i
			# if i%100 == 0:
			# 	print(totalCost)

	def test(self):
		#testing how accurate the nn will do with test samples
		accuracy = 0
		for i in range(len(self.x_Test)):
			outputs = self.feed_forward(self.x_Test[i])[-1]
			if np.where(self.y_Test[i] == np.amax(self.y_Test[i])) == np.where(outputs == np.amax(outputs)):
				accuracy += 1
		accuracy = accuracy / len(self.x_Test)
		return accuracy * 100

	def guess(self, inputs):
		#to classify unknown data
		return self.feed_forward(inputs)

	def replace(self, nn):
		self.Ws = np.copy(nn.Ws)
		self.Bs = np.copy(nn.Bs)

	def mutate(self, mutationRate):
		def mutationFunc(x):
			if random.random() < mutationRate:
				return x + random.uniform(-1, 1) * 3 + 1
			else:
				return x
		mutation = np.vectorize(mutationFunc)
		for i in range(len(self.Ws)):
			self.Ws[i] = mutation(self.Ws[i])
			self.Bs[i] = mutation(self.Bs[i])

	def save(self):
		pickle_out = open("SmartBirdySan.pickle","wb")
		pickle.dump(self, pickle_out)
		pickle_out.close()




if __name__ == "__main__":

	inputs = [np.array([1, 1], ndmin=2).T, np.array([0, 1], ndmin=2).T, np.array([1, 0], ndmin=2).T, np.array([0, 0], ndmin=2).T]
	labels = [np.array([0, 1], ndmin=2).T, np.array([1, 0], ndmin=2).T, np.array([1, 0], ndmin=2).T, np.array([0, 1], ndmin=2).T]
	#the hidden layers are entered in a list that contains the number of nodes in each layer
	nn = NeuralNetwork(2, [4], 2, sigmoid, inputs, labels, inputs, labels, 0.1, dSigmoid)
	a = time.time()
	nn.train(100000)
	b = time.time()
	print("It took the neural network ", b - a, "seconds to train")
	print("The accuracy is : ", nn.test())
	print("* TRUE  TRUE")
	print("   Guess: \n   ", str(nn.guess(inputs[0])[-1][0][0] * 100)[:4] + "% :" , "TRUE \n   ", str(nn.guess(inputs[0])[-1][1][0] * 100)[:4] + "% :"  , "FALSE")
	print("* FALSE  TRUE")
	print("   Guess: \n   ", str(nn.guess(inputs[1])[-1][0][0] * 100)[:4] + "% :" , "TRUE \n   ", str(nn.guess(inputs[1])[-1][1][0] * 100)[:4] + "% :"  , "FALSE")
	print("* TRUE  FALSE")
	print("   Guess: \n   ", str(nn.guess(inputs[2])[-1][0][0] * 100)[:4] + "% :" , "TRUE \n   ", str(nn.guess(inputs[2])[-1][1][0] * 100)[:4] + "% :"  , "FALSE")
	print("* FALSE  FALSE")
	print("   Guess: \n   ", str(nn.guess(inputs[3])[-1][0][0] * 100)[:4] + "% :" , "TRUE \n   ", str(nn.guess(inputs[3])[-1][1][0] * 100)[:4] + "% :"  , "FALSE")


	nn1 = nn.copy()
	print(nn1.test())
