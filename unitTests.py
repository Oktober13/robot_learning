#!/usr/bin/env python

import argparse
import numpy as np

from cvDistance import CVDistance
from neuralNet import NeuralNet
from rosBoss import RosBoss

class UnitTests(object):
	def __init__(self):
		convPass = self.convolutionTest()
		activationTest = self.activationTest()
		maxpoolPass = self.maxpoolTest()
<<<<<<< HEAD
		trainTest = self.trainTest()
=======
>>>>>>> 2948b14d3efcfd00fd87931ba827ad243d931c87

		if (convPass and maxpoolPass and activationTest):
			print "All tests passed! :)"
		if not convPass:
			print "convolutionTest unit test failed!"
		if not activationTest:
			print "activationTest unit test failed!"
		if not maxpoolPass:
			print "maxpoolTest unit test failed!"
<<<<<<< HEAD
		trainTest
=======
>>>>>>> 2948b14d3efcfd00fd87931ba827ad243d931c87

	def trainTest(self, dataFilepath = None):
		nn = NeuralNet()

		nn.train()
		return

	def convolutionTest(self):
		nn = NeuralNet()
		expected = np.array([[-1,0], [-3,-11], [-5,-17]])

		# Convolve the vertical Sobel edge detector
		data = np.array([[1,1,1,0], [0,1,0,0], [2,1,3,4], [5,1,6,7], [8,9,10,11]])
		kernel = np.array([[1,0,-1], [2,0,-2], [1,0,-1]])
		output = nn.convolve(data, kernel)

		return self.arrayIsSame(output, expected)
<<<<<<< HEAD

	def activationTest(self):
		nn = NeuralNet()

		data = np.array([[1,0,0,0], [1,1,1,0], [1,0,0,0], [1,0,0,0], [1,0,0,1]])
		kernel = np.array([[1,0,-1], [2,0,-2], [1,0,-1]])
		output = nn.convolve(data, kernel)

		expectedRelu = np.array([[2,2], [3,1], [4,0]])
		expectedBin = np.array([[1,1], [1,1], [1,0]])
		expectedSigmoid = np.array([[0.119,0.119], [0.047,0.269], [0.0180,0.731]])
		expectedTanH = np.array([[0.964,0.964], [0.995,0.762], [0.999,-0.762]])
		expected = np.array([[-1,0], [-3,-11], [-5,-17]])

		reluOutput = nn.activationFunction(output, 'relu')
		binOutput = nn.activationFunction(output, 'binary')
		sigmoidOutput = nn.activationFunction(output, 'sigmoid')
		tanhOutput = nn.activationFunction(output, 'tanh')

		for index, value in np.ndenumerate(sigmoidOutput): sigmoidOutput[index] = round(value, 3)
		for index, value in np.ndenumerate(tanhOutput): tanhOutput[index] = round(value, 3)

		if not UnitTests.arrayIsSame(reluOutput, expectedRelu):
			print "Relu activationFunction test failed."
			print "Expected:"
			print expectedRelu
			print "Actual:"
			print reluOutput
			return False
		if not UnitTests.arrayIsSame(binOutput, expectedBin):
			print "Binary activationFunction test failed."
			print "Expected:"
			print expectedBin
			print "Actual:"
			print binOutput
			return False
		if not UnitTests.arrayIsSame(sigmoidOutput, expectedSigmoid):
			print "Sigmoid activationFunction test failed."
			print "Expected:"
			print expectedSigmoid
			print "Actual:"
			print sigmoidOutput
			return False
		if not UnitTests.arrayIsSame(tanhOutput, expectedTanH):
			print "tanh activationFunction test failed."
			print "Expected:"
			print expectedTanH
			print "Actual:"
			print tanhOutput
			return False

		return True

	def maxpoolTest(self):
		nn = NeuralNet()

		# Should fail - Wrong size array
		expected = np.array([[4,4], [0,4]])

		data = np.array([[4,0,1,3,4], [0,0,2,4,4], [0,0,4,4,4], [0,0,4,4,4]])
		windowsize = 2 # 2x2 window

		output = nn.maxPool(data, windowsize)

		if output is not None:
			return False

		# Should pass
		expected = np.array([[4,4], [0,4]])

		data = np.array([[4,0,1,3], [0,0,2,4], [0,0,4,4], [0,0,4,4]])
		windowsize = 2 # 2x2 window

		output = nn.maxPool(data, windowsize)

		return self.arrayIsSame(output,expected)

=======

	def activationTest(self):
		nn = NeuralNet()

		data = np.array([[1,0,0,0], [1,1,1,0], [1,0,0,0], [1,0,0,0], [1,0,0,1]])
		kernel = np.array([[1,0,-1], [2,0,-2], [1,0,-1]])
		output = nn.convolve(data, kernel)

		expectedRelu = np.array([[2,2], [3,1], [4,0]])
		expectedBin = np.array([[1,1], [1,1], [1,0]])
		expectedSigmoid = np.array([[0.119,0.119], [0.047,0.269], [0.0180,0.731]])
		expectedTanH = np.array([[0.964,0.964], [0.995,0.762], [0.999,-0.762]])
		expected = np.array([[-1,0], [-3,-11], [-5,-17]])

		reluOutput = nn.activationFunction(output, 'relu')
		binOutput = nn.activationFunction(output, 'binary')
		sigmoidOutput = nn.activationFunction(output, 'sigmoid')
		tanhOutput = nn.activationFunction(output, 'tanh')

		for index, value in np.ndenumerate(sigmoidOutput): sigmoidOutput[index] = round(value, 3)
		for index, value in np.ndenumerate(tanhOutput): tanhOutput[index] = round(value, 3)

		if not UnitTests.arrayIsSame(reluOutput, expectedRelu):
			print "Relu activationFunction test failed."
			print "Expected:"
			print expectedRelu
			print "Actual:"
			print reluOutput
			return False
		if not UnitTests.arrayIsSame(binOutput, expectedBin):
			print "Binary activationFunction test failed."
			print "Expected:"
			print expectedBin
			print "Actual:"
			print binOutput
			return False
		if not UnitTests.arrayIsSame(sigmoidOutput, expectedSigmoid):
			print "Sigmoid activationFunction test failed."
			print "Expected:"
			print expectedSigmoid
			print "Actual:"
			print sigmoidOutput
			return False
		if not UnitTests.arrayIsSame(tanhOutput, expectedTanH):
			print "tanh activationFunction test failed."
			print "Expected:"
			print expectedTanH
			print "Actual:"
			print tanhOutput
			return False

		return True

	def maxpoolTest(self):
		nn = NeuralNet()

		# Should fail - Wrong size array
		expected = np.array([[4,4], [0,4]])

		data = np.array([[4,0,1,3,4], [0,0,2,4,4], [0,0,4,4,4], [0,0,4,4,4]])
		windowsize = 2 # 2x2 window

		output = nn.maxPool(data, windowsize)

		if output is not None:
			return False

		# Should pass
		expected = np.array([[4,4], [0,4]])

		data = np.array([[4,0,1,3], [0,0,2,4], [0,0,4,4], [0,0,4,4]])
		windowsize = 2 # 2x2 window

		output = nn.maxPool(data, windowsize)

		return self.arrayIsSame(output,expected)

>>>>>>> 2948b14d3efcfd00fd87931ba827ad243d931c87
	@staticmethod
	def arrayIsSame(output, expected):
		if(not np.array_equal(output, expected)):
			return False
		else:
			return True

if __name__ == "__main__":
	UnitTests()