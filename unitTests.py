#!/usr/bin/env python

import argparse
import numpy as np

from cvDistance import CVDistance
from neuralNet import NeuralNet
from rosBoss import RosBoss

class UnitTests(object):
	def __init__(self):
		convPass = self.convolutionTest()
		maxpoolPass = self.maxpoolTest()

		if (convPass and maxpoolPass):
			print "All tests passed! :)"
		if not convPass:
			print "convolutionTest unit test failed!"
		if not maxpoolPass:
			print "maxpoolTest unit test failed!"


	def convolutionTest(self):
		nn = NeuralNet()
		expected = np.array([[-1,0], [-3,-11], [-5,-17]])

		# Convolve the vertical Sobel edge detector
		data = np.array([[1,1,1,0], [0,1,0,0], [2,1,3,4], [5,1,6,7], [8,9,10,11]])
		kernel = np.array([[1,0,-1], [2,0,-2], [1,0,-1]])
		output = nn.convolve(data, kernel)

		return self.arrayIsSame(output, expected)

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

	@staticmethod
	def arrayIsSame(output, expected):
		if(not np.array_equal(output, expected)):
			return False
		else:
			return True

if __name__ == "__main__":
	UnitTests()