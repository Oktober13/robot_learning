#!/usr/bin/env python

import argparse
import numpy as np

from cvDistance import CVDistance
from neuralNet import NeuralNet
from rosBoss import RosBoss

class UnitTests(object):
	def __init__(self):
		convPass = self.convolutionTest()

		if (convPass):
			print "All tests passed! :)"
		elif not convPass:
			print "convolutionTest unit test failed!"


	def convolutionTest(self):
		nn = NeuralNet()
		expected = np.array([[-1,0], [-3,-11], [-5,-17]])

		# Convolve the vertical Sobel edge detector
		data = np.array([[1,1,1,0], [0,1,0,0], [2,1,3,4], [5,1,6,7], [8,9,10,11]])
		kernel = np.array([[1,0,-1], [2,0,-2], [1,0,-1]])
		output = nn.convolve(data, kernel)

		# print output

		if(output.all() != expected.all()):
			return False
		else:
			return True

if __name__ == "__main__":
	UnitTests()