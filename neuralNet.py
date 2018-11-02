#!/usr/bin/env python

import math
import numpy as np
import rospy
from scipy.signal import convolve2d as sciConvolve

from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

class NeuralNet(object):
	def __init__(self, weights = None):
		if weights != None:
			self.weights = weights
			self.training = False
		else:
			self.training = True

		# self.output = np.zeros(y.shape)
		self.verbose = False

	def maxPool(self, data, windowSize):
		"""
		Force the network to focus on a few neurons instead of all of them. (Downsampling)
		Saves processing time/power and makes the network less likely to overfit.
		"""
		(yShape, xShape) = data.shape

		if (xShape % windowSize is 0) and (yShape % windowSize is 0):
			output = np.zeros((int(yShape / windowSize), int(xShape / windowSize)))

			for y in range(0, yShape, windowSize):
				for x in range(0, xShape, windowSize):
					output[int(y / windowSize), int(x / windowSize)] = np.amax(data[y:(y+windowSize),x:(x+windowSize)])
			return output
		else:
			if self.verbose: print "Data size (", xShape, ",", yShape, ") not divisible by window size (", windowSize, ",", windowSize, ")  :("
			return None

	def convolve(self, data, kernel, stepSize = 1):
		"""
		Bring out certain aspects of the data by convolving it with kernels.
		"""
		# X is horizontal (columns), Y is vertical (rows)
		(yKernel, xKernel) = kernel.shape # kernel (0,0) is top left corner.
		if not ((xKernel % 2 == 0) and (yKernel % 2 == 0)):

			yBound, xBound = int(math.floor(yKernel / 2)), int(math.floor(xKernel / 2))
			(yShape, xShape) = data.shape

			output = np.zeros((yShape - 2*yBound, xShape - 2*xBound))

			for y in range(yBound, yShape - yBound, stepSize): # Avoid invalid indexing when convolving by changing the range of the loop
				for x in range(xBound, xShape - xBound, stepSize):
					miniData = data[(y - yBound):(y + yBound + 1), (x - xBound):(x + xBound + 1)] # Get slice of data that we care about

					""" 
					Convolve the slice of our data with the kernel, then output to corresponding pixel.
					We flip the kernel to abide by convention, so that we get the expected (elementwise) answer from scipy.
					See https://www.mathworks.com/matlabcentral/answers/74274-why-do-we-need-to-flip-the-kernel-in-2d-convolution
					If you have numpy version v 1.12.0 or above, you should instead use np.flip(kernel,1).
					"""
					output[y - yBound, x - xBound] = sciConvolve(miniData, np.fliplr(kernel), 'valid')

					if(self.verbose):
						print "center xy: ", x,y
						print "data[", xBound, ":", xShape - xBound, ", ", yBound, ":", yShape - yBound, "]"
						print "slice:"
						print miniData
						print "kernel: "
						print kernel
						print "convolution result:"
						print sciConvolve(miniData, kernel, 'valid')
			return output
		else: 
			print "Kernel has even dimension."
			return None

	def activationFunction(self, value):
		pass