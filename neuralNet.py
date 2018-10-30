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

	def baseLayer(self):
		l1_filter = numpy.zeros((2,3,3))
		pass

	def hiddenLayer(self):
		pass

	def poolLayer(self):
		pass

	def convolve(self, data, kernel, stepSize = 1):
		# X is horizontal (columns), Y is vertical (rows)
		(yKernel, xKernel) = kernel.shape # kernel (0,0) is top left corner.
		xBound, yBound = int(math.floor(xKernel / 2)), int(math.floor(yKernel / 2))
		(yShape, xShape) = data.shape

		output = np.zeros((yShape - 2*yBound, xShape - 2*xBound))

		for y in range(yBound, yShape - yBound, stepSize): # Avoid invalid indexing when convolving by changing the range of the loop
			for x in range(xBound, xShape - xBound, stepSize):
				miniData = data[(y - yBound):(y + yBound + 1), (x - xBound):(x + xBound + 1)] # Get slice of data that we care about

				output[y - yBound, x - xBound] = sciConvolve(miniData, kernel, 'valid') # Convolve the slice of our data with the kernel, then output to corresponding pixel

				if(self.verbose):
					print "center xy: ", x,y
					print "data[", xBound, ":", xShape - xBound, ", ", yBound, ":", yShape - yBound, "]"
					print "slice:"
					print miniData
					print "convolution result:"
					print sciConvolve(miniData, kernel, 'valid')
		return output

	def activationFunction(self, value):
		pass