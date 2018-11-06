#!/usr/bin/env python

import cv2
import math
import numpy as np
import PIL.Image
import os
import rospkg
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

		self.verbose = False

		self.lowerBound = (200,80,220) # 205,85,227 is BGR of post it
		self.upperBound = (210,90,235)

	def retrieveData(self, dataFilepath, weightFilepath):
		rosPack = rospkg.RosPack()
		pkgRoot = rosPack.get_path('robot_learning') # Gets the package

		### Retrieve data

		if dataFilepath is None: # If no data directory filepath is given
			dataDirectory = os.path.join(pkgRoot, "data") # Create filepath for data directory

			if not (os.path.isdir(dataDirectory)) or (len(os.listdir(dataDirectory)) == 0): # If data directory does not exist or is empty
				print "No image data provided for training. Please specify filepath to data or place images in the /data directory."
				if not (os.path.isdir(dataDirectory)): os.makedirs(dataDirectory)
				return None, None
		else: # Use given data directory
			dataDirectory = dataFilepath

		data = []

		# Convert all image files in the given data directory filepath into np matrices
		for filename in os.listdir(dataDirectory):
			if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
				fp = open(os.path.join(dataDirectory, filename))
				img = PIL.Image.open(fp).convert('RGBA') # Convert .jpg to RGBA PIL image

				opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) # Convert PIL RGBA to OpenCV BGR
				mask = cv2.inRange(opencvImage, self.lowerBound, self.upperBound)
				mask = cv2.erode(mask, None, iterations=2)
				mask = cv2.dilate(mask, None, iterations=2)
				data.append(mask)
		data = np.asarray(data)

		print "data size: " + str(data.shape)

		### Retrieve weights

		if weightFilepath is None:
			weightsDirectory = os.path.join(pkgRoot, "weights") # Create filepath for weight directory
			weightsPath = os.path.join(weightsDirectory, "weights.txt") # Create filepath for weights file
			if not (os.path.isdir(weightsDirectory)): os.makedirs(weightsDirectory) # If data directory does not exist or is empty
			if not (os.path.isfile(weightsPath)): # If no weights file exists
				file = open("weights.txt", "w")
				(numImg, h, l) = data.shape
				weights = np.random.rand(h,l)
				file.write(weights)
				file.close()
			else: # Set weights equal to contents of weights.txt
				file = open(os.path.join(weightsPath, "weights.txt"), "r")
				weights = file.read()
				file.close()
		else:
			weightsPath = weightFilepath

		return data, weights

	def train(self, dataFilepath = None, weightFilepath = None):
		trainImg, seedWeights = self.retrieveData(dataFilepath, weightFilepath)
		print seedWeights.shape

		for epoch in range(1,11):
			print self.fullNetwork(trainImg)
		return

	def fullNetwork(self, data):
		# Normally, weights in kernel would be dynamically adusted by network in backpropagation step.
		# For now, we'll just use the Sobel kernel for proof of concept.
		weights = np.array([[1,0,-1], [2,0,-2], [1,0,-1]])

		### DOWNSAMPLING
		# Downsample to simplify data (640,480) to (64, 48)
		downsample = self.maxPool(data[1,:,:], 10)
		# print downsample.shape

		### HIDDEN LAYER
		hidden_layer_input = self.convolve(downsample,weights) # Bias is set to zero right now
		hidden_layer_activations = self.activationFunction(hidden_layer_input)
		# print hidden_layer_activations.shape
		hidden_layer_output = np.reshape(-1,1)

		### OUTPUT LAYER
		y,x = hidden_layer_activations.shape
		# Each weight/output contributes a certain amount to the overall distance estimate. Currently trying to generate small random values
		output_weights = 0.001 * np.random.rand(y*x)
		output_layer_input = np.multiply(hidden_layer_output, output_weights)
		output = np.sum(output_layer_input)

		actual = 4 # No actual data right now; need to match LIDAR data timestamp with photo timestamps
		error = actual - output # Distance.
		return error

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

	def activationFunction(self, data, function = 'relu'):
		"""
		Function that acts on the output array to eliminate fuzz and generally clean up the output.
		"""
		output = np.zeros(data.shape)

		for index, value in np.ndenumerate(data):
			output[index] = value

			if function is 'relu':
				if value <= 0:
					output[index] = 0
			elif function is 'binary':
				if value <= 0:
					output[index] = 0
				else:
					output[index] = 1
			elif function is 'sigmoid':
				output[index] = 1.0 / float(1 + math.e ** (value))
			elif function is 'tanh':
				output[index] = (2.0 / (1 + math.e ** (-2 * value))) - 1.0
			else:
				print "Entered activation function is not supported at this time."
				return None

		return output

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
