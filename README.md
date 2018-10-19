# robot_learning

This repository contains code written for Computational Robotics, Fall 2018, by Lydia Zuehsow.

A video stream is collected from a Neato. A before and after image is captured as the Neato drives towards the object, which can then be fed (along with the odometry data) as a training data set to a homebrew neural network that will output the actual bounding box size and distance of the object from the Neato. The data will be compared against a testing dataset of the actual object size (static) and its distance from the Neato as measured by the Neato's LIDAR scan as it drives.

This code will implement my own, homebrewed convolutional neural network, of undetermined layer number. Preexisting toolkits will not be used. I plan to collect data by driving the Neato at various objects and collecting a short (203 second) rosbag of video, which I will then preprocess. I plan to compare the results of my neural network to [the triangle similarity method described here](https://www.pyimagesearch.com/2015/01/19/find-distance-camera-objectmarker-using-python-opencv/), which I will be using as a baseline for accuracy in distance estimation to objects of known size.

Minimum Viable Product: Homebrew neural network is able to estimate depth of an object of known size and color. Neato drives directly perpendicular towards the object.

Stretch Goal: Homebrew neural network is able to distinguish objects from their background using the parallax effect, and is able to use the change in size and frame location relative to the distance covered returned by the odometry in order to independently determine the size and distance of the object from the Neato. The Neato may drive towards, away from, or parallel to the objects. Multiple objects may be present in one test run.

Resources:
- [Target Distance Estimation Using Monocular Vision](https://ieeexplore.ieee.org/document/6079296)
- [High Speed Obstacle Avoidance Using Monocular Vision and Reinforcement Learning](http://ai.stanford.edu/~asaxena/rccar/ICML_ObstacleAvoidance.pdf)
- [Learning Depth from Single Monocular Images](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/saxena-nips-05.pdf)
