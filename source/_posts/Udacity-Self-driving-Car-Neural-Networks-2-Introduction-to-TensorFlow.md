---
title: 'Udacity Self-driving Car - Neural Networks: 2. Introduction to TensorFlow'
date: 2019-04-09 10:46:18
tags:
	- TensorFlow
	- Self driving
	- Neural networks
	- Deep learning
	- Machine learning
categories:
	- Udacity Nanodegree Self-Driving Car
mathjax: true 
---
# "Hello, World" Code
<!-- more -->
	import tensorflow as tf
	# Create a tensorflow constant variable
	hello_constant = tf.constant('Hello World!')

	with tf.Session() as sess:
	# Run the tensorflow constant in the session
	output = sess.run(hello_constant)
	print(output)

## Note
The data in TensorFlow is not stored as integers, floats, doubles ...... or strings. These values are encapsulated in the object called "tensor". The tensor used here is a constant tensor, a tensor comes in a variety of size:  
	
	# 0-dimensional int32 tensor
	A = tf.constant(153)
	# 1-dimensional int32 tensor
	B = tf.constant([12,51,654])
	# 2-dimensional int32 tensor
	C = tf.constant([[21,32,1], [32,343,55]])

A TensorFlow Session is an environment for running a graph. The session is in charge of allocating the operations to GPU(s) and/or CPU(s), including remote machines. the `tf.Session()` creates a session instance and the `tf.Session().run()` function evaluates the tensor and returns the results.

# Non-constant variables
## TensorFlow Placeholder
A `tf.placeholder()` returns a tensor that gets its value from data passed to the `te.Session().run()` which allows you to set the input right before the session runs.  
## Session's feed_dict
The parameter `feed_dict` in `tf:Session().run()` feed the values into the Placeholder, for example:  
	
		x = tf.placeholder(tf.string)
		with tf.Session() as sess:
			output = sess.run(x, feed_dict={x: 'Hello World!'})

Note: the data passed to the feed_dict should match the tensor's type.  

# TensorFlow Math
## Addition, subtraction, multiplication and division in TensorFlow
	x = tf.constant(9)
	y = tf.constant(3)

	# Addition
	z = tf.add(x, y)

	# Subtraction
	k = tf.subtract(x, y)

	# multiplication
	l = tf.multiply(x, y)

	# division
	m = tf.divide(x, y)

## Matrices multiplication

	tf.matmul(a, b)

## Variable type Converting in TensorFlow
	tf.cast(tf.const(9.99), tf.int32)

## TensorFlow Linear function
The linear function $$y = Wx + b$$ in TensorFlow should be : $$y = xW + b$$, where the variables $$W$$ and $$b$$ should not be `tf.placeholder()` or `tf.constant()`, because these two type can not be modified.
### `tf.Variable()`
It creates a tensor with an initial value that can be modified. The tensor stores its state in the session, and the state of the tensor must be initialized manually.  
	
	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)
## `tf.truncated_normal()`
It returns a tensor with random values from a normal distribution whose magnitude is no more than 2 standard deviations from the mean.

	n_features = 100
	n_labels = 10
	weights = tf.Variable(tf.truncated_normal((n_features, n_labels)))

## `tf.zeros()`
It returns a tensor with all zeros.  
	
	bias = tf.Variable(tf.zeros(n_labels))

## TensorFlow Nonlinear functions
- ReLU: `tensorflow.nn.relu()`