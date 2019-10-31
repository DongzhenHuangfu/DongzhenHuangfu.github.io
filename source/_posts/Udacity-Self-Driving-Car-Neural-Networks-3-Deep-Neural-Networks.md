---
title: 'Udacity Self-Driving Car - Neural Networks: 3. Deep Neural Networks'
date: 2019-04-11 10:56:41
tags:
	- Deep learning
	- Neural networks
	- Machine learning
	- Self driving
	- Classifier
	- TensorFlow
categories:
	- Udacity Nanodegree Self-Driving Car
mathjax: true 
---
# Logistic Classifier
A linear classifier, it takes in the inputs and applies the linear function to them to generate the predictions (classifications).
<!-- more -->

# Training, validation and test set
- Training set: Data set for training.  
- Validation set: Data set for judging the model during the training process.  
- Test set: Data set for test the model after the training process.  

# Cross-Validation
See [here](https://towardsdatascience.com/cross-validation-70289113a072)

# Stochastic Gradient Descent (SGD)
Estimate with a random, small part of the training data (1-1000). Compute the lost and the derivative for this sample and update the parameters with this derivative as the direction for the gradient descent.
## Momentum
Keep a running average of the gradient and use that running average instead of the direction of the current batch of the data.  
- Better convergence.

## Hyper-parameters
- Initial learning rate  
- Learning rate decay  
- Momentum  
- Batch size  
- Weight initialization  

Reduce the learning rate can be helpful if the thing goes not well!  

## ADAGRAD
An optimized SGD, which chooses the initial learning rate, learning rate decay and momentum for you.  

# Mini-Batching
A technique for training on subsets of the dataset instead of all the data at one time. This provides the ability to train a model, even if a computer lacks the memory to store the entire dataset.  
## Combination with SGD
 Randomly shuffle the data at the start of each epoch, then create the mini-batches. For each mini-batch, you train the network weights with gradient descent. Since these batches are random, you're performing SGD with each batch.  

# Epochs
An epoch is a single forward and backward pass of the whole dataset. This is used to increase the accuracy of the model without requiring more data.  

# Example: Classify the letters in the MNIST database
	# import the data
	from tensorflow.examples.tutorials.mnist import input_data
	mnist = input_data.read_data_sets(".", one_hot=True, reshape=False)

	# define the learning parameters
	import tensorflow as tf

	learning_rate = 0.001
	training_epochs = 20
	batch_size = 128  # Decrease batch size if you don't have enough memory
	display_step = 1

	n_input = 784  # MNIST data input (img shape: 28*28)
	n_classes = 10  # MNIST total classes (0-9 digits)

	# define the size of the hidden layer
	n_hidden_layer = 256 # layer number of features

	# define the weights and biases
	weights = { 
		'hidden_layer': tf.Variable(tf.random_normal([n_input, n_hidden_layer])),
		'out': tf.Variable(tf.random_normal([n_hidden_layer, n_classes]))
	}
	biases = {
		'hidden_layer': tf.Variable(tf.random_normal([n_hidden_layer])),
    	'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # define the inputs and outputs
    x = tf.placeholder("float", [None, 28, 28, 1])
	y = tf.placeholder("float", [None, n_classes])
	x_flat = tf.reshape(x, [-1, n_input])

	# define the neural network
	# Hidden layer with RELU activation
	layer_1 = tf.add(tf.matmul(x_flat, weights['hidden_layer']), biases['hidden_layer'])
	layer_1 = tf.nn.relu(layer_1)
	# Output layer with linear activation
	logits = tf.add(tf.matmul(layer_1, weights['out']), biases['out'])

	# Define loss and optimizer
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

	# Initializing the variables
	init = tf.global_variables_initializer()

	# Launch the graph
	with tf.Session() as sess:
	    sess.run(init)
	    # Training cycle
	    for epoch in range(training_epochs):
	        total_batch = int(mnist.train.num_examples/batch_size)
	        # Loop over all batches
	        for i in range(total_batch):
	            batch_x, batch_y = mnist.train.next_batch(batch_size)
	            # Run optimization op (backprop) and cost op (to get loss value)
	            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

Note: The code `minst.train.next_batch()` returns a subset of the training data.

# Save and restore the TensorFlow Models
## Save Variables
	import tensorflow as tf

	# The file path to save the data
	save_file = './model.ckpt'

	# Two Tensor Variables: weights and bias
	weights = tf.Variable(tf.truncated_normal([2, 3]))
	bias = tf.Variable(tf.truncated_normal([3]))

	# Class used to save and/or restore Tensor Variables
	saver = tf.train.Saver()

	with tf.Session() as sess:
	    # Initialize all the Variables
	    sess.run(tf.global_variables_initializer())

	    # Show the values of weights and bias
	    print('Weights:')
	    print(sess.run(weights))
	    print('Bias:')
	    print(sess.run(bias))

	    # Save the model
	    saver.save(sess, save_file)
## Load Variables
	# Remove the previous weights and bias
	tf.reset_default_graph()

	# Two Variables: weights and bias
	weights = tf.Variable(tf.truncated_normal([2, 3]))
	bias = tf.Variable(tf.truncated_normal([3]))

	# Class used to save and/or restore Tensor Variables
	saver = tf.train.Saver()

	with tf.Session() as sess:
	    # Load the weights and bias
	    saver.restore(sess, save_file)

	    # Show the values of weights and bias
	    print('Weight:')
	    print(sess.run(weights))
	    print('Bias:')
	    print(sess.run(bias))
## Save Model
	# Remove previous Tensors and Operations
	tf.reset_default_graph()

	from tensorflow.examples.tutorials.mnist import input_data
	import numpy as np

	learning_rate = 0.001
	n_input = 784  # MNIST data input (img shape: 28*28)
	n_classes = 10  # MNIST total classes (0-9 digits)

	# Import MNIST data
	mnist = input_data.read_data_sets('.', one_hot=True)

	# Features and Labels
	features = tf.placeholder(tf.float32, [None, n_input])
	labels = tf.placeholder(tf.float32, [None, n_classes])

	# Weights & bias
	weights = tf.Variable(tf.random_normal([n_input, n_classes]))
	bias = tf.Variable(tf.random_normal([n_classes]))

	# Logits - xW + b
	logits = tf.add(tf.matmul(features, weights), bias)

	# Define loss and optimizer
	cost = tf.reduce_mean(\
	    tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)\
	    .minimize(cost)

	# Calculate accuracy
	correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	import math

	save_file = './train_model.ckpt'
	batch_size = 128
	n_epochs = 100

	saver = tf.train.Saver()

	# Launch the graph
	with tf.Session() as sess:
	    sess.run(tf.global_variables_initializer())

	    # Training cycle
	    for epoch in range(n_epochs):
	        total_batch = math.ceil(mnist.train.num_examples / batch_size)

	        # Loop over all batches
	        for i in range(total_batch):
	            batch_features, batch_labels = mnist.train.next_batch(batch_size)
	            sess.run(
	                optimizer,
	                feed_dict={features: batch_features, labels: batch_labels})

	        # Print status for every 10 epochs
	        if epoch % 10 == 0:
	            valid_accuracy = sess.run(
	                accuracy,
	                feed_dict={
	                    features: mnist.validation.images,
	                    labels: mnist.validation.labels})
	            print('Epoch {:<3} - Validation Accuracy: {}'.format(
	                epoch,
	                valid_accuracy))

	    # Save the model
	    saver.save(sess, save_file)
	    print('Trained Model Saved.')
## Load Model
	saver = tf.train.Saver()

	# Launch the graph
	with tf.Session() as sess:
	    saver.restore(sess, save_file)

	    test_accuracy = sess.run(
	        accuracy,
	        feed_dict={features: mnist.test.images, labels: mnist.test.labels})

	print('Test Accuracy: {}'.format(test_accuracy))

# Naming Error
Error description:  
InvalidArgumentError (see above for traceback): Assign requires shapes of both tensors to match.

Example:  
	import tensorflow as tf

	# Remove the previous weights and bias
	tf.reset_default_graph()

	save_file = 'model.ckpt'

	# Two Tensor Variables: weights and bias
	weights = tf.Variable(tf.truncated_normal([2, 3]))
	bias = tf.Variable(tf.truncated_normal([3]))

	saver = tf.train.Saver()

	# Print the name of Weights and Bias
	print('Save Weights: {}'.format(weights.name)) # Save Weights: Variable:0
	print('Save Bias: {}'.format(bias.name)) # Save Bias: Variable_1:0

	with tf.Session() as sess:
	    sess.run(tf.global_variables_initializer())
	    saver.save(sess, save_file)

	# Remove the previous weights and bias
	tf.reset_default_graph()

	# Two Variables: weights and bias
	bias = tf.Variable(tf.truncated_normal([3])) # Load Weights: Variable_1:0
	weights = tf.Variable(tf.truncated_normal([2, 3])) # Load Bias: Variable:0

	saver = tf.train.Saver()

	# Print the name of Weights and Bias
	print('Load Weights: {}'.format(weights.name))
	print('Load Bias: {}'.format(bias.name))

	with tf.Session() as sess:
	    # Load the weights and bias - ERROR
	    saver.restore(sess, save_file)

Reason: TensorFlow use a string identifier "name" for Tensors and Operations. If it is not given, TensorFlow will create one automatically.  
You can define the "name" manually:  
	
	import tensorflow as tf

	tf.reset_default_graph()

	save_file = 'model.ckpt'

	# Two Tensor Variables: weights and bias
	weights = tf.Variable(tf.truncated_normal([2, 3]), name='weights_0')
	bias = tf.Variable(tf.truncated_normal([3]), name='bias_0')

	saver = tf.train.Saver()

	# Print the name of Weights and Bias
	print('Save Weights: {}'.format(weights.name))
	print('Save Bias: {}'.format(bias.name))

	with tf.Session() as sess:
	    sess.run(tf.global_variables_initializer())
	    saver.save(sess, save_file)

	# Remove the previous weights and bias
	tf.reset_default_graph()

	# Two Variables: weights and bias
	bias = tf.Variable(tf.truncated_normal([3]), name='bias_0')
	weights = tf.Variable(tf.truncated_normal([2, 3]) ,name='weights_0')

	saver = tf.train.Saver()

	# Print the name of Weights and Bias
	print('Load Weights: {}'.format(weights.name))
	print('Load Bias: {}'.format(bias.name))

	with tf.Session() as sess:
	    # Load the weights and bias - No Error
	    saver.restore(sess, save_file)

	print('Loaded Weights and Bias successfully.')

# L2 Regularization
$$\ell_2$$ regularization add the $\ell_2$ norm of the weight multiplied with a little value to punish too large weights, as the equation \ref{L2} shows.

\begin{equation}
l' = l + \frac{1}{2}\beta\lVert w \rVert^2_2
\label{L2}
\end{equation}

Where $l'$ is the new loss, $l$ is the old loss and $w$ is the weight.  

# Dropout
The Dropout unit can be inserted in the network structure, it works like a layer. The layer takes in the input of the former layer, randomly sets some of them as zero and enlarges the other correspondingly. For example if the half of the input values are set as zero, the others will be scaled by a factor $2$. It makes the model "not so concentrate" on a specific feature and become more universal.  
When evaluating the model, the dropout rate should be set as $1$, which means no dropout will appear. It is because the Dropout layer only works on the training process.  
See [here](https://www.tensorflow.org/api_docs/python/tf/nn/dropout).  
## Code
`tf.nn.dropout()`  
## Example
	keep_prob = tf.placeholder(tf.float32) # probability to keep units

	hidden_layer = tf.add(tf.matmul(features, weights[0]), biases[0])
	hidden_layer = tf.nn.relu(hidden_layer)
	hidden_layer = tf.nn.dropout(hidden_layer, keep_prob)

	logits = tf.add(tf.matmul(hidden_layer, weights[1]), biases[1])
Where:  
1. hidden_layer: the tensor to which you would like to apply dropout.  
2. keep_prob: the probability of keeping (i.e. not dropping) any given unit. (recommend 0.5)  