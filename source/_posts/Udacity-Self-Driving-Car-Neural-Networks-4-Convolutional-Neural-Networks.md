---
title: 'Udacity Self-Driving Car - Neural Networks: 4. Convolutional Neural Networks'
date: 2019-04-13 10:53:09
tags:
	- Convolutional neural networks
	- Deep learning
	- Machine learning
	- Self driving
	- Neural networks
	- TensorFlow
categories:
	- Udacity Nanodegree Self-Driving Car
mathjax: true 
---
# Main idea
A neural networks that share their parameters across space. Aims to pick the properties regardless of the position.  
<!-- more -->

# Padding
## Calculate the size of the output
Given:  
- Input layer has a width of W and a height of H  
- Convolutional layer has a filter size F  
- Stride of S  
- Numbers of filters K  

Width of the next layer: $W_{out} = \frac{W - F + 2P}{S} + 1$   
Height of the next layer: $H_{out} = \frac{H - F + 2P}{S} + 1$  
Depth of the next layer: $D_{out} = K$ 

### Valid padding
Don't go pass the boundary.  
$out_{height} = ceil(\frac{float(in_{height} - filter_{height} + 1)}{float(strides[1]})$  
$out_{width} = ceil(\frac{float(in_{width} - filter_{width} + 1)}{float(strides[2])})$
### Same padding
Go off the Edge and pad with zeros in such a way that the output size is exactly the same size as the input map. (in case stride is 1)  
$out_{height} = ceil(\frac{in_{height}}{float(strides[1])})$  
$out_{width} = ceil(\frac{in_{width}}{float(strides[2])})$  

# Filter Depth
Different filters pick up different qualities and will be connected to the different neurons in the next layer. The depth $k$ means the number of the filters.  

# Visualizing CNNs
See the [relevant paper](https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf) and the [video](https://www.youtube.com/watch?v=ghEmQSxT6tw) from Zeiler and Fergus.  

## Code
	# image_input: the test image being fed into the network to produce the feature maps
	# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
	# Note: that to get access to tf_activation, the session should be interactive which can be achieved with the following commands.
	# sess = tf.InteractiveSession()
	# sess.as_default()

	# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and    max values of the output
	# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

	def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
	    # Here make sure to preprocess your image_input in a way your network expects
	    # with size, normalization, ect if needed
	    # image_input =
	    # Note: x should be the same name as your network's tensorflow data placeholder variable
	    # If you get an error tf_activation is not defined it maybe having trouble accessing the variable from inside a function
	    activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
	    featuremaps = activation.shape[3]
	    plt.figure(plt_num, figsize=(15,15))
	    for featuremap in range(featuremaps):
	        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
	        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
	        if activation_min != -1 & activation_max != -1:
	            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
	        elif activation_max != -1:
	            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
	        elif activation_min !=-1:
	            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
	        else:
	            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")

# Implement a CNN in TensorFlow
	# Output depth
	k_output = 64

	# Image Properties
	image_width = 10
	image_height = 10
	color_channels = 3

	# Convolution filter
	filter_size_width = 5
	filter_size_height = 5

	# Input/Image
	input = tf.placeholder(
	    tf.float32,
	    shape=[None, image_height, image_width, color_channels])

	# Weight and bias
	weight = tf.Variable(tf.truncated_normal(
	    [filter_size_height, filter_size_width, color_channels, k_output]))
	bias = tf.Variable(tf.zeros(k_output))

	# Apply Convolution
	conv_layer = tf.nn.conv2d(input, weight, strides=[1, 2, 2, 1], padding='SAME')
	# Add bias
	conv_layer = tf.nn.bias_add(conv_layer, bias)
	# Apply activation function
	conv_layer = tf.nn.relu(conv_layer)

Note: 
- "stride" in `tf.nn.conv2d`: [batch, input_height, input_width, input_channels], the "batch" and "input_channels" here usually be set to be 1.  
- `tf.nn.bias_add()` adds a 1-d bias to the last dimension in a matrix.  

# Pooling/Max-pooling method
The pooling method is a sample-based discretization process which is well-known as a method for resolving over-fitting problems by abstracting the information from the input and reduce the calculation cost by reducing the number of the parameters for learning, as the Figure shows:

<img width="90%" src="https://github.com/DongzhenHuangfu/pictures-for-blog/raw/master/Udacity_self_driving_car_neural_networks/3_1_pooling.png"/>

The pooling unit has an user defined size and will slide on the input data with an user defined step size (stride). Each time the unit moves, it will concentrate the information it takes in, for example, the max-pooling method will output the maximum value of the input. The Figure shows the basic idea of a max-pooling method with size $2 \times 2$ and stride $2$:

<img width="90%" src="https://github.com/DongzhenHuangfu/pictures-for-blog/raw/master/Udacity_self_driving_car_neural_networks/3_2_maxpooling.png"/>  

# $1 \times 1$ Convolutions
Very inexpensive way to make the model deeper and have more parameters.  

# Inception modules
See the video [here](https://www.youtube.com/watch?v=VxhSouuSZDY).  
<img width="90%" src="https://i.ytimg.com/vi/VxhSouuSZDY/maxresdefault.jpg"/>  

# LeNet
See the paper [here](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf).  
<img width="90%" src="https://www.researchgate.net/profile/Yiren_Zhou/publication/312170477/figure/fig1/AS:448817725218816@1484017892071/Structure-of-LeNet-5.png"/>  

# Additional Resources
- Andrej Karpathy's [CS231n Stanford course](http://cs231n.github.io/) on Convolutional Neural Networks.  
- Michael Nielsen's [free book](http://neuralnetworksanddeeplearning.com/) on Deep Learning.  
- Goodfellow, Bengio, and Courville's more advanced [free book](http://deeplearningbook.org/) on Deep Learning.  