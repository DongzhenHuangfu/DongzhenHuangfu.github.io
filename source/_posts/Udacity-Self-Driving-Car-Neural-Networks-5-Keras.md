---
title: 'Udacity Self-Driving Car - Neural Networks: 5. Keras'
date: 2019-04-13 21:06:54
tags:
	- Deep learning
	- Machine learning
	- Neural networks
	- Keras
	- Self driving
categories:
	- Udacity Nanodegree Self-Driving Car
---
[Keras](http://faroit.com/keras-docs/1.2.1/) make the coding for building a neural networks simpler.
<!-- more -->
# Sequential Model
The [Sequential model in Keras](https://keras.io/models/sequential/) is a wrapper for neural network model. It provides common functions like `fit()`, `evaluate()` and `compile()`.

# Layers
Use the model's `add()` to add a layer:  

    from keras.models import Sequential
    from keras.layers.core import Dense, Activation, Flatten

    # Create the Sequential model
    model = Sequential()

    #1st Layer - Add a flatten layer
    model.add(Flatten(input_shape=(32, 32, 3)))

    #2nd Layer - Add a fully connected layer
    model.add(Dense(100))

    #3rd Layer - Add a ReLU activation layer
    model.add(Activation('relu'))

    #4th Layer - Add a fully connected layer
    model.add(Dense(60))

    #5th Layer - Add a ReLU activation layer
    model.add(Activation('relu'))

Keras will automatically infer the shape of each layer, which means you only need to set the input dimensions for the first layer.

# Convolutional Neural Networks in Keras

	from keras.models import Sequential
	from keras.layers.convolutional import Conv2D

	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3)))

# Pooling in Keras

	from keras.layers.pooling import MaxPooling2D

	model.add(MaxPooling2D((2, 2)))

# Dropout in Keras

	from keras.layers.core import Dense, Activation, Flatten, Dropout

	model.add(Dropout(0.5))

# Compile, fit and evaluate the model

	model.compile('adam', 'categorical_crossentropy', ['accuracy'])
	history = model.fit(X, y, epochs=10, validation_split=0.2)
	metrics = model.evaluate(X_test, y_test)