---
title: 'Udacity Self-driving Car - Neural Networks: 1. Fundamentals'
date: 2019-04-08 11:02:29
tags:
	- Machine learning
	- Deep learning
	- Self driving
	- Neural networks
categories:
	- Udacity Nanodegree Self-Driving Car
mathjax: true 
---
# Main idea
Take in an input, process the information (for example: $z = w * x + b$), give an output(for example using activation functions).  
<!-- more -->
# Perceptron
An artificial neurons, the basic unit of a neural network, each takes in an input data and decides how to react to(categorize) that data.  
# Weights and bias
- The input of the perceptron will be multiplied by a weight value, this weight will be tuned later for a better result w.r.t. the output evaluation.  
- Higher weight means the network considers that input is more important.  
- The bias is a value which will be added to the multiplied result, also tunable.  

# Activation functions
Activation functions are functions which decide the output of the node.  
The most common activation functions are: Sigmoid , ReLu , Tanh  and softmax, where the first three activation functions follow the formulas below and the softmax function (see below) is used for the multi-class classifications and output the possibility of this result.  
Sigmoid: $f(z) = \frac{1}{1 + e^{-z}}$  
ReLU:    $f(z) = \max(0, z)$  
tanh:    $f(z) = \tanh(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}$  

# Softmax function
The softmax function is also called normalized exponential function, it is used for transforming the from the model calculated value into the probability distribution of the different classes.  
For example, a model needs to predict the color of the input and there are three options: red, green and blue. The output value from the model of this three options are: $2$, $9$ and $0$, now the model needs to transform the value into the probability for the further prediction, the corresponding distribution of the probability has the following rules:  
1. The probability needs to be positive.  
2. The summary of the probability should be equal to $1$.  
3. The class with higher value has the higher probability.  

To achieve this, the transformation can be realized by the equation:  
$$P(A_i) = \frac{e^{Z_i}}{\sum^{n}_{j=1} e^{Z_j}}$$  

Where $P(A_i)$ is the probability for classifying this input to the Class $A_i$, $Z_i$ is the output value from the model of the $i$th class, $n$ is the number of the classes.  
For the prediction, the model classifies the input to the class with the maximum probability.

# Perception Algorithm
## Description
Find a line $$w * x + b * y + c = 0$$ that separates the samples.

## Algorithm
- Start with random weights and bias: a, b, c.  
- For every misclassified points:  
	- If the point is under the line: $$a += \alpha * x_i%, $b += \alpha * y_i$, $c += \alpha$$.  
	- If the point is above the line: $$a -= \alpha * x_i%, $b -= \alpha * y_i$, $c -= \alpha$$.  

Note: The $\alpha$ here should be defined by the user.  

# Error Functions
## Conditions to be met
1. differentiable.  
2. continuous. 

# Cross Entropy
## Function
$$\sum^{n}_{j=1} -ln(P_j)$$

Note: Smaller cross entropy means higher probability.  

# Logistic Regression
## Mainly steps:
1. Take the data.  
2. Pick a random model.  
3. Calculate the error. 
4. Minimize the error, and obtain a better model.  

# Gradient Descent Algorithm
## Main idea:
Take the negative of the gradient of the error function as the moving direction, use the learning rate to avoid dramatic changes.
$$w_{i}' -= \alpha * \frac{\partial E}{\partial w_i} $$  
Where $w_i$ means the i-th weight, and $E$ means the error.  

# Neural Network
## Main idea
The linear combination of the output from the last models (cells).  
## Feedforward
Take input, get the output of each layers.
## Mathematical Knowledges:

[Introduction to vectors](https://www.khanacademy.org/math/linear-algebra/vectors-and-spaces/vectors/v/vector-introduction-linear-algebra)  
[Introduction to matrices](https://www.khanacademy.org/math/precalculus/precalc-matrices) 
[Chain rules](https://www.khanacademy.org/math/ap-calculus-ab/ab-differentiation-2-new/ab-3-1a/v/chain-rule-introduction)  

# Backpropagation
## Main idea
Update the parameters in each layers with regard to the error function.  
Gradient of the error function:
$$\nabla E = (\frac{\partial E}{\partial w_{11}}, ... , \frac{\partial E}{\partial w_{mn}}, \frac{\partial E}{\partial b_1}, ... , \frac{\partial E}{\partial b_m})$$  
## Main steps
1. Doing a feedforward operation.  
2. Comparing the output of the model with the desired output and calculating the error.  
4. Running the feedforward operation backwards (backpropagation) to spread the error to each of the weights.  
5. Use this to update the weights, and get a better model.  
6. Continue this until we have a model that is good.  

# Further Reading
[Yes, you should understand backprop](https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b#.vt3ax2kg9)  
[a lecture from Stanford's CS231n course](https://www.youtube.com/watch?v=59Hbtz7XgjM)  
