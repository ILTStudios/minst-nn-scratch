# An MNIST Neural Network From Scratch

The Initial Inspiration of this project came from an Instagram post from _Green Code YT_. Nowadays with the help of libraries such as PyTorch or Tensorflow's kesar, making a neural network is as easy as about 10 lines of code and some data.
<p align="center">
  <a href="https://www.youtube.com/watch?v=cAkMcPfY_Ns" target="_blank">
    <img src="https://img.youtube.com/vi/cAkMcPfY_Ns/hqdefault.jpg" alt="Awesome Demo" />
  </a>
</p>

This video inspired me to take on the challenge to not only figure out the mathematics of a neural network, but actually build one myself from scratch with just numpy. My goal was a neural network that worked, and had a decently fair accuracy. 

# Neural Network as a Multidimensional Equation

A neural network can be understood as a **complex, multidimensional mathematical function** that maps input data to output predictions. This was my first realisation while I explored the world of Neural Networks and how they work.

## What does this mean?

- Each **layer** in the network performs a linear transformation on its inputs by:
  - Computing a **weighted sum** of the inputs plus a bias. 
  - Applying a **nonlinear activation function** (e.g., ReLU, Sigmoid).

- These transformations can be expressed mathematically as:

![Layer Output Equation](https://latex.codecogs.com/png.latex?y%20%3D%20f%28W%20%5Ccdot%20x%20%2B%20b%29)

  where:  
  - x is the input vector,  
  - W is the weight matrix,  
  - b is the bias vector,  
  - f is the activation function,  
  - y is the output vector of the layer.

- By stacking multiple layers, the network composes these functions, resulting in a highly flexible composite function:

![Multi-layer Equation](https://latex.codecogs.com/png.latex?y%20%3D%20f%5E%7B%28n%29%7D%28W%5E%7B%28n%29%7D%20%5Ccdot%20f%5E%7B%28n-1%29%7D%28W%5E%7B%28n-1%29%7D%20%5Ccdots%20f%5E%7B%281%29%7D%28W%5E%7B%281%29%7D%20%5Ccdot%20x%20%2B%20b%5E%7B%281%29%7D%29%20%2B%20%5Ccdots%20%2B%20b%5E%7B%28n-1%29%7D%29%20%2B%20b%5E%7B%28n%29%7D%29)


- Because inputs and outputs are vectors in multidimensional space and weights are matrices, this composite function is effectively a **multidimensional nonlinear equation**. This can stack up to over a 13,002 dimensional space for an MNIST neural network.

## Key mathematical pieces to understand

### Loss Function

## Categorical Cross-Entropy

## Categorical Cross-Entropy

Categorical Cross-Entropy is a loss function used for multi-class classification. It measures how different the predicted probabilities are from the true labels.

The formula for one sample is:

![Categorical Cross-Entropy Equation](https://latex.codecogs.com/png.latex?CCE%20%3D%20-%5Csum_%7Bi%3D1%7D%5E%7BC%7D%20y_i%20%5Clog%28%5Chat%7By%7D_i%29)

Where:

- \(C\) = number of classes  
- \(y_i\) = 1 if class i is the correct class, otherwise 0  
- \(\hat{y}_i\) = predicted probability for class i  

This loss penalizes incorrect predictions more heavily when the model is confident but wrong.

### Gradient Descent

![Gradient Descent Illustration](https://i0.wp.com/aicorr.com/wp-content/uploads/2024/03/What-is-Gradient-Descent-in-Machine-Learning_.jpg?fit=1024%2C576&ssl=1)



### Activation and Softmax

The Rectified Linear Unit (ReLU) is defined as:

![ReLU Equation](https://latex.codecogs.com/png.latex?%5Ctext%7BReLU%7D%28x%29%20%3D%20%5Cmax%280%2C%20x%29)

### Softmax Activation Function

The Softmax function converts a vector of values into probabilities:

![Softmax Equation](https://latex.codecogs.com/png.latex?%5Csigma%28%5Cmathbf%7Bz%7D%29_i%20%3D%20%5Cfrac%7Be%5E%7Bz_i%7D%7D%7B%5Csum_%7Bj%7D%20e%5E%7Bz_j%7D%7D)
