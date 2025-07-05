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

## 1. What does this mean?

- Each **layer** in the network performs a linear transformation on its inputs by:
  - Computing a **weighted sum** of the inputs plus a bias. 
  - Applying a **nonlinear activation function** (e.g. ReLU, Sigmoid, Tanh etc).

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

## 2. Key mathematical pieces to understand

### Forward Pass
# Formula

![Formula](https://wikimedia.org/api/rest_v1/media/math/render/svg/4f396cadb80fdca91d513da05d3c8c217d597a46)


This is the forward pass for one layer of a network. Let's take the input layer to first hidden layer as an example.  

This function will multiply the weight matrix with shape (Input Layer Size, hidden layer size), where each row corresponds to the weights connected to one neuron. This matrix is mulitplied by the inputs of each neuron from the first row, and then a final bias is added at the end from the hidden layers neurons. This is then wrapped in a function that allows the network to handle non-linear data.  

This is synonymous to the idea of chefs in a restaurant. Each neuron is a chef responsible for one element of the dish (salt in this case). It receives elements from previous chefs, during which the elements are cooked, or steamed, or caramelized (this is the weights and their contribution). The chef then adds salt (his bias) and passes the food onwards till it reaches the customer (output layer)

This continues until we reach the output layer.

### Loss Function
At the end of the network, to check wether the network has done well or not comparitvely to what the output should've been, we use a loss function such as MSE, CCE, or RMSE. This project used the Categorical Cross-Entropy formula or CCE.

### Categorical Cross-Entropy

Categorical Cross-Entropy is a loss function used for multi-class classification. It measures how different the predicted probabilities are from the true labels. This is especially useful for a MNIST data identifier like this one, where we have more than a binary output from the neural network.

The formula for one sample is:

![Forward and Backward Propagation Diagram](https://miro.medium.com/v2/resize:fit:774/0*vteMfTAGWsIZSaOW)

Here we find the sum of the products of the natural logarithm of the predicted outcomes multiplied by what the outcome should've been. 

This loss penalizes incorrect predictions more heavily when the model is confident but wrong. This assists with the vanishing gradient problem (more on that later) and ensures that our network doesn't plateau.

### Gradient Descent

![Gradient Descent Illustration](https://i0.wp.com/aicorr.com/wp-content/uploads/2024/03/What-is-Gradient-Descent-in-Machine-Learning_.jpg?fit=1024%2C576&ssl=1)

Neural Networks use the idea of gradient descent. Since the Loss of a function is to be minimised, we essentially plot the Loss function and find a local minima. This is done by predicting wether to move right or left (indicating how the value of the weight or bias should change) and slowly making your way to a minima.  

This is synonymous to numerical methods to find minima or roots such as the famous Newton Raphson. This is useful as the Loss function itself is a heavily multidimensional function and can't be plotted in a simple 2d or 3d space.


### Activation and Softmax

The Rectified Linear Unit (ReLU) is defined as:

![Backpropagation Gradients](https://miro.medium.com/v2/resize:fit:732/1*LVV3mkrBnwdpcbelaePIqg.png)

This will be delve into later

### Softmax Activation Function

The Softmax function converts a vector of values into probabilities:

![Backpropagation Pipeline](https://cdn.prod.website-files.com/60d1a7f6aeb33c5c595468b4/64f1b88c32bc6c0287ec6d27_4O_7Wfj5dV04MZdm0fUrGnxv1rGmHVl6TcXOn1qJyUA6blA1L8aimwdA3Fup5hTdm0luPtzlD-HW4StkqDANERDYcIXqbo3yZ01gd4AnzLk3KSBvgjLZwuV094i5eX9aAgdaW05PPZsQgt02Gi55iyc.png)

This function outputs a probability distribtuion from an array of numbers. It is useful to interpret the output of the network at the last layer.

# How all of this pieces together

## 1. Initialising the network

To do this we will make a class with an input size, hidden layer schema, and output size. The class contains a function called RandomiseNetwork which adds random values to all weights and bias of the class. We do this layer by layer to ensure that going through the data is easy.  

Along with this I made the weights, bias', activations, and z_values accessible throughout the class.

```python
import numpy as np

class Neural_Network:
    def __init__(self, InputSize=784, HiddenLayers=[512, 512], OutputSize=10):
        self.InputSize=InputSize
        self.HiddenLayers=HiddenLayers
        self.OutputSize=OutputSize
        self.weights=[]
        self.bias=[]
        self.values=[]
        self.z_values=[]

    def RandomiseNetwork(self):
        #generating weights for input to hidden layers
        self.weights.append(np.random.randn(self.InputSize, self.HiddenLayers[0]) * np.sqrt(1 / self.InputSize))
        self.bias.append(np.zeros((1, self.HiddenLayers[0])))

        #generating weights for all hidden layers
        for x in range(len(self.HiddenLayers) - 1):
            self.weights.append(np.random.randn(self.HiddenLayers[x], self.HiddenLayers[x+1]) * np.sqrt(1 / self.HiddenLayers[x]))
            self.bias.append(np.zeros((1, self.HiddenLayers[x+1])))

        #generating weights for hidden layers to output
        self.weights.append(np.random.randn(self.HiddenLayers[-1], self.OutputSize) * np.sqrt(1 / self.HiddenLayers[-1]))
        self.bias.append(np.zeros((1, self.OutputSize)))
```

## 2. The Forward Pass

This will populate the activations of the network and ensure that we get an output.

```python
    #ReLU Activation Function 
    def ReLU(self, x):
        return np.maximum(0, x)
    
    #Softmax Activation function applied at output
    def SoftMaxActivation(self, x):
        x = x - np.max(x, axis=1, keepdims=True)
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    #Produces all activations for the randomised network, and then adjusted network
    def ForwardPass(self, inputs):
        self.values = [inputs]
        self.z_values = []

        for x in range(len(self.weights)):
            z=np.dot(self.values[-1], self.weights[x]) + self.bias[x] # linear algebra can recursively perform all necessary calculations
            self.z_values.append(z)

            # at the end you don't want to apply ReLU, and instead apply the softmax activation function
            if x == len(self.weights) - 1:
                a = self.SoftMaxActivation(z)
            else:
                a = self.ReLU(z)
            
            self.values.append(a)

        self.values=self.values
        return self.values[-1]
```

The ReLU activation functions looks like this.

![ReLU Activation Function](https://media.geeksforgeeks.org/wp-content/uploads/20250129162127770664/Relu-activation-function.png)
*Source: [GeeksforGeeks](https://www.geeksforgeeks.org/activation-functions-neural-networks/)*

This function allows the network to learn from non-linear data and can sometimes make certain weights and bias completely dormant if not used. Imagine if parts of your brain's neural pathways never fired no matter what stimuli is experienced. 

The Softmax activation functions looks like this and works as so. Arrays of numbers are fed into the function and an array of probabilities to choose from is output. 

![Neural Network Architecture](https://images.contentstack.io/v3/assets/bltac01ee6daa3a1e14/blte5e1674e3883fab3/65ef8ba4039fdd4df8335b7c/img_blog_image1_inline_(2).png)  
*Source: [Contentstack Blog](https://www.contentstack.com/blog/ai-deep-learning-and-machine-learning)*  

  
```python
 z=np.dot(self.values[-1], self.weights[x]) + self.bias[x]
```

This is responsible for all activation values. It is a simple linear algebraic expression and produces a **z_value**. This value is to be passed through a ReLU activation function for every neuron, and is to be passed through **only** softmax activation at the end. No ReLU is used at the final layer of neurons.

## 3. The Backward Pass (Backpropagation)

<table>
  <tr>
    <td align="center">
      <a href="https://www.youtube.com/watch?v=Ilg3gGewQ5U">
        <img src="https://img.youtube.com/vi/Ilg3gGewQ5U/0.jpg" width="300"/>
        <br/>
        <strong>Backpropagation Calculus – 3Blue1Brown</strong>
      </a>
    </td>
    <td align="center">
      <a href="https://www.youtube.com/watch?v=tIeHLnjs5U8">
        <img src="https://img.youtube.com/vi/tIeHLnjs5U8/0.jpg" width="300"/>
        <br/>
        <strong>What is a Neural Network? – 3Blue1Brown</strong>
      </a>
    </td>
  </tr>
</table>

The following were very useful to understand what backpropagation is essentially at its core and will help explain the following code. However in practice, I used Matt Mazur's tutorial online: 
[Matt Mazur’s step-by-step backpropagation example](https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/). to implement it into my program.

```python
    #Adjusts the network as required, the bases of the neural network is backpropagation
    def BackwardsPropagation(self, y, output):

        change_weights = []
        change_bias = []

        delta = output - y #Using Matt Mazur's https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/ formulas
        for x in reversed(range(len(self.weights))):

            previous_activations = self.values[x].reshape(1, -1)

            # linear algebra again to calculate necessary delta values of weights
            # and bias that affect the cost at the end
            del_weights = np.dot(previous_activations.T, delta)
            del_bias = np.sum(delta, axis=0, keepdims=True)

            # insert instead of append because insert puts it in the front of the list
            # we are looping backwards through the network and this adjusts for that
            change_weights.insert(0, del_weights)
            change_bias.insert(0, del_bias)

            # if we're still looping through the network, set
            # new values of delta based on what layer we're on
            if x > 0:
                previous_z_values = self.z_values[x - 1]
                delta = np.dot(delta, self.weights[x].T) * self.ReLUDer(previous_z_values)

        return change_weights, change_bias
```

Everything is explained in the comments.

## 4. Things to Note

Backpropagation with other activation functions such as Sigmoid or Tanh didn't give as good performance. 
Ensure that all numpy arrays are of fit size and don't cause an algebraic error
Backpropagation is best suited with batch updates, instead of singular updates

# Testing and Using the Network

This code is easily implemented for any other sort of neural network. But to use it with the MNIST Dataset we first have to extract the MNIST images. In the directory you will find the necessary zip files.

Note that for the minst data set, each image is of 28 by 28 pixels, so there are 784 pixels. This is why our input is 784. We supply each intensity of the pixel as our input. For different networks, you will need to arrange a different input architecture.

```python
gerald = Neural_Network()
gerald.RandomiseNetwork()

# loading the MNIST_Train csv every time you run the code is a bit 
# tedious and requires a lot of processing time.
# instead we'll save the file as a .npy and access it within milliseconds 
# with numpy's fast loading functions

# run the follolwing code once
# data = np.loadtxt('./train.csv', delimiter=',', skiprows=1, dtype=np.uint8)
# np.save('mnist_train.npy', data)

# use this after theres a mnist_train.npy in your files
data = np.load('./mnist_train.npy')

# seperate the label and image data
labels = data[:, 0]
images = data[:, 1:]
```

This will create the neural network, populate with random weights, and then not only load the mnist dataset but then save it as a .npy  
I found this faster and more convenient whilst training and experimenting

## 1. Training
We need to first initialise our training variables.
```python 
# training variables
batch_size = 16
batches = 2000

# randomise the images from the data set using permutations
perm = np.random.permutation(len(images))
images = images[perm]
labels = labels[perm]

# for the graph of loss over time
losses = []
```

This shuffles the images, creates the batch size and batches total, as well as a losses array to compute average gradients.

```python
for indent in range(batches):

    weight_grads = [np.zeros_like(w) for w in gerald.weights]
    bias_grads = [np.zeros_like(b) for b in gerald.bias]

    for i in range(batch_size):
        idx = indent * batch_size + i
        input_values = images[idx] / 255.0
        y = np.eye(10)[labels[idx]].reshape(1, 10)
        
        output = gerald.ForwardPass(input_values)

        grad_w, grad_b = gerald.BackwardsPropagation(y, output)

        for l in range(len(gerald.weights)):
            weight_grads[l] += grad_w[l]
            bias_grads[l] += grad_b[l]
```

We then loop through each batch and calculate the gradients for each weight and bias and append them to our array

```python 
    # find the mean average of all the gradients calculated
    # so that we can add them at the end of the batch
    for l in range(len(gerald.weights)):
        weight_grads[l] /= batch_size
        bias_grads[l] /= batch_size

    # update weights and bias from the gradients calculated
    gerald.UpdateNetwork(weight_grads, bias_grads, 0.1)
```

At the end of 32 images we find the mean average and update the network

```python
loss = gerald.DetermineCategoricalCrossEntropy(gerald.ForwardPass(images[batch_size * indent] / 255.0), labels[batch_size * indent])
losses.append(loss)
plt.plot(losses)
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.title("Loss over Time")
plt.show()
```

Compute the losses and plot a graph over time to analyse how the network learns.

# 2. Testing

Similarly we can test the network and produce an overall accuracy over n amounts of images

```python
# this section tests the accuracy of the network
counter = 0 
test_size = 1000

perm = np.random.permutation(len(images))
images = images[perm]
labels = labels[perm]

for i in range(test_size):
    input_values = images[i] / 255.0 
    y = np.eye(10)[labels[i]].reshape(1, 10)
    output = gerald.ForwardPass(input_values)

    if np.argmax(y) == gerald.IdentifyNum(output):
        counter += 1
    
    #
    # to see which images the network got wrong, uncomment the following:
    #
    # else:
    #     plt.figure(figsize=(2, 2))
    #     plt.imshow(input_values.reshape(28, 28), cmap='gray', interpolation='none')
    #     plt.axis('off')
    #     plt.show()
    #     gerald.ShowProbabiiltyPlot(output)

print(f'Net Accuracy: {(counter/test_size) * 100:.2f}%')
```

I got an accuracy close to 96.67% with two hidden layers of 512.

# Final Statements

This project was loads of fun and hopefully this will encourage you to explore other ideas such as LLM's, Decision Trees, etc.
