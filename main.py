import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.keras import optimizers

"""
This file is a simple implementation of a ANN using tensorflow
By Elias Taylor
"""

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

"""
# Example of image 
plt.imshow(X_train[0], cmap="gray")
print(y_train[0])
plt.show()
"""

"""There are multiple reasons to normalize data, here it is done mostly because it is good practice to normalize data 
before training. But, in the MNIST dataset, the pixel values of the images range from 0 to 255. This means that some 
features, such as the brightness of the pixels, will have a much larger impact on the model than other features. 
Normalizing addresses this problem by scaling the features from 0 to 1. """
# Normalize the data
X_train = X_train / 255.0
X_test = X_test / 255.0

"""
Explanation: 
Sequential groups a linear stack of layers into a tf.keras.Model.

The flatten function will then flatten the tensor, which means that it will convert it into a tensor with a single 
dimension. This is good because most models require the input to be a single dimension. 

The dense layer in TensorFlow is a fully connected layer. 
This means that each neuron in the layer is connected to every neuron in the previous layer. 
The dense layer takes two arguments:
  - units: This is the number of neurons in the layer.
  - activation: This is the activation function that will be used for the layer.
We are using RELU as our activation as it is commonly used in image classification
"""
# Create a model.
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

"""
The model.compile method is used to configure the model for training. 
It takes three arguments:
- optimizer: This is the algorithm that will be used to update the weights of the network during training.
- loss: This is a measure of how well the model is performing on the training data.
- metrics: This is a set of metrics that will be used to evaluate the model's performance on the training and test data.

We are using Stochastic Gradient Descent (SGD) as the optimizer in conjunction with sparse categorical Crossentropy(SCC)
for our loss. SDG uses the gradient to update the weights of the model in the direction of the steepest descent. This 
means that the model is updated in the direction that will most quickly reduce the loss function. Additionally, 
SDG also uses mini-batches to reduce computation time. 

Note: when using keras some things are done automatically, which makes things easier but also less custom. For 
example, the learning_rate=0.01 for SDG, we can change this but that takes extra steps and is out of the scope of 
this project. """
# Compile the model
model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'],optimizer=tf.keras.optimizers.SGD(learning_rate=0.1))

"""
The model.fit() method in TensorFlow is used to train a model. It takes four arguments:
 - x_train: This is the training data.
 - y_train: This is the labels for the training data.
 - epochs: This is the number of times that the model will be trained on the training data.
 - batch_size: This is the number of data points that will be used in each training step.
If you do not specify a batch size when calling the model.fit() method, the function will use a batch size of 1. 
This means that the model will be trained on one data point at a time. This is called Batch Gradient Descent (BGD)
or vanilla gradient descent.
"""
# train model using
model.fit(X_train, y_train, epochs=10, batch_size=40)

"""
The model.evaluate() method in TensorFlow is used to evaluate the performance of a model. It takes two arguments:

X_test: This is the test data.
y_test: This is the labels for the test data.
"""
loss, accuracy = model.evaluate(X_test, y_test)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("Accuracy:", accuracy)
    print("Loss:", loss)

