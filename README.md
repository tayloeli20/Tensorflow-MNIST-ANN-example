# Simple Implementation of Artificial Neural Network (ANN) for Handwritten Digit Classification using TensorFlow

## 1. Data Loading
The code uses `tf.keras.datasets.mnist.load_data()` to load the MNIST dataset, which contains handwritten digits. It splits the dataset into training and test sets, denoted as `X_train`, `y_train`, `X_test`, and `y_test`.

## 2. Data Normalization
Since the pixel values of the MNIST images range from 0 to 255, the code performs data normalization. Each pixel value is divided by 255.0 to scale the features between 0 and 1. This ensures that all features have a similar impact on the model during training.

## 3. Model Architecture
The ANN model is defined using `tf.keras.Sequential`. It consists of three layers:
- Input layer: The `Flatten` layer reshapes the 28x28 images into a 1D array of size 784.
- Hidden layer: The `Dense` layer with 10 neurons and a ReLU activation function. ReLU (Rectified Linear Unit) is commonly used in image classification tasks.
- Output layer: The final `Dense` layer with 10 neurons and a softmax activation function. The softmax activation is used for multi-class classification, providing the probabilities for each class.

## 4. Model Compilation
The model is compiled using `model.compile()`. The following configurations are used:
- Loss: Sparse categorical cross-entropy (`sparse_categorical_crossentropy`) is employed since the dataset has integer labels (0 to 9) for multi-class classification.
- Optimizer: Stochastic Gradient Descent (SGD) with a learning rate of 0.1 is used to update the model weights during training.

## 5. Model Training
The model is trained using `model.fit()`, which iterates over the training data for a specified number of epochs (10 in this case). A batch size of 40 is used for each training step.

## 6. Model Evaluation
After training, the model's performance is evaluated on the test set using `model.evaluate()`. The accuracy and loss metrics are printed to assess the model's classification performance.

Overall, this code provides a simple yet effective implementation of an ANN for handwritten digit classification using TensorFlow, demonstrating the fundamental steps involved in building and training a neural network.
