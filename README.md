# ✍️ Handwritten Digit Classification using CNN (MNIST Dataset)

This project implements a Convolutional Neural Network (CNN) using TensorFlow/Keras to classify handwritten digits from the MNIST dataset with high accuracy.

---

## 📁 Files Included

- `CNN_MNIST.ipynb`: Jupyter Notebook containing all code and plots.
- `README.md`: This documentation file.

---

## 📊 Project Overview

The MNIST dataset is a benchmark dataset consisting of 70,000 grayscale images of handwritten digits (0–9), each 28x28 pixels in size. This project uses a CNN to achieve high-performance classification of these digits.

---

## 🧪 Step-by-Step Process and Code Summary

### 1. **Import Libraries**
- Essential libraries: `tensorflow`, `keras`, `numpy`, `matplotlib`.

```python
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
```

### 2. **Load and Preprocess the Data**
- Load the MNIST dataset using Keras.

- Normalize image pixel values (scale between 0 and 1).

- Reshape the data to include a channel dimension.

```python
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]
```

### 3. **Build the CNN Model**
- Use 2 convolutional layers followed by pooling, flattening, and dense output.

```python
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
```

### 4. **Compile and Train the Model**
- Optimizer: Adam

- Loss: Sparse Categorical Crossentropy

- Metric: Accuracy

```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=5, 
                    validation_data=(x_test, y_test))
```

### 5. **Evaluate and Visualize Results**
- Evaluate accuracy on the test set.

- Plot training vs validation accuracy and loss.

```python
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'Test accuracy: {test_acc}')
```

## 📈 Model Performance
- Training Accuracy: ~99%

- Test Accuracy: ~99%

- Loss & accuracy trends are plotted for performance insights.
