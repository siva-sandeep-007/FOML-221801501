import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

# 1. Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. Normalize the data
x_train, x_test = x_train / 255.0, x_test / 255.0

# 3. Build the neural network model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # Flatten the 28x28 images
    layers.Dense(128, activation='relu'),  # Hidden layer with ReLU activation
    layers.Dense(10, activation='softmax')  # Output layer with softmax for classification
])

# 4. Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. Train the model and capture the history
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# 6. Evaluate the model on the test dataset
test_loss, test_acc = model.evaluate(x_test, y_test)

# Print out the results
print(f"Test accuracy: {test_acc:.4f}")
print(f"Test loss: {test_loss:.4f}")

# 7. Plotting training history
# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
