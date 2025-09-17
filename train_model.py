import tensorflow as tf
import numpy as np

# --- 1. Load and Preprocess Data ---
print("Loading CIFAR-10 dataset...")
# Load the dataset from Keras
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One-hot encode the labels (e.g., convert label '3' to [0,0,0,1,0,0,0,0,0,0])
# This is necessary for the 'categorical_crossentropy' loss function
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

print("Data loaded and preprocessed successfully.")
print(f"Training data shape: {x_train.shape}")
print(f"Test data shape: {x_test.shape}")


# --- 2. Build the CNN Model ---
print("Building the CNN model...")
model = tf.keras.models.Sequential([
    # Input layer specifies the shape of our images (32x32 pixels, 3 color channels)
    tf.keras.layers.Input(shape=(32, 32, 3)),
    
    # Convolutional Block 1
    # 32 filters, each 3x3 in size. 'relu' is a common activation function.
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    # Max pooling reduces the spatial dimensions (height, width) of the output volume.
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    # Convolutional Block 2
    # Using more filters to learn more complex patterns
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    # Flattening the 3D output to 1D to feed into the dense layers
    tf.keras.layers.Flatten(),
    
    # Dense (fully connected) layer with 64 neurons
    tf.keras.layers.Dense(64, activation='relu'),
    
    # Output layer with 10 neurons (one for each class)
    # 'softmax' activation converts the output to a probability distribution across the classes.
    tf.keras.layers.Dense(10, activation='softmax')
])

# Get a summary of the model architecture
model.summary()


[Image of a Convolutional Neural Network architecture]


# --- 3. Compile the Model ---
print("Compiling the model...")
model.compile(
    optimizer='adam',  # Adam is an efficient and popular optimization algorithm
    loss='categorical_crossentropy',  # Suitable for multi-class classification
    metrics=['accuracy']
)

# --- 4. Train the Model ---
print("Starting model training...")
# We'll train for 15 epochs. An epoch is one complete pass through the entire training dataset.
# validation_split reserves a portion of the training data to evaluate the loss and any model metrics at the end of each epoch.
history = model.fit(
    x_train,
    y_train,
    epochs=15,
    batch_size=64,
    validation_split=0.2
)
print("Model training finished.")


# --- 5. Evaluate the Model ---
print("Evaluating model performance on the test set...")
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")


# --- 6. Save the Model ---
print("Saving the trained model...")
# Save the entire model to a single H5 file.
# This includes the model's architecture, weights, and training configuration.
model.save('cifar10_classifier.h5')
print("Model saved as 'cifar10_classifier.h5'")
