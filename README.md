# README for Digit Recognition Project

## Introduction

The Digit Recognition project is a Python-based application that combines Pygame for drawing, Pillow for image processing, and TensorFlow for machine learning. It provides an interactive way to test digit recognition using a Convolutional Neural Network (CNN) model trained on the MNIST dataset.

Below are snippets from the main code, followed by a brief explanation.
## Main Components

### 1. Drawing Interface - `drawer.py`
This module manages the drawing interface where users can draw digits.
- **Key Functionality**:
  ```python
  # Main game loop for drawing
  running = True
  while running:
      for event in pygame.event.get():
          if event.type == pygame.QUIT:
              running = False
          elif event.type == pygame.MOUSEBUTTONDOWN:
              is_drawing = True
          elif event.type == pygame.MOUSEBUTTONUP:
              is_drawing = False
          elif event.type == pygame.MOUSEMOTION and is_drawing:
              mouse_pos = pygame.mouse.get_pos()
              pygame.draw.circle(window, drawing_color, mouse_pos, 8)
  ```

### 2. Image Processing - `image_processing.py`
Processes the saved image for digit recognition.
- **Key Functionality**:
  ```python
  # Convert to grayscale and resize
  img = Image.open(image_path).convert("L")
  img = img.resize((28, 28), Image.LANCZOS)
  img_array = np.array(img)
  img_array = img_array.reshape((1, 28, 28, 1)).astype("float32") / 255
  ```

### 3. Main Application - `main.py`
The entry point of the application.
- **Key Functionality**:
  ```python
  if __name__ == "__main__":
      choice = input("Train new model (1)\nUse existing model (2)\n-> ")
      if choice == "1":
          initialise_model()
      elif choice == "2":
          run_app()
  ```

### 4. CNN Model - `model_cnn.py`
Defines and trains the CNN model for digit recognition.
- **Model Architecture**:
  ```python
  model = keras.Sequential([
      keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)),
      keras.layers.BatchNormalization(),
      keras.layers.Conv2D(64, (3, 3), activation="relu"),
      keras.layers.BatchNormalization(),
      keras.layers.Conv2D(16, (3, 3), activation="relu"),
      keras.layers.BatchNormalization(),
      keras.layers.MaxPooling2D(pool_size=(2, 2)),
      keras.layers.Dropout(0.35),
      keras.layers.Flatten(),
      keras.layers.Dense(128, activation="relu"),
      keras.layers.Dropout(0.55),
      keras.layers.Dense(10, activation="softmax"),
  ])
  ```
- **Explanation of Layers**:
  - **Conv2D Layers**: These layers perform convolution operations, capturing the spatial features from the image. The first layer has 32 filters, followed by 64 and then 16, each with a kernel size of 3x3. The 'relu' activation function introduces non-linearity, allowing the model to learn more complex patterns.
  - **BatchNormalization**: This layer normalizes the activations from the previous layer, which helps in speeding up the training process and reducing the sensitivity to network initialization.
  - **MaxPooling2D**: This layer reduces the spatial dimensions (height and width) of the input volume, which helps in reducing the number of parameters, computation in the network, and also controls overfitting.
  - **Dropout**: Dropout layers randomly set a fraction of input units to 0 at each update during training, which helps in preventing overfitting. The dropout rates are 0.35 and 0.55 after the pooling and dense layers, respectively.
  - **Flatten**: This layer flattens the input without affecting the batch size. It is used when transitioning from convolutional layers to dense layers.
  - **Dense**: These are fully connected layers. The first dense layer has 128 neurons, and the final layer has 10 neurons (one for each digit) with a 'softmax' activation function, which is used for multi-class classification.

### 5. Dependencies - `requirements.txt`
Lists all the necessary Python libraries for the project, including Pillow, Pygame, Matplotlib, and TensorFlow.

---
