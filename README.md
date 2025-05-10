# 🧠 Digit Recognition

**An Interactive Handwritten Digit Classifier Built from Scratch**

---

## 🎯 Project Overview

This project is a Python-based application that combines Pygame for drawing, Pillow for image processing, and TensorFlow for machine learning. It provides an interactive way to test digit recognition using a Convolutional Neural Network (CNN) model trained on the MNIST dataset.([GitHub][1])

---

## 🧩 Core Components

### 1. **Drawing Interface (`drawer.py`)**

* **Functionality**: Provides a Pygame-based canvas for users to draw digits.
* **Implementation**: Captures mouse events to render strokes, allowing users to input digits in real-time.([GitHub][1], [GitHub][2])

### 2. **Image Processing (`image_processing.py`)**

* **Functionality**: Processes the drawn image to match the MNIST dataset format.
* **Implementation**:

  * Resizes the image to 28x28 pixels.
  * Converts the image to grayscale.
  * Normalizes pixel values to prepare for model input.([Medium][3])

### 3. **Model Architecture (`model_cnn.py`)**

* **Functionality**: Defines the CNN model architecture for digit recognition.
* **Implementation**:

  * Utilizes TensorFlow to build a sequential model.
  * Includes convolutional, pooling, and dense layers.
  * Employs softmax activation in the output layer for multi-class classification.([GitHub][1])

### 4. **Model Training and Storage (`mnist_model.h5`)**

* **Functionality**: Stores the trained CNN model.
* **Implementation**:

  * Trained on the MNIST dataset to achieve high accuracy.
  * Saved in HDF5 format for easy loading and deployment.([GitHub][4], [GitHub][5])

### 5. **Main Application (`main.py`)**

* **Functionality**: Integrates all components to run the application.
* **Implementation**:

  * Loads the trained model.
  * Captures user input from the drawing interface.
  * Processes the image and predicts the digit using the CNN model.
  * Displays the prediction to the user.([GitHub][1])

---

## 🧠 Learning Objectives

* **Integration of Multiple Libraries**: Combining Pygame, Pillow, and TensorFlow to create a cohesive application.
* **Understanding CNNs**: Gaining hands-on experience with convolutional neural networks and their application in image recognition.
* **Real-Time Processing**: Implementing real-time image processing and prediction to provide immediate feedback to users.
* **User Interface Design**: Designing an intuitive interface for users to interact with the application seamlessly.([GitHub][1])

---

## 🚀 Getting Started

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/cjp-afk/Digit-Recognition.git
   cd Digit-Recognition
   ```

2. **Install Dependencies**:
   Ensure you have Python installed. Then, install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:

   ```bash
   python main.py
   ```
