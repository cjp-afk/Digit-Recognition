import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import load_model

tf.get_logger().setLevel("ERROR")


def initialise_model(X_train=None, y_train=None, X_test=None, y_test=None):
    print("[+] Loading the data")
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    print("[+] Normalizing the data")
    X_train = X_train.reshape((60000, 28, 28, 1)).astype("float32") / 255
    X_test = X_test.reshape((10000, 28, 28, 1)).astype("float32") / 255

    print("[+] Creating the model")
    model = keras.Sequential(
        [
            keras.layers.Conv2D(
                32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)
            ),
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
        ]
    )

    print("[+] Compiling the model")
    model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    print("[+] Training the model")
    model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=64,
        validation_data=(X_test, y_test),
        verbose=1,
        shuffle=True,
    )

    print("[+] Evaluating the model")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f"Test accuracy: {round(test_acc*100, 2)}%")
    print(f"Test loss: {test_loss}")

    print("[+] Saving model as mnist_model.h5")
    model.save("mnist_model.h5")


def predict(image):
    model = load_model("mnist_model.h5")

    prediction = model.predict(image)
    prediction

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))

    colors = ["blue" if x < 0.5 else "green" for x in prediction[0]]

    ax1.bar(range(10), prediction[0] * 100, width=0.8, color=colors)
    ax1.set_title("Model Predictions")
    ax1.set_xlabel("Digit")
    ax1.set_ylabel("Confidence (%)")
    ax1.set_ylim(0, 100)

    ax2.imshow(image[0], cmap="gray")
    ax2.set_title(f"Prediction: {np.argmax(prediction[0])}")

    plt.tight_layout()
    plt.show()
