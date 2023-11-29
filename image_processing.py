from PIL import Image
import numpy as np


def process_image(image_path):
    print("[+] Processing the image")

    # Step 1: Read the image
    img = Image.open(image_path).convert("L")  # Convert to grayscale

    # Step 2: Resize the image
    img = img.resize((28, 28), Image.LANCZOS)

    # Step 3: Convert to a numpy array
    img_array = np.array(img)

    # Step 4: Reshape the image
    img_array = (
        img_array.reshape((1, 28, 28, 1)).astype("float32") / 255
    )  # Reshape to (1, 28, 28, 1) for Keras

    # Step 5: Convert to a tensor
    return img_array
