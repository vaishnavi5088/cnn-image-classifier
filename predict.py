import tensorflow as tf
import numpy as np
import sys
import os

MODEL_PATH = '../models/my_cnn_model.keras'
IMG_HEIGHT = 180
IMG_WIDTH = 180

# UPDATE THIS LIST TO MATCH YOUR TRAINING FOLDERS
# Example: If you have folders 'cat' and 'dog', list them alphabetically.
CLASS_NAMES = ['class_a', 'class_b'] 

def predict_image(image_path):
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Run train.py first.")
        return

    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    img = tf.keras.utils.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    
    if len(CLASS_NAMES) > 2:
        score = tf.nn.softmax(predictions[0])
        class_index = np.argmax(score)
        confidence = 100 * np.max(score)
    else:
        score = predictions[0][0]
        if score > 0.5:
            class_index = 1
            confidence = 100 * score
        else:
            class_index = 0
            confidence = 100 * (1 - score)

    # Safety check for index
    if class_index < len(CLASS_NAMES):
        print(f"Result: {CLASS_NAMES[class_index]} ({confidence:.2f}%)")
    else:
        print(f"Result: Class index {class_index}, Confidence {confidence:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
    else:
        predict_image(sys.argv[1])
