import os

# --- CONTENT DEFINITIONS ---

requirements_txt = """tensorflow
numpy
matplotlib
pillow
"""

readme_md = """# Image Classification Project

## How to run
1. Install dependencies: `pip install -r requirements.txt`
2. Put your images in `data/train/` (e.g., `data/train/cats` and `data/train/dogs`).
3. Train the model: `python src/train.py`
4. Predict: `python src/predict.py path/to/image.jpg`
"""

model_py = """import tensorflow as tf
from tensorflow.keras import layers, models

def create_cnn_model(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Rescaling(1./255, input_shape=input_shape))
    
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    
    activation = 'softmax' if num_classes > 2 else 'sigmoid'
    units = num_classes if num_classes > 2 else 1
    model.add(layers.Dense(units, activation=activation))
    
    return model
"""

train_py = """import tensorflow as tf
import os
from model import create_cnn_model

# --- CONFIG ---
DATA_DIR = '../data/train'
IMG_HEIGHT = 180
IMG_WIDTH = 180
BATCH_SIZE = 32
EPOCHS = 10
MODEL_SAVE_PATH = '../models/my_cnn_model.keras'

def main():
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' not found. Please create it and add images.")
        return

    print("Loading data...")
    try:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            DATA_DIR, validation_split=0.2, subset="training", seed=123,
            image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            DATA_DIR, validation_split=0.2, subset="validation", seed=123,
            image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE
        )
    except ValueError:
        print("No images found. Make sure you have subfolders like 'data/train/cats' inside.")
        return

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"Classes found: {class_names}")

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    model = create_cnn_model((IMG_HEIGHT, IMG_WIDTH, 3), num_classes)
    loss_fn = 'sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy'
    
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

    print("Starting training...")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    if not os.path.exists('../models'):
        os.makedirs('../models')
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
"""

predict_py = """import tensorflow as tf
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
"""

# --- BUILDER LOGIC ---

def create_file(path, content):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Created file: {path}")

def main():
    root = "image_classifier"
    folders = [
        f"{root}/data/train", 
        f"{root}/data/val", 
        f"{root}/models", 
        f"{root}/src"
    ]
    
    # Create Folders
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"Created directory: {folder}")

    # Create Files
    create_file(f"{root}/requirements.txt", requirements_txt)
    create_file(f"{root}/README.md", readme_md)
    create_file(f"{root}/src/__init__.py", "") # Empty init file
    create_file(f"{root}/src/model.py", model_py)
    create_file(f"{root}/src/train.py", train_py)
    create_file(f"{root}/src/predict.py", predict_py)

    print("\nSuccess! Project structure created in 'image_classifier/'.")
    print("Next steps:")
    print(f"1. cd {root}")
    print("2. pip install -r requirements.txt")
    print("3. Add your images to data/train/")

if __name__ == "__main__":
    main()