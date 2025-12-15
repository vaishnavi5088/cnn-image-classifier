import tensorflow as tf
import os
from model import create_cnn_model
from tensorflow.keras.callbacks import EarlyStopping

# --- CONFIGURATION ---
DATA_DIR = '../data/train'
IMG_HEIGHT = 180
IMG_WIDTH = 180
BATCH_SIZE = 32
# Set max epochs high, EarlyStopping will stop it sooner
MAX_EPOCHS = 30 
MODEL_SAVE_PATH = '../models/my_cnn_model_augmented.keras'

def main():
    # 1. Load Data
    print("Loading data...")
    # This assumes your data/train folder now contains daisy, roses, etc.
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="training", seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="validation", seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"Classes found: {class_names}")

    # 2. Define Data Augmentation & Performance Optimization
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])
    
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # 3. Build and Define Model
    print("Building MobileNetV2 model...")
    # This function creates the MobileNetV2 base model (FROZEN) plus our classification head
    model_wrapper, base_model = create_cnn_model((IMG_HEIGHT, IMG_WIDTH, 3), num_classes)
    
    loss_fn = 'sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy'
    
    # 4. Define Early Stopping Callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True
    )

    # ------------------------------------------------------------------
    # PHASE 1: Feature Extraction (Train only the classification head)
    # ------------------------------------------------------------------
    print("\n--- PHASE 1: Feature Extraction (Frozen Base) ---")
    
    model = tf.keras.Sequential([
        data_augmentation,
        model_wrapper
    ])

    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    # Train only for a short time initially
    history = model.fit(
        train_ds, 
        validation_data=val_ds, 
        epochs=10, 
        callbacks=[early_stopping]
    )
    
    # ------------------------------------------------------------------
    # PHASE 2: FINE-TUNING (Unfreeze and Retrain Gently)
    # ------------------------------------------------------------------
    print("\n--- PHASE 2: Starting Fine-Tuning (Unfrozen Base) ---")

    # Unfreeze the top layers of the base model
    base_model.trainable = True
    
    # Freeze all layers except the last 50 (to prevent catastrophic forgetting)
    for layer in base_model.layers[:-50]:
        layer.trainable = False
        
    # Recompile the model with a very low learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
        loss=loss_fn,
        metrics=['accuracy']
    )

    # Continue training from where Phase 1 left off
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=MAX_EPOCHS + 10, # Give it plenty of max time (40 total epochs)
        callbacks=[early_stopping],
        initial_epoch=history.epoch[-1] # Start counting epochs from the end of Phase 1
    )

    # 6. Save Model
    if not os.path.exists('../models'):
        os.makedirs('../models')
    model.save(MODEL_SAVE_PATH)
    print(f"\nModel saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()