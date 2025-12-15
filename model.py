import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2 

def create_cnn_model(input_shape, num_classes):
    """
    Creates a Transfer Learning model using MobileNetV2 as the base.
    Returns: (The classification model, the MobileNetV2 base model)
    """
    # 1. Load the Pre-trained Base Model (MobileNetV2)
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # 2. Freeze the Base Layers (Default state for Phase 1)
    base_model.trainable = False

    # 3. Build the New Model Head (Our Classifier)
    model = models.Sequential()
    
    # Start with Rescaling
    model.add(layers.Rescaling(1./255, input_shape=input_shape))
    
    # Add the pre-trained feature extractor
    model.add(base_model)
    
    # Add a layer to reduce the huge feature maps to a single vector
    model.add(layers.GlobalAveragePooling2D())
    
    # Add our custom classification layers
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5)) 
    
    # Output Layer (based on number of classes)
    activation = 'softmax' if num_classes > 2 else 'sigmoid'
    units = num_classes if num_classes > 2 else 1
    
    model.add(layers.Dense(units, activation=activation))

    # We return the compiled model and the base model separately
    return model, base_model