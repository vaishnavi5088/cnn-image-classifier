import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from tensorflow.keras.callbacks import EarlyStopping # Required to ensure Keras loads the custom model layers

# --- CONFIGURATION ---
MODEL_PATH = '../models/my_cnn_model_augmented.keras'
IMG_HEIGHT = 180
IMG_WIDTH = 180
CLASS_NAMES = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips'] 

# --- FUNCTIONS ---

@st.cache_resource
def load_model():
    """Load the model and check file existence."""
    if not os.path.exists(MODEL_PATH):
        return None, False
    
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model, True
    except Exception as e:
        # Handle corruption or incompatibility
        st.exception(e)
        return None, False

def predict(model, image):
    """Preprocess image and make prediction."""
    img_array = image.resize((IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.preprocessing.image.img_to_array(img_array)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array, verbose=0)
    score = tf.nn.softmax(predictions[0])
    
    predicted_label = CLASS_NAMES[np.argmax(score)]
    confidence_score = np.max(score).item() # <--- FIXED: Use .item() to extract standard Python float
    
    return predicted_label, confidence_score

# --- LAYOUT AND UI (Best Professional) ---

# 1. Page Configuration
st.set_page_config(
    page_title="Advanced Image Classifier 🌻",
    layout="wide", # Use the full width of the screen
    initial_sidebar_state="expanded"
)

# 2. Sidebar for Status and Project Info
with st.sidebar:
    st.header("⚙️ System Status")
    model, model_loaded = load_model()

    if model_loaded:
        st.success("Model Status: Online (MobileNetV2)")
        st.info(f"Target Classes: {len(CLASS_NAMES)}")
    else:
        st.error("Model Status: OFFLINE. Check 'models' folder.")
        
    st.markdown("---")
    st.header("📚 Instructions")
    st.markdown("1. Use the file uploader in the main panel.")
    st.markdown("2. The model will instantly classify the flower and display confidence.")
    st.caption(f"Model Path: {MODEL_PATH}")


# 3. Main App Header
st.title("🌷 Fine-Tuned Flower Classifier Demo")
st.caption("Using Transfer Learning (MobileNetV2) for high-accuracy prediction.")
st.markdown("---")

if not model_loaded:
    st.error("Cannot proceed. The model file was not found or failed to load. Please ensure `python train.py` ran successfully.")
    st.stop()

# 4. Prediction Area
uploaded_file = st.file_uploader("🖼️ Upload a Flower Image (JPG/PNG) for Classification", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Use st.spinner for professional loading indication
    with st.spinner('🔍 Analyzing the intricate flower structure...'):
        
        image = Image.open(uploaded_file)
        
        # Create two columns for image and results
        col1, col2 = st.columns([1, 1.5]) 

        # Column 1: Display the Image
        with col1:
            st.subheader("Source Image")
            st.image(image, width='stretch', caption=uploaded_file.name) 

        # Column 2: Display Results
        with col2:
            st.subheader("Prediction Results")
            
            # Prediction logic
            label, confidence_score = predict(model, image)
            confidence_percent = confidence_score * 100
            
            # Use Metric for the main prediction (large, clear result)
            st.metric(
                label="Predicted Class", 
                value=label.upper(), 
                delta=f"{confidence_percent:.2f}% Confidence" # Display confidence in delta for sleek look
            )
            
            # Visualize confidence with a progress bar (Highly Professional Look)
            st.markdown("##### Certainty Level")
            st.progress(confidence_score)
            
            if confidence_percent > 95:
                st.balloons() # Celebrate very high confidence
                st.success(f"🎉 High Certainty! The model is highly confident this is a {label.upper()}.")
            elif confidence_percent < 75:
                st.warning("⚠️ Low Confidence. The model is uncertain. Try a clearer or different photo.")
            else:
                st.info("✅ Result validated. Confidence is good.")
            
st.markdown("---")
st.caption("Developed using TensorFlow 2.0 and MobileNetV2. Optimized for generalization.")