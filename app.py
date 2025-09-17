import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- Configuration ---
st.set_page_config(
    page_title="Image Classifier",
    page_icon="🖼️",
    layout="centered"
)

# Class names for CIFAR-10 dataset
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# --- Model Loading ---
# Use st.cache_resource to load the model only once
@st.cache_resource
def load_my_model():
    """Loads and returns the trained Keras model."""
    try:
        model = tf.keras.models.load_model('cifar10_classifier.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_my_model()

# --- Preprocessing Function ---
def preprocess_image(image):
    """Preprocesses the uploaded image to fit the model's input requirements."""
    # Resize the image to 32x32 pixels, as required by the model
    img = image.resize((32, 32))
    # Convert image to a numpy array
    img_array = np.array(img)
    # The model expects a batch of images, so add a batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Normalize the pixel values to be between 0 and 1
    img_array = img_array.astype('float32') / 255.0
    return img_array

# --- UI and Prediction ---
st.title("🖼️ CIFAR-10 Image Classifier")
st.markdown("Upload an image and the model will predict which of the 10 classes it belongs to.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    
    with st.spinner('Classifying...'):
        # Preprocess the image
        preprocessed_image = preprocess_image(image)
        
        # Make a prediction
        prediction = model.predict(preprocessed_image)
        
        # Get the class with the highest probability
        predicted_class_index = np.argmax(prediction)
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        confidence = np.max(prediction) * 100
    
    st.success(f"**Prediction:** {predicted_class_name}")
    st.info(f"**Confidence:** {confidence:.2f}%")

elif uploaded_file is None:
    st.info("Please upload an image file.")
