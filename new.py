import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array  # Updated import

# Define the custom InstanceNormalization class (same as provided above)
class InstanceNormalization(Layer):
    # ... (InstanceNormalization definition as before)

# Load the model that uses InstanceNormalization
model = load_model("first.h5", custom_objects={'InstanceNormalization': InstanceNormalization})

# Preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((32, 32))  # Resize to fit model input size
    image_array = img_to_array(image) / 255.0  # Normalize image
    return np.expand_dims(image_array, axis=0)  # Add batch dimension

# Example Streamlit App
st.title("CT to MRI Image Generator")

uploaded_file = st.file_uploader("Upload a CT Scan Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    preprocessed_image = preprocess_image(image)

    # Predict MRI from CT using the loaded model
    generated_mri = model.predict(preprocessed_image)
    generated_mri_image = np.squeeze(generated_mri)  # Remove extra dimensions

    # Display the generated MRI image
    st.image(generated_mri_image, caption="Generated MRI", use_column_width=True)
