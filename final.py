import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.utils import get_custom_objects


# Register the custom layer
get_custom_objects().update({'InstanceNormalization': InstanceNormalization})

# Custom CSS for sidebar styling
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        font-size: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar Styling
st.sidebar.markdown(
    """
    <style>
    .sidebar-title {
        font-size: 35px;
        text-align: center;
        font-weight: bold;
        color: #FFF;
    }
    </style>
    <div class="sidebar-title">MRIze</div>
    """,
    unsafe_allow_html=True
)

# Sidebar Navigation
st.sidebar.header("Options: ")

# Main Page
st.title("CT to MRI Conversion and Analysis")
st.subheader("Welcome to the CT to MRI conversion tool!")

# Load the pre-trained models (CT to MRI conversion model and analysis model)
@st.cache_resource
def load_models():
    model_first = load_model('first.h5')  # Load first model for CT to MRI conversion
    model_second = load_model('second.h5')  # Load second model for analysis
    return model_first, model_second

model_first, model_second = load_models()

# Sidebar Navigation Options
option = st.sidebar.selectbox(
    "Select a feature",
    ("Home", "Upload CT Scan", "Generated MRI", "Analysis"),
)

# Helper functions for processing images
def preprocess_image(image):
    # Preprocess the uploaded image before feeding it to the models
    image = image.resize((32, 32))  # Resize the image to the input size required by the model
    image_array = img_to_array(image) / 255.0  # Convert image to numpy array and normalize
    return np.expand_dims(image_array, axis=0)  # Add batch dimension

def generate_mri_from_ct(ct_image):
    # Preprocess the CT image
    preprocessed_ct = preprocess_image(ct_image)

    # Use the first model to convert the CT image to an MRI-like image
    generated_mri = model_first.predict(preprocessed_ct)
    generated_mri_image = np.squeeze(generated_mri)  # Remove extra dimensions
    return generated_mri_image

# Home Page
if option == "Home":
    st.write("Welcome to the home page!")

# Upload CT Scan
elif option == "Upload CT Scan":
    st.write("Upload your CT scan here.")
    uploaded_ct_file = st.file_uploader("Upload CT File", type=["png", "jpg", "jpeg", "dcm"])
    if uploaded_ct_file is not None:
        ct_image = Image.open(uploaded_ct_file)
        st.image(ct_image, caption="Uploaded CT Scan", use_container_width=True)
        st.success("CT scan uploaded successfully!")

        # Generate MRI from the uploaded CT
        generated_mri = generate_mri_from_ct(ct_image)
        generated_mri_image = Image.fromarray((generated_mri * 255).astype(np.uint8))  # Convert back to image format
        st.image(generated_mri_image, caption="Generated MRI Image", use_container_width=True)

# View Generated MRI
elif option == "Generated MRI":
    st.write("View the generated MRI scan.")
    # This is shown when the MRI is generated from the uploaded CT scan
    if 'generated_mri_image' in locals():
        st.image(generated_mri_image, caption="Generated MRI Image", use_container_width=True)
    else:
        st.write("Please upload a CT scan first to generate the MRI.")

# Analysis
elif option == "Analysis":
    st.write("Analyze your scans.")
    uploaded_ct_file_for_analysis = st.file_uploader("Upload CT File for Analysis", type=["png", "jpg", "jpeg", "dcm"])
    if uploaded_ct_file_for_analysis is not None:
        # Preprocess and generate MRI for the uploaded CT
        ct_image_for_analysis = Image.open(uploaded_ct_file_for_analysis)
        preprocessed_ct_for_analysis = preprocess_image(ct_image_for_analysis)
        generated_mri_for_analysis = generate_mri_from_ct(ct_image_for_analysis)

        # Prepare the inputs for the second model (both CT and MRI)
        ct_array = np.expand_dims(preprocessed_ct_for_analysis, axis=-1)  # Shape: (1, 32, 32, 3)
        mri_array = np.expand_dims(np.expand_dims(generated_mri_for_analysis, axis=-1), axis=-1)

        # Predict the brain tumor using the second model
        prediction = model_second.predict([ct_array, mri_array])
        if prediction > 0.5:
            st.success("Brain tumor detected!")
        else:
            st.success("No brain tumor detected.")
