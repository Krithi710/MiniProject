import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# TensorFlow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return predictions  # Return the prediction array
# Streamlit Theme Customization
st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
    .main {
        background-color: #121212;  /* Dark background */
        color: #E0E0E0;  /* Light text color */
    }
    .sidebar .sidebar-content {
        background-color: #1E1E1E;  /* Dark sidebar */
        color: #E0E0E0;  /* Light text color */
    }
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF;  /* White for headers */
    }
    .stButton > button {
        background-color: #76FF03;  /* Bright green for buttons */
        color: #121212;  /* Dark text color for buttons */
        padding: 10px 20px;  /* Increase button padding */
        border-radius: 5px;  /* Add button border radius */
        border: none;  /* Remove border */
        font-weight: bold;  /* Bold text */
    }
    .stFileUploader > div:first-child > label {
        background-color: #76FF03;  /* Bright green for file uploader label */
        color: #121212;  /* Dark text color for file uploader label */
        padding: 10px 20px;  /* Increase file uploader label padding */
        border-radius: 5px;  /* Add file uploader label border radius */
        font-weight: bold;  /* Bold text */
    }
    .stSpinner {
        color: #76FF03;  /* Spinner color */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.title("🌟 Dashboard 🌟")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Detection"])

# Main Page
if app_mode == "Home":
    st.header("🌱 Plant Disease Detection System 🌱")
    st.image(r"C:\Users\anvit\Downloads\Plant disease\Streamlit-demo\farmer.jpg", width=300)  # Adjust the width as per your requirement
    st.markdown("""
    Welcome to the **Plant Disease Detection System**! 🌿🔍
    This tool uses advanced machine learning techniques to help you identify plant diseases.
    
    **How to Use:**
    1. Go to the 'Disease Detection' page.
    2. Upload a clear image of your plant leaf.
    3. Click 'Predict' to receive results.

    **Features:**
    - **Real-Time Prediction**: Quickly identify plant diseases from uploaded images.
    - **User-Friendly Interface**: Easy navigation and clear results.
    - **Confidence Threshold**: Set a minimum confidence level for predictions.
    """)

# About Project
elif app_mode == "About":
    st.header("🔍 About This Project 🔍")
    st.markdown("""
                This project leverages deep learning to accurately detect plant diseases from leaf images. 
                The model was trained on a large dataset with:
                - **Training Images**: 70,295
                - **Test Images**: 33
                - **Validation Images**: 17,572
                
                **Objectives:**
                - Provide early disease detection.
                - Improve crop health management.
                - Assist farmers with actionable insights.
                """)

# Prediction Page
elif app_mode == "Disease Detection":
    st.header("🌟 Disease Detection 🌟")
    st.markdown("""
    **Upload an image of a plant leaf to get started.** Our model will analyze the image and provide predictions along with confidence scores.
    """)
    
    test_image = st.file_uploader("Choose an Image:", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")
    
    if test_image is not None:
        image = Image.open(test_image)
        st.image(image, caption="Uploaded Image", width=300)
        st.write("Processing...")

        # Add a progress bar
        with st.spinner('Analyzing the image...'):
            # Predict button
            if st.button("Predict"):
                predictions = model_prediction(test_image)
                confidence = np.max(predictions)  # Get the highest confidence score
                result_index = np.argmax(predictions)  # Get the index of the highest confidence score
                
                # Reading Labels
                class_name = [
                    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
                    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
                    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 
                    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
                    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 
                    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 
                    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 
                    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 
                    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 
                    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
                ]

                confidence_threshold = 0.75  # Set your desired confidence threshold here

                if confidence > confidence_threshold:
                    st.success(f"🌟 **Prediction:** {class_name[result_index]} with confidence {confidence:.2f}")
                else:
                    st.warning("⚠️ The model is not confident enough in its prediction. Please try another image.")
    else:
        st.info("📁 Please upload an image to get a prediction.")