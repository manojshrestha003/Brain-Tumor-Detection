import streamlit as st
import numpy as np
import cv2
import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model('model.h5')


# Define class labels (make sure they match your training labels order)
labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

st.title("🧠 Brain Tumor Prediction App")

uploaded_file = st.file_uploader("Upload an MRI Image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channels="RGB", caption='Uploaded Image', use_column_width=True)
    
    # Resize and reshape as required
    img_resized = cv2.resize(img, (150, 150))
    img_array = np.array(img_resized)
    img_array = img_array.reshape(1, 150, 150, 3)
    
    st.write(f"Image shape after reshaping: {img_array.shape}")
    
    # Make prediction
    prediction = model.predict(img_array)
    
    # Get index of highest probability
    predicted_class = labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    
    st.markdown(f"### 🩺 Prediction: **{predicted_class.capitalize()}**")
    st.markdown(f"### 🔍 Confidence: **{confidence:.2f}%**")
