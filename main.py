import streamlit as st
import os
import numpy as np
from numpy.linalg import norm
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import pickle
from PIL import Image

# Check if the 'uploads' directory exists; create it if not
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Load the embeddings and filenames
feature_list = np.array(pickle.load(open('E:\\Reverse-Image-Search-ML-DL-Project\\embeddings.pkl', 'rb')))
filenames = pickle.load(open('E:\\Reverse-Image-Search-ML-DL-Project\\filenames.pkl', 'rb'))

# Load ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

st.title('Reverse Image Search (E-Commerce)')

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except Exception as e:
        st.error(f"An error occurred while saving the file: {e}")
        return 0

def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices

# File upload -> save
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # Display the uploaded image
        display_image = image.load_img(os.path.join("uploads", uploaded_file.name), target_size=(224, 224))
        st.image(display_image, caption="Uploaded Image", use_column_width=True)
        
        # Feature extraction
        img_array = image.img_to_array(display_image)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        result = model.predict(preprocessed_img).flatten()
        normalized_result = result / norm(result)

        # Recommendation
        indices = recommend(normalized_result, feature_list)
        
        # Show recommended images
        st.subheader("Similar Images:")
        for file_index in indices[0][1:6]:  # Exclude the uploaded image itself
            recommended_image = Image.open(filenames[file_index])
            st.image(recommended_image, caption="Recommended Image", use_column_width=True)

    else:
        st.header("Some error occurred in file upload")
