import pickle
import numpy as np
from numpy.linalg import norm
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
import streamlit as st

# Load embeddings and filenames
feature_list = np.array(pickle.load(open('E:\\Reverse-Image-Search-ML-DL-Project\\embeddings.pkl', 'rb')))
filenames = pickle.load(open('E:\\Reverse-Image-Search-ML-DL-Project\\filenames.pkl', 'rb'))

# Load ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Upload and preprocess the test image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    img = image.load_img(uploaded_image, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    # Use Nearest Neighbors to find similar images
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([normalized_result])

    # Display similar images
    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.subheader("Similar Images:")
    for file_index in indices[0][1:]:  # Exclude the uploaded image itself
        st.image(filenames[file_index], use_column_width=True)
