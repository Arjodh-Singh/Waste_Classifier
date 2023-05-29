import tensorflow as tf
model = tf.keras.models.load_model('/Users/arjodh_singh/Downloads/Waste_Classifier.h5')
model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

import streamlit as st
st.write("""
         # Waste Item Prediction
         """
         )
st.write("This is a simple image classification web app to predict waste items into recyclable and non-recyclable")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

import cv2
from PIL import Image, ImageOps
import numpy as np
def import_and_predict(image_data, model):
    
    size = (256,256)    
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    
    prediction = model.predict(np.expand_dims(img/255, 0))
    return prediction
            
if file is None:
    st.text("Please upload an image file")
else:
    
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    if prediction>0.5:
        st.write("Predicted class is Recyclable!")
    
    else:
        st.write("Predicted class is Non-Recyclable!")
    
    st.write(prediction)