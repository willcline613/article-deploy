import numpy as np
import streamlit as st
from PIL import Image, ImageOps


def import_and_predict(image_data, model):
    
        size = (256,256) 
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = image.convert('RGB')
        image = np.asarray(image)
        image = (image.astype(np.float32) / 256.0)

        img_reshape = image[np.newaxis,...]

        prediction = model.predict(img_reshape)
        
        return prediction

class Model:

    def predict(self, X):
        return 'We predict that you uploaded an image!'

model = Model()


st.write("""
         # Image Predictor
         """
         )

st.write("This is a simple image classification web app.")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])
#
if file is None:
    st.text("You haven't uploaded an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    

    st.write(prediction)

    
