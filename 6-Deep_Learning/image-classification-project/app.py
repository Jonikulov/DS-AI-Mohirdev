import pathlib
import platform

from fastai.vision.all import *
import streamlit as st
import plotly.express as px

# Avoid PosixPath problem
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
# else:
#     pathlib.WindowsPath = pathlib.PosixPath

# title
st.title("Traffic Signs Classification Model")

# model
model = load_learner("traffic_signs_model.pkl")

# image upload widget
file = st.file_uploader(
    "Upload an Image of a traffic light / stop sign / fire hydrant:",
    type=['jpg', 'jpeg', 'png', 'svg']
)

if file != None:
    # display an image
    st.image(file)
    # transfor an image to a TensorImage
    img = PILImage.create(file)
    # prediction
    pred, pred_id, probs = model.predict(img)
    st.success(f'Prediction: {pred}')
    st.info(f'Probability: {probs[pred_id]*100:.2f}%')

    # plot the probability
    fig = px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)
