import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from model import model

model = model(1,10,10)
model.load_state_dict(torch.load("model.pth"))
model.eval()

st.set_page_config(
    page_title="MNIST AI",
    page_icon="🤖",
    layout="wide")
st.title("MNIST DIGIT PREDICTOR")
with st.expander("About this app"):
    st.info("This is a pytorch based model. In this app, we use deep learning to predict a mnist image.")

uploaded_file = st.file_uploader("Upload a image file",
                                 type = ["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")
    st.image(image,caption="Uploaded image.",width=200)
    transform = transforms.Compose([
        transforms.Resize((28,28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,))
    ])
    img = transform(image).unsqueeze(0)

    with torch.inference_mode():
        output = model(img)
        prediction = torch.softmax(output,dim=1).argmax(dim=1).item()
        st.success(f"Our model prediction is : {prediction}")