import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn

# -----------------------------
# Load Model (runs only once)
# -----------------------------
@st.cache_resource
def load_model():
    
    model = models.resnet18(weights=None)   # same as training
    model.fc = nn.Linear(model.fc.in_features, 2)

    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()

    return model


model = load_model()


# -----------------------------
# Class names (same order as training dataset)
# -----------------------------
classes = ['hotspot', 'nonhotspot']


# -----------------------------
# Image Transform (MUST match training)
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Lithography Hotspot Detection")

st.write(
"""
Upload a lithography layout image and the model will classify whether it contains a **hotspot** or **non-hotspot** pattern.
"""
)


uploaded_file = st.file_uploader("Upload Layout Image", type=["png","jpg","jpeg"])


# -----------------------------
# Prediction
# -----------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_column_width=True)

    image_tensor = transform(image).unsqueeze(0)

    if st.button("Detect Hotspot"):

        with torch.no_grad():

            outputs = model(image_tensor)

            probs = torch.softmax(outputs, dim=1)

            confidence, predicted = torch.max(probs,1)

        result = classes[predicted.item()]

        confidence_score = confidence.item()*100


        st.subheader("Prediction Result")

        if result == "hotspot":
            st.error(f"Hotspot Detected ⚠️  (Confidence: {confidence_score:.2f}%)")

        else:
            st.success(f"Non-Hotspot Pattern ✅  (Confidence: {confidence_score:.2f}%)")