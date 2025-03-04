import cv2
import torch
import numpy as np
import streamlit as st
from PIL import Image
import time
import asyncio  # Fix for event loop issues

# Ensure a running event loop exists
asyncio.set_event_loop(asyncio.new_event_loop())

# Streamlit UI
st.title("Real-Time Depth Estimation using MiDaS")

# Controls
st.title("Controls")
start_stream = st.button("Start Stream")
stop_stream = st.button("Stop Stream")

# Load MiDaS model
@st.cache_resource
def load_midas():
    midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
    midas.to('cpu')
    midas.eval()
    return midas

@st.cache_resource
def load_transform():
    transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
    return transforms.small_transform

midas = load_midas()
transform = load_transform()

# Placeholders for side-by-side display
col1, col2 = st.columns(2)
frame_placeholder = col1.empty()
depth_placeholder = col2.empty()

# Stream control flag
streaming = False

if start_stream:
    streaming = True
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if stop_stream:
    streaming = False
    if 'cap' in locals():
        cap.release()

# Start video processing if streaming is True
while streaming:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgbatch = transform(img).to('cpu')

    # Perform depth estimation
    with torch.no_grad():
        prediction = midas(imgbatch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze().cpu().numpy()

    # Normalize depth for visualization
    depth_map = (prediction - prediction.min()) / (prediction.max() - prediction.min()) * 255
    depth_map = depth_map.astype(np.uint8)

    # Convert depth map to image
    depth_colormap = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
    depth_pil = Image.fromarray(depth_colormap)

    # Display images side by side
    frame_placeholder.image(img, channels="RGB", caption="Live Camera Feed", use_container_width=True)
    depth_placeholder.image(depth_pil, caption="Depth Map", use_container_width=True)

    # Sleep to reduce CPU usage
    time.sleep(0.03)

if not streaming:
    st.write("Stream Stopped")
    frame_placeholder.empty()
    depth_placeholder.empty()
