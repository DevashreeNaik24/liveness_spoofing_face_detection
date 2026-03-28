import streamlit as st
import cv2
import numpy as np
from liveness_detector import LivenessDetector

# Configure Streamlit page
st.set_page_config(page_title="Anti-Spoofing Face Liveness Detection System", page_icon="👤", layout="wide")

# App title and description
st.title("Anti-Spoofing Face Liveness Detection System")
st.markdown("""
This application detects whether a person's face is **LIVE** or a **SPOOF** (photo/video).
- **Liveness Indicators:** Blink detection and 3D face structure analysis.
- **Instructions:** Ensure your face is clearly visible to the camera. Blink to prove you are live.
""")

# Sidebar settings
st.sidebar.header("System Settings")
reset_btn = st.sidebar.button("Reset Counters")

# Initialize liveness detector
if 'detector' not in st.session_state:
    st.session_state.detector = LivenessDetector()

if reset_btn:
    st.session_state.detector.reset_counters()
    st.success("Counters reset!")

# Main UI layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Webcam Feed")
    # Streamlit webcam widget (simple version)
    # Note: Streamlit's `st.camera_input` is great for snapshots, but for a live feed,
    # we usually use a loop or a specialized component like `streamlit-webrtc`.
    # For a "basic UI", we'll use a simple loop with `st.image` if we can access the webcam.
    # On some environments, direct webcam access via OpenCV in a loop might be tricky.
    # But for a local project, this is standard.

    run = st.checkbox("Start Webcam Feed")
    frame_placeholder = st.empty()

    if run:
        cap = cv2.VideoCapture(0)
        while cap.isOpened() and run:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture video.")
                break

            # Process frame for liveness
            processed_frame, current_status = st.session_state.detector.detect(frame)

            # Convert BGR to RGB for Streamlit
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

            # Update the frame in the UI
            frame_placeholder.image(rgb_frame, channels="RGB")

            # Update status in result_container
            if "REAL" in current_status:
                result_container.success(f"STATUS: {current_status}")
            elif "FAKE" in current_status:
                result_container.error(f"STATUS: {current_status}")
            else:
                result_container.warning(f"STATUS: {current_status}")

            # Check if user clicked the stop checkbox
            # (Note: this simple loop might be slow in Streamlit, but it's the "basic" way)
            # A more robust way would be using `streamlit-webrtc` but it's more complex.

        cap.release()
    else:
        st.info("Check 'Start Webcam Feed' to begin detection.")

with col2:
    st.subheader("Live Results")
    # Placeholders for dynamic updates
    result_container = st.empty()
    st.info("The system analyzes eye blinks and face depth variation.")
    st.warning("Ensure good lighting for better accuracy.")

# Footer
st.markdown("---")
st.caption("Built for Anti-Spoofing Face Liveness Detection.")
