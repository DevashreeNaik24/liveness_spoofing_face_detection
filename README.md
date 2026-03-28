# Anti-Spoofing Face Liveness Detection System

A basic Streamlit application for detecting whether a face is live or a spoof (photo/video).

## Features
- **Real-time Face Mesh tracking** using MediaPipe.
- **Blink Detection** to verify user interaction.
- **3D Depth analysis** (via landmark Z-coordinates) to distinguish from flat photos.

## How to Run Locally
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Deployment
This app can be deployed on **Streamlit Community Cloud** by linking this GitHub repository.
