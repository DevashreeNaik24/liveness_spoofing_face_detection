import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist

class LivenessDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

        # Landmark indices for EAR calculation
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]

        self.blink_count = 0
        self.eye_closed = False
        self.ear_threshold = 0.2  # Threshold for eye closed

    def calculate_ear(self, eye_landmarks):
        # Compute the distances between the vertical eye landmarks
        v1 = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
        v2 = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
        # Compute the distance between the horizontal eye landmarks
        h = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
        
        # Eye Aspect Ratio
        ear = (v1 + v2) / (2.0 * h)
        return ear

    def detect(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        status = "Unknown"
        color = (255, 255, 255) # White

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape
                landmarks = []
                for lm in face_landmarks.landmark:
                    landmarks.append((lm.x * w, lm.y * h))

                # Extract eye landmarks
                left_eye_pts = [landmarks[i] for i in self.LEFT_EYE]
                right_eye_pts = [landmarks[i] for i in self.RIGHT_EYE]

                left_ear = self.calculate_ear(left_eye_pts)
                right_ear = self.calculate_ear(right_eye_pts)
                avg_ear = (left_ear + right_ear) / 2.0

                # Simple blink detection logic
                if avg_ear < self.ear_threshold:
                    if not self.eye_closed:
                        self.eye_closed = True
                else:
                    if self.eye_closed:
                        self.blink_count += 1
                        self.eye_closed = False

                # For a basic liveness check, we can use the depth variation of landmarks
                # Flat photos won't show the same 3D structure as a real face.
                # MediaPipe Face Mesh provides Z-coordinates which are relative.
                # A simple check: if the face is too "flat" (small Z variation), it might be a spoof.
                z_coords = [lm.z for lm in face_landmarks.landmark]
                z_std = np.std(z_coords)

                # Heuristic thresholds (may need tuning)
                if z_std > 0.02 and self.blink_count > 0:
                    status = "REAL / LIVE"
                    color = (0, 255, 0) # Green
                elif z_std < 0.015:
                    status = "FAKE / SPOOF"
                    color = (0, 0, 255) # Red
                else:
                    status = "Detecting..."
                    color = (255, 165, 0) # Orange

                # Draw landmarks
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=self.drawing_spec,
                    connection_drawing_spec=self.drawing_spec
                )

                # Display info on frame
                cv2.putText(frame, f"Status: {status}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame, f"Blinks: {self.blink_count}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Depth (std): {z_std:.4f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame, status

    def reset_counters(self):
        self.blink_count = 0
        self.eye_closed = False
