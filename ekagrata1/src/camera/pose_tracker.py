import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp

USE_LEGACY = hasattr(mp, 'solutions')

if not USE_LEGACY:
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

# Landmark indices in the MediaPipe 33-point pose model
_LANDMARK_INDICES = {
    'left_shoulder':  11,
    'right_shoulder': 12,
    'left_elbow':     13,
    'right_elbow':    14,
    'left_wrist':     15,
    'right_wrist':    16,
    'left_hip':       23,
    'right_hip':      24,
}

# Landmarks with visibility below this threshold are treated as missing
VISIBILITY_THRESHOLD = 0.5


class PoseTracker:
    def __init__(self, camera_index=0, width=1280, height=720):
        """
        Initializes the MediaPipe Pose tracker and opens the webcam.
        Supports both legacy mp.solutions (MediaPipe < 1.0) and
        modern MediaPipe Tasks Vision (MediaPipe >= 1.0).
        """
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera index {camera_index}")

        if USE_LEGACY:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                model_complexity=1
            )
            self.mp_drawing = mp.solutions.drawing_utils
            # Map joint name -> legacy landmark enum value
            self._legacy_indices = {
                name: self.mp_pose.PoseLandmark[name.upper()].value
                for name in _LANDMARK_INDICES
            }
        else:
            possible_paths = [
                os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "pose_landmarker.task")),
                os.path.abspath("pose_landmarker.task"),
                "pose_landmarker.task"
            ]
            model_path = next((p for p in possible_paths if os.path.exists(p)), None)

            if model_path is None:
                model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "pose_landmarker.task"))
                print("Downloading pose_landmarker model...")
                url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
                urllib.request.urlretrieve(url, model_path)
                print("Download complete.")

            base_options = mp_python.BaseOptions(model_asset_path=model_path)
            options = mp_vision.PoseLandmarkerOptions(
                base_options=base_options,
                output_segmentation_masks=False,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_landmark(self, lm_list, index):
        """
        Safely pulls one landmark out of a list.

        Returns:
            (np.array([x, y, z]), float visibility)  – if visible enough
            (None, float visibility)                  – if below VISIBILITY_THRESHOLD
        """
        lm = lm_list[index]
        vis = getattr(lm, 'visibility', None)
        vis = float(vis) if vis is not None else 1.0  # Tasks API may not expose visibility
        if vis < VISIBILITY_THRESHOLD:
            return None, vis
        return np.array([lm.x, lm.y, lm.z], dtype=np.float32), vis

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_landmarks(self):
        """
        Captures a frame and processes it with MediaPipe Pose.

        Returns:
            landmarks_3d (dict | None):
                Keys: joint names from _LANDMARK_INDICES
                      ('left_shoulder', 'right_shoulder', 'left_elbow',
                       'right_elbow', 'left_wrist', 'right_wrist',
                       'left_hip', 'right_hip').
                Each value: {'pos': np.array([x, y, z]), 'vis': float}
                Only joints with visibility >= VISIBILITY_THRESHOLD are
                included. Returns None when no pose is detected at all.
            annotated_frame (ndarray): BGR frame with skeleton overlay.
        """
        success, frame = self.cap.read()
        if not success:
            return None, None

        landmarks_3d = {}
        annotated_frame = frame.copy()

        if USE_LEGACY:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = self.pose.process(image_rgb)
            image_rgb.flags.writeable = True
            annotated_frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            if results.pose_world_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS
                )
                wlm = results.pose_world_landmarks.landmark
                for name, idx in self._legacy_indices.items():
                    pos, vis = self._extract_landmark(wlm, idx)
                    if pos is not None:
                        landmarks_3d[name] = {'pos': pos, 'vis': vis}
            else:
                return None, annotated_frame

        else:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            results = self.landmarker.detect(mp_image)

            if results.pose_world_landmarks and len(results.pose_world_landmarks) > 0:
                wlm = results.pose_world_landmarks[0]
                for name, idx in _LANDMARK_INDICES.items():
                    pos, vis = self._extract_landmark(wlm, idx)
                    if pos is not None:
                        landmarks_3d[name] = {'pos': pos, 'vis': vis}
            else:
                return None, annotated_frame

            # Draw 2D skeleton overlay
            if results.pose_landmarks and len(results.pose_landmarks) > 0:
                lm2d = results.pose_landmarks[0]
                h, w, _ = frame.shape
                connections = [
                    (11, 12), (12, 14), (14, 16),  # Right arm
                    (11, 13), (13, 15),             # Left arm
                    (11, 23), (12, 24), (23, 24)    # Torso
                ]
                for p1, p2 in connections:
                    pt1 = (int(lm2d[p1].x * w), int(lm2d[p1].y * h))
                    pt2 = (int(lm2d[p2].x * w), int(lm2d[p2].y * h))
                    cv2.line(annotated_frame, pt1, pt2, (0, 255, 0), 3)
                for idx in [11, 12, 13, 14, 15, 16, 23, 24]:
                    pt = (int(lm2d[idx].x * w), int(lm2d[idx].y * h))
                    cv2.circle(annotated_frame, pt, 6, (0, 0, 255), -1)
                    cv2.circle(annotated_frame, pt, 2, (255, 255, 255), -1)

        return landmarks_3d if landmarks_3d else None, annotated_frame

    def close(self):
        """Releases camera resources."""
        self.cap.release()
        if USE_LEGACY and hasattr(self, 'pose'):
            self.pose.close()
        elif hasattr(self, 'landmarker'):
            self.landmarker.close()
