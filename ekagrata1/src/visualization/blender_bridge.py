
import socket
import json
import time
import numpy as np


# ---------------------------------------------------------------------------
# Coordinate-frame conversion: MediaPipe world → Blender world
# ---------------------------------------------------------------------------
# MediaPipe world space:  x = right,  y = DOWN,  z = toward camera (negative)
# Blender world space:    x = right,  y = forward (into scene), z = UP
#
# Mapping:  blender_x =  mp_x
#           blender_y = -mp_z
#           blender_z = -mp_y
#
# To flip axes (e.g. mirroring left/right), change the signs here.
# ---------------------------------------------------------------------------

def mediapipe_to_blender(vec: np.ndarray) -> np.ndarray:
    """
    Convert a 3-D vector from MediaPipe world space to Blender world space.

    MediaPipe:  +x right  |  +y DOWN          |  +z toward camera (negative depth)
    Blender:    +x right  |  +y forward (into) |  +z UP

    Mapping applied:
        blender = (mp_x, -mp_z, -mp_y)

    Edit this function to flip axes for mirroring or re-orientation.
    """
    mp_x, mp_y, mp_z = float(vec[0]), float(vec[1]), float(vec[2])
    return np.array([mp_x, -mp_z, -mp_y], dtype=np.float64)


def _unit(vec: np.ndarray) -> np.ndarray:
    """Return the unit vector of *vec*, or a zero vector if near-zero."""
    n = np.linalg.norm(vec)
    if n < 1e-6:
        return np.zeros(3, dtype=np.float64)
    return vec / n


# Exponential-smoothing factor: 0 = no smoothing, 1 = frozen.
# Values 0.3-0.5 reduce jitter while still feeling responsive.
_ALPHA = 0.4


class BlenderBridge:
    def __init__(self, host="127.0.0.1", port=9999, enabled=True):
        """
        TCP client that sends arm direction vectors to Blender every frame.

        Payload format (JSON, newline-terminated):
        {
          "right_upper": [dx, dy, dz],   # unit vector: right shoulder -> right elbow (Blender space)
          "right_fore":  [dx, dy, dz],   # unit vector: right elbow   -> right wrist  (Blender space)
          "left_upper":  [dx, dy, dz],
          "left_fore":   [dx, dy, dz]
        }

        Missing or low-visibility vectors are omitted from the payload;
        Blender falls back to the last known value.
        """
        self.host = host
        self.port = port
        self.enabled = enabled
        self.sock = None
        self.connected = False
        self._last_reconnect_attempt = 0.0

        # Smoothed direction vectors (Blender space, unit length)
        self._smooth: dict[str, np.ndarray] = {}

        # Last sent payload – re-sent when landmarks are missing
        self._last_payload: dict | None = None

        if self.enabled:
            self.connect()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(0.1)
            self.sock.connect((self.host, self.port))
            self.connected = True
            print(f"[BlenderBridge] Connected to Blender at {self.host}:{self.port}")
        except Exception as e:
            self.connected = False
            self.sock = None

    def _try_reconnect(self):
        now = time.time()
        if now - self._last_reconnect_attempt > 2.0:
            self._last_reconnect_attempt = now
            self.connect()

    def send_payload(self, payload: dict):
        """Send precomputed direction vectors payload directly to Blender."""
        if not self.enabled:
            return
        if not self.connected:
            self._try_reconnect()
        if not self.connected:
            return
        self._last_payload = payload
        self._send_json(payload)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _smooth_vec(self, key: str, new_vec: np.ndarray) -> np.ndarray:
        """Apply exponential smoothing and return the smoothed unit vector."""
        if key not in self._smooth:
            self._smooth[key] = new_vec
        else:
            self._smooth[key] = _ALPHA * new_vec + (1.0 - _ALPHA) * self._smooth[key]
        return _unit(self._smooth[key])

    def _direction(self, landmarks: dict, from_key: str, to_key: str) -> np.ndarray | None:
        """
        Compute the unit direction vector from one landmark to another,
        converted to Blender space.  Returns None if either landmark is absent.
        """
        if from_key not in landmarks or to_key not in landmarks:
            return None
        raw = landmarks[to_key]['pos'] - landmarks[from_key]['pos']
        blender_vec = mediapipe_to_blender(raw)
        return _unit(blender_vec)

    def _send_json(self, payload: dict):
        """Send *payload* as a newline-terminated JSON string."""
        try:
            msg = json.dumps(payload) + "\n"
            self.sock.sendall(msg.encode('utf-8'))
        except Exception as e:
            print(f"[BlenderBridge] Lost connection to Blender: {e}")
            self.connected = False
            if self.sock:
                self.sock.close()
                self.sock = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send_vectors(self, landmarks: dict | None):
        """
        Build direction vectors from *landmarks* (the dict returned by
        PoseTracker.get_landmarks), smooth them, and send to Blender.

        If *landmarks* is None or a vector cannot be computed (missing joint),
        the last valid value is resent so Blender never gets stale data.

        Args:
            landmarks: dict keyed by joint name, each value:
                         {'pos': np.array([x, y, z]), 'vis': float}
                       or a pre-computed payload dict {'right_upper': [x,y,z], ...}
                       or None if no pose was detected this frame.
        """
        if not self.enabled:
            return

        if isinstance(landmarks, dict) and any(k in landmarks for k in ("right_upper", "left_upper")):
            first_val = landmarks.get("right_upper") or landmarks.get("left_upper")
            if isinstance(first_val, (list, tuple, np.ndarray)):
                self.send_payload(landmarks)
                return

        payload: dict[str, list] = {}
        lm = landmarks or {}

        # Compute (or fall back to) each direction vector
        arm_segments = [
            ("right_upper", "right_shoulder", "right_elbow"),
            ("right_fore",  "right_elbow",    "right_wrist"),
            ("left_upper",  "left_shoulder",  "left_elbow"),
            ("left_fore",   "left_elbow",     "left_wrist"),
        ]

        for key, from_joint, to_joint in arm_segments:
            vec = self._direction(lm, from_joint, to_joint)
            if vec is not None:
                smoothed = self._smooth_vec(key, vec)
                payload[key] = smoothed.tolist()
            elif key in self._smooth:
                # Resend last known smoothed value
                payload[key] = _unit(self._smooth[key]).tolist()
            # If we have no previous value at all, omit – Blender keeps its rest pose

        if not payload:
            return

        self._last_payload = payload

        if not self.connected:
            return

        self._send_json(payload)

    # ------------------------------------------------------------------
    # Legacy compatibility shim (used by main.py's angle path)
    # ------------------------------------------------------------------

    def send_angles(self, angles_dict: dict):
        """
        Legacy method kept so the existing main.py call site compiles.
        Does nothing – angle-based control is superseded by send_vectors().
        """
        pass

    # ------------------------------------------------------------------

    def close(self):
        if self.sock:
            self.sock.close()
            self.sock = None
