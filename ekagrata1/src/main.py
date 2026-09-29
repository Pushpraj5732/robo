import yaml
import time
import math
import sys
import argparse
import cv2
import numpy as np

from camera.pose_tracker import PoseTracker
from sensors.imu_receiver import ImuReceiver
from fusion.pose_imu_fusion import PoseImuFusion
from analysis import joint_angles
from analysis.smash_detector import SmashDetector
from visualization.blender_bridge import BlenderBridge
from report.session_report import SessionReport


def load_config(path="config.yaml"):
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}. Using defaults.")
        return {}


# ---------------------------------------------------------------------------
# Coordinate-frame conversion: MediaPipe world -> Blender world
# ---------------------------------------------------------------------------

def mediapipe_to_blender(vec: np.ndarray, mirror_x: bool = False, flip_depth: bool = False) -> np.ndarray:
    """
    Convert a 3-D direction vector from MediaPipe world space to Blender world space in ONE function.

    MediaPipe world space:
        +x = right (from camera's perspective)
        +y = DOWN
        +z = toward camera / depth

    Blender world space:
        +x = right
        +y = forward (into scene)
        +z = UP

    Base mapping:
        blender = (mp_x, -mp_z, -mp_y)

    Toggles:
        mirror_x:   flips the horizontal axis (blender_x = -mp_x)
        flip_depth: flips the depth sign     (blender_y = +mp_z)
    """
    mp_x, mp_y, mp_z = float(vec[0]), float(vec[1]), float(vec[2])
    bx = -mp_x if mirror_x else mp_x
    by =  mp_z if flip_depth else -mp_z
    bz = -mp_y
    return np.array([bx, by, bz], dtype=np.float64)


# ---------------------------------------------------------------------------
# 3D Rotation helper for T-pose calibration
# ---------------------------------------------------------------------------

def rotation_matrix_from_vectors(vec_from: np.ndarray, vec_to: np.ndarray) -> np.ndarray:
    """
    Compute the 3x3 rotation matrix that rotates vec_from to vec_to
    using Rodrigues' rotation formula.
    """
    a = vec_from / (np.linalg.norm(vec_from) + 1e-12)
    b = vec_to / (np.linalg.norm(vec_to) + 1e-12)

    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))

    if s < 1e-6:
        if c > 0.0:
            return np.eye(3, dtype=np.float64)
        else:
            axis = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            axis = np.cross(a, axis)
            norm_ax = np.linalg.norm(axis)
            if norm_ax > 1e-6:
                axis = axis / norm_ax
            return 2.0 * np.outer(axis, axis) - np.eye(3, dtype=np.float64)

    vx = np.array([
        [ 0.0, -v[2],  v[1]],
        [ v[2],  0.0, -v[0]],
        [-v[1],  v[0],  0.0]
    ], dtype=np.float64)

    R = np.eye(3, dtype=np.float64) + vx + (vx @ vx) * ((1.0 - c) / (s * s))
    return R


# Arm joint definitions: (JSON key, from_joint, to_joint)
ARM_SEGMENTS = [
    ("right_upper", "right_shoulder", "right_elbow"),
    ("right_fore",  "right_elbow",    "right_wrist"),
    ("left_upper",  "left_shoulder",  "left_elbow"),
    ("left_fore",   "left_elbow",     "left_wrist"),
]

# Exponential smoothing alpha
SMOOTH_ALPHA = 0.4


# ---------------------------------------------------------------------------
# Test mode: sweep the right arm from pointing DOWN to pointing UP
# ---------------------------------------------------------------------------

def run_arm_raise_test(blender: BlenderBridge, duration_s: float = 3.0):
    """
    Sends a synthetic sweep of the right upper-arm direction vector from
    pointing straight down (0, 0, -1) to pointing straight up (0, 0, 1)
    over *duration_s* seconds. The forearm stays along the upper arm.
    """
    print(f"[TEST] Starting arm-raise sweep over {duration_s}s …")
    print("       Right arm should move from pointing DOWN → pointing UP.")
    print("       Press Ctrl-C to cancel.")

    start = time.time()
    try:
        while True:
            elapsed = time.time() - start
            t = min(elapsed / duration_s, 1.0)

            angle_deg = -90.0 + 180.0 * t
            angle_rad = math.radians(angle_deg)

            upper_dir = np.array([0.0, 0.0, math.sin(angle_rad)], dtype=np.float64)
            fore_dir  = upper_dir.copy()

            def _unit(v):
                n = np.linalg.norm(v)
                return v / n if n > 1e-6 else v

            upper_dir = _unit(upper_dir)
            fore_dir  = _unit(fore_dir)

            payload = {
                "right_upper": upper_dir.tolist(),
                "right_fore":  fore_dir.tolist(),
            }
            if blender.connected:
                blender.send_payload(payload)
            else:
                print(f"[TEST] angle={angle_deg:+.1f}°  vec={upper_dir}  (Blender not connected)")

            if t >= 1.0:
                print("[TEST] Sweep complete.")
                break

            time.sleep(0.033)  # ~30 Hz

    except KeyboardInterrupt:
        print("[TEST] Cancelled.")


# ---------------------------------------------------------------------------
# Main session loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="EKAGRATA Motion Capture")
    parser.add_argument(
        "--test-arm-raise",
        action="store_true",
        help="Send a synthetic arm-raise sweep to Blender (no camera needed)."
    )
    args = parser.parse_args()

    print("Initializing EKAGRATA...")
    config = load_config()

    # 1. Initialize Blender Bridge
    blender_cfg = config.get('blender', {})
    blender = BlenderBridge(
        host=blender_cfg.get('host', '127.0.0.1'),
        port=blender_cfg.get('port', 9999),
        enabled=blender_cfg.get('enabled', True)
    )

    if args.test_arm_raise:
        run_arm_raise_test(blender, duration_s=10.0)
        blender.close()
        return

    # 2. Initialize Camera / Pose Tracking
    cam_cfg = config.get('camera', {})
    try:
        pose_tracker = PoseTracker(
            camera_index=cam_cfg.get('index', 0),
            width=cam_cfg.get('resolution_w', 1280),
            height=cam_cfg.get('resolution_h', 720)
        )
        print("Camera initialized.")
    except RuntimeError as e:
        print(f"CRITICAL ERROR: {e}")
        blender.close()
        return

    # 3. Initialize IMU Receiver
    imu_cfg = config.get('imu', {})
    imu = ImuReceiver(host=imu_cfg.get('host', '0.0.0.0'), port=imu_cfg.get('port', 8888))
    imu_started = imu.start()
    if not imu_started:
        print("Warning: IMU Receiver failed to start. Falling back to camera-only mode.")

    # 4. Initialize Fusion
    fusion_alpha = config.get('fusion', {}).get('alpha', 0.98)
    fusion_elbow    = PoseImuFusion(alpha=fusion_alpha)
    fusion_shoulder = PoseImuFusion(alpha=fusion_alpha)

    # 5. Initialize Smash Detector
    smash_detector = SmashDetector(config)

    # State variables for reporting
    session_best_score   = 0
    session_best_metrics = None
    angle_history = []
    time_history  = []

    # Digital twin pipeline state
    mirror_x   = False
    flip_depth = False
    smoothed_vectors: dict[str, np.ndarray] = {}

    # 3-second T-pose calibration state
    calibrating = False
    calib_start_time = 0.0
    calib_samples = {k: [] for k, _, _ in ARM_SEGMENTS}
    calib_rotations = {k: np.eye(3, dtype=np.float64) for k, _, _ in ARM_SEGMENTS}
    is_calibrated = False
    calib_display_timer = 0.0

    print("Starting Main Loop.")
    print("Controls in OpenCV window:")
    print("  'm' - Toggle mirror axis (+mp_x / -mp_x)")
    print("  'x' - Toggle depth sign  (-mp_z / +mp_z)")
    print("  'c' - Start 3-second T-pose calibration")
    print("  'q' - Exit and generate session report")

    try:
        while True:
            loop_start = time.time()
            now = loop_start

            # --- A. Get Camera Pose ---
            landmarks, frame = pose_tracker.get_landmarks()

            if frame is None:
                break  # Camera dropped

            # --- B. Get Latest IMU Data ---
            imu_data = imu.get_latest()

            # --- C. Compute Angles (for scoring / IMU fusion) ---
            lm_compat = {}
            if landmarks:
                for k, v in landmarks.items():
                    lm_compat[k] = v['pos']

            cam_elbow    = joint_angles.get_elbow_angle(lm_compat)
            cam_shoulder = joint_angles.get_shoulder_angle(lm_compat)
            cam_trunk    = joint_angles.get_trunk_rotation(lm_compat)

            final_elbow    = fusion_elbow.update(cam_elbow, imu_data)   if cam_elbow   else None
            final_shoulder = fusion_shoulder.update(cam_shoulder, imu_data) if cam_shoulder else None

            # --- D. Analyze Movement ---
            if final_elbow is not None and cam_trunk is not None:
                angle_history.append(final_elbow)
                time_history.append(now)

                smash_result = smash_detector.update(final_elbow, cam_trunk)
                if smash_result:
                    score = smash_result['score']
                    print(f"💥 SMASH DETECTED! Suggested Score: {score}/100")
                    if score > session_best_score:
                        session_best_score   = score
                        session_best_metrics = smash_result['metrics']

            # --- E. T-Pose Calibration Completion Check ---
            if calibrating:
                elapsed_c = now - calib_start_time
                if elapsed_c >= 3.0:
                    calibrating = False
                    is_calibrated = True
                    calib_display_timer = now + 3.0
                    for key, _, _ in ARM_SEGMENTS:
                        samples = calib_samples[key]
                        if len(samples) > 0:
                            avg_v = np.mean(samples, axis=0)
                            norm_avg = np.linalg.norm(avg_v)
                            if norm_avg > 1e-6:
                                avg_v = avg_v / norm_avg
                                # In T-pose, right arm points sideways (+X or -X), left arm opposite
                                x_sign = 1.0 if "right" in key else -1.0
                                if avg_v[0] < 0 and "right" in key:
                                    x_sign = -1.0
                                elif avg_v[0] > 0 and "left" in key:
                                    x_sign = 1.0
                                target_v = np.array([x_sign, 0.0, 0.0], dtype=np.float64)
                                calib_rotations[key] = rotation_matrix_from_vectors(avg_v, target_v)
                                print(f"[EKAGRATA] Calibrated '{key}': avg={np.round(avg_v, 3)} -> target={target_v}")
                    print("[EKAGRATA] T-pose calibration complete!")

            # --- F. Compute Arm Vectors & Send to Blender ---
            # Pipeline: landmarks -> unit vectors -> axis conversion -> calibration -> smoothing (alpha 0.4)
            payload = {}

            for key, from_joint, to_joint in ARM_SEGMENTS:
                has_valid_vector = False
                if landmarks is not None and from_joint in landmarks and to_joint in landmarks:
                    lm_from = landmarks[from_joint]
                    lm_to   = landmarks[to_joint]
                    vis_from = lm_from.get('vis', 1.0)
                    vis_to   = lm_to.get('vis', 1.0)

                    # Skip landmarks with visibility < 0.5
                    if vis_from >= 0.5 and vis_to >= 0.5:
                        diff = lm_to['pos'] - lm_from['pos']
                        norm = np.linalg.norm(diff)
                        if norm > 1e-6:
                            unit_mp = diff / norm

                            # Convert MediaPipe axes -> Blender axes in ONE function
                            blender_vec = mediapipe_to_blender(unit_mp, mirror_x=mirror_x, flip_depth=flip_depth)

                            # Accumulate for T-pose calibration if active
                            if calibrating:
                                calib_samples[key].append(blender_vec)

                            # Apply T-pose calibration rotation if active
                            if is_calibrated:
                                blender_vec = calib_rotations[key] @ blender_vec
                                norm_b = np.linalg.norm(blender_vec)
                                if norm_b > 1e-6:
                                    blender_vec = blender_vec / norm_b

                            # Exponential smoothing (alpha 0.4)
                            if key not in smoothed_vectors:
                                smoothed_vectors[key] = blender_vec
                            else:
                                smoothed = SMOOTH_ALPHA * blender_vec + (1.0 - SMOOTH_ALPHA) * smoothed_vectors[key]
                                norm_s = np.linalg.norm(smoothed)
                                smoothed_vectors[key] = smoothed / norm_s if norm_s > 1e-6 else smoothed

                            has_valid_vector = True

                # If landmarks missing or low visibility, smoothed_vectors[key] retains last valid value
                if key in smoothed_vectors:
                    payload[key] = smoothed_vectors[key].tolist()

            # Send via blender_bridge every frame in JSON keys
            if payload:
                blender.send_payload(payload)

            # --- G. Overlay and Display on OpenCV Window ---
            x_str = "-mp_x" if mirror_x else "+mp_x"
            y_str = "+mp_z" if flip_depth else "-mp_z"
            z_str = "-mp_y"
            mapping_str = f"Mapping: ({x_str}, {y_str}, {z_str})"

            # Semi-transparent dark background for top HUD
            hud_height = 135 if calibrating else 100
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], hud_height), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

            # 1. Print mapping in use on screen
            cv2.putText(frame, mapping_str, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)

            # Blender connection status
            conn_txt = "Blender: Connected" if blender.connected else "Blender: Disconnected"
            conn_col = (0, 255, 0) if blender.connected else (0, 0, 255)
            cv2.putText(frame, conn_txt, (frame.shape[1] - 250, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, conn_col, 2, cv2.LINE_AA)

            # 2. Controls help text
            controls_str = "[m] Mirror X  |  [x] Flip Depth  |  [c] Calib T-pose  |  [q] Quit"
            cv2.putText(frame, controls_str, (15, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)

            # 3. Calibration UI
            if calibrating:
                rem = max(0.0, 3.0 - (now - calib_start_time))
                cv2.putText(frame, f"CALIBRATING T-POSE: {rem:.1f}s - HOLD T-POSE ARMS OUT",
                            (15, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 165, 255), 2, cv2.LINE_AA)
                bar_w = 280
                prog = int(bar_w * (1.0 - rem / 3.0))
                cv2.rectangle(frame, (15, 102), (15 + prog, 118), (0, 255, 255), -1)
                cv2.rectangle(frame, (15, 102), (15 + bar_w, 118), (255, 255, 255), 1)
            elif is_calibrated:
                status_txt = "T-Pose: Calibrated (Active)"
                cv2.putText(frame, status_txt, (15, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 1, cv2.LINE_AA)
                if now < calib_display_timer:
                    cv2.putText(frame, "Calibration saved!", (240, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, "T-Pose: Default (Press 'c' to run 3s calibration)", (15, 84),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)

            # Elbow angle & smash score overlays
            if final_elbow:
                cv2.putText(frame, f"Elbow: {final_elbow:.1f} deg", (15, frame.shape[0] - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

            if smash_detector.last_score:
                cv2.putText(frame, f"Score: {smash_detector.last_score}/100", (15, frame.shape[0] - 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow("EKAGRATA - AI Digital Twin", frame)

            # --- H. Keyboard Event Handling ---
            elapsed = time.time() - loop_start
            wait_ms = max(1, int((0.033 - elapsed) * 1000))

            key = cv2.waitKey(wait_ms) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                mirror_x = not mirror_x
                print(f"[EKAGRATA] Toggled mirror axis: mirror_x={mirror_x} -> x_axis={'-mp_x' if mirror_x else '+mp_x'}")
            elif key == ord('x'):
                flip_depth = not flip_depth
                print(f"[EKAGRATA] Toggled depth sign: flip_depth={flip_depth} -> y_axis={'+mp_z' if flip_depth else '-mp_z'}")
            elif key == ord('c'):
                calibrating = True
                calib_start_time = time.time()
                calib_samples = {k: [] for k, _, _ in ARM_SEGMENTS}
                print("[EKAGRATA] Starting 3-second T-pose calibration... Hold arms straight out horizontally!")

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        print("Cleaning up resources...")
        pose_tracker.close()
        imu.stop()
        blender.close()
        cv2.destroyAllWindows()

        # --- I. Generate Report ---
        print("Generating session report...")
        reporter = SessionReport(config)
        reporter.generate(
            athlete_name="Player 1",
            final_score=session_best_score,
            metrics=session_best_metrics,
            angle_history=angle_history,
            time_history=time_history
        )
        print("Done.")


if __name__ == "__main__":
    main()
