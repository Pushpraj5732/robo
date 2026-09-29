import bpy
import socket
import json
import threading
import math
import time
from mathutils import Vector, Matrix

# ==============================================================
# EKAGRATA Real-Time Rig Receiver for Blender  (direction-vector edition)
# ==============================================================
#
# HOW TO USE
# ----------
# 1. Open your .blend file.
# 2. Fill in the CONFIGURATION block below (bone names, armature name).
# 3. In Blender's Python console (or Text Editor → Run Script), run:
#        exec(open("/path/to/ekagrata_rig_receiver.py").read())
#    OR append the script in the Text Editor and press Run Script.
#
# The script starts a TCP server that the Python side connects to.
# ==============================================================


# ==============================================================
# CONFIGURATION  ←  Edit this block to match your rig
# ==============================================================

# Armature object name in Blender's Outliner
ARMATURE_NAME = "Armature"

# Bone names for Mixamo Beta character (samba.fbx)
# If bones are not found, run the diagnostic line in Blender's Python console:
#   [(o.name,[b.name for b in o.pose.bones]) for o in bpy.data.objects if o.type=='ARMATURE']
BONES = {
    "right_upper": "mixamorig:RightArm",
    "right_fore":  "mixamorig:RightForeArm",
    "left_upper":  "mixamorig:LeftArm",
    "left_fore":   "mixamorig:LeftForeArm",
}

# Mirror flag: set True when the webcam image is horizontally mirrored
# (most built-in webcams produce a mirror image).
# When True, left and right arm data are swapped before being applied.
MIRROR_LEFT_RIGHT = True

# Startup self-test: sweeps right_upper from down (0,0,-1) to up (0,0,1) for 5 seconds
SELF_TEST = True

# DEBUG: set True to print incoming vectors and bone names (throttled to 1/sec)
DEBUG = True

# Network
HOST = '127.0.0.1'
PORT = 9999

# ==============================================================
# END OF CONFIGURATION
# ==============================================================

keep_running = True
latest_data = None     # latest parsed JSON dict
last_data_time = None  # timestamp of last received packet
_lock = threading.Lock()

# Timing and rate-limiting state
_self_test_start = None
_self_test_done = False
_stream_start_time = None
_last_debug_time = 0.0
_last_rx_debug_time = 0.0
_last_warn_time = 0.0

# Populated at startup by cache_rest_pose()
_rest_dirs = {}   # bone_key -> Vector (rest-pose head→tail, armature-local, normalised)


# --------------------------------------------------------------
# Startup: print bone names + cache rest-pose data
# --------------------------------------------------------------

def cache_rest_pose():
    """
    Prints all armature objects and bone names so you can fill in BONES/ARMATURE_NAME.
    Caches each configured bone's rest direction in armature-LOCAL space.
    """
    # -- Diagnostic: show all armatures and their bones --
    print("[EKAGRATA] === ARMATURE OBJECTS IN SCENE ===")
    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE':
            print(f"[EKAGRATA]   Armature object: '{obj.name}'")
            for b in obj.pose.bones:
                print(f"[EKAGRATA]     bone: '{b.name}'")

    obj = bpy.data.objects.get(ARMATURE_NAME)
    if not obj or obj.type != 'ARMATURE':
        print(f"[EKAGRATA] ERROR: Armature '{ARMATURE_NAME}' not found.")
        print("[EKAGRATA] Set ARMATURE_NAME to one of the names printed above.")
        return

    for key, bone_name in BONES.items():
        bone = obj.data.bones.get(bone_name)
        if bone is None:
            print(f"[EKAGRATA] WARNING: bone '{bone_name}' (key='{key}') NOT found — update BONES dict.")
            continue

        # Store rest Y axis in ARMATURE-LOCAL space.
        # bone.matrix_local is the bone's rest 4x4 matrix IN ARMATURE SPACE.
        # Its second column (col[1]) is the bone Y axis = the direction the bone points.
        rest_y = Vector(bone.matrix_local.col[1][:3]).normalized()
        _rest_dirs[key] = rest_y

        if DEBUG:
            print(f"[EKAGRATA][cache] '{key}' bone='{bone_name}' rest_y={rest_y}")

    print(f"[EKAGRATA] Rest pose cached for {len(_rest_dirs)}/{len(BONES)} bones.")


# --------------------------------------------------------------
# Per-frame update: aim each bone at its target direction
# --------------------------------------------------------------

def _aim_bone(obj, pbone, bone_key, target_dir_world: Vector, do_debug: bool = False):
    """
    Rotate *pbone* so its Y axis points in *target_dir_world* (world space).

    Uses direct pbone.matrix assignment — Blender decomposes it into the
    correct local rotation quaternion automatically, handling all the
    parent-chain and rest-pose math internally. No manual space conversion.
    """
    if bone_key not in _rest_dirs:
        if do_debug:
            print(f"[EKAGRATA] No rest dir cached for '{bone_key}' — bone missing at startup?")
        return

    rest_y_arm = _rest_dirs[bone_key]   # bone Y axis in armature space at rest

    # 1. Target direction in armature-local space
    arm_inv_rot = obj.matrix_world.inverted().to_3x3()
    target_arm = (arm_inv_rot @ target_dir_world).normalized()

    if target_arm.length < 1e-6:
        return

    # 2. Rotation from rest Y axis -> target direction (both in armature space)
    q_arm = rest_y_arm.rotation_difference(target_arm)

    # 3. Apply rotation to bone while preserving its CURRENT head position.
    #    Rotation part = q_arm.to_matrix() @ rest_mat.to_3x3()
    #    Translation = pbone.head (current pose-space head in armature space).
    #    WHY: We must NOT rotate the translation of rest_mat. Multiplying the full 4x4
    #    rest matrix rotates rest_mat's translation around the armature origin (0, 0, 0),
    #    which dislocates the bone. Furthermore, as parent bones rotate in the pose chain,
    #    the child's joint position moves. Using pbone.head preserves the bone's actual
    #    current pivot in armature space, preventing joint detachment and position drift.
    rest_mat = pbone.bone.matrix_local.copy()          # 4x4, armature space
    target_pose_mat = (q_arm.to_matrix() @ rest_mat.to_3x3()).to_4x4()
    target_pose_mat.translation = pbone.head

    # 4. Write back — Blender decomposes target_pose_mat into the bone's
    #    local quaternion (accounting for parent's current pose chain).
    pbone.rotation_mode = 'QUATERNION'
    pbone.matrix = target_pose_mat

    if do_debug:
        print(f"[EKAGRATA][aim] bone='{pbone.name}' target_arm={target_arm} q_arm={q_arm}")


def update_bones(data: dict, is_self_test: bool = False, do_debug: bool = False):
    """
    Apply incoming direction vectors to the rig.
    Processes upper-arm bones before forearm bones so parent rotation is
    already baked in when the forearm is computed.

    data keys: 'right_upper', 'right_fore', 'left_upper', 'left_fore'
    Each value: [x, y, z]  (unit vector in Blender WORLD space)
    """
    try:
        obj = bpy.data.objects.get(ARMATURE_NAME)
        if not obj or obj.type != 'ARMATURE':
            if do_debug:
                print(f"[EKAGRATA] Armature '{ARMATURE_NAME}' not found.")
            return

        pose = obj.pose

        # When the webcam is mirrored, swap left<->right (only for socket data, not self-test)
        if MIRROR_LEFT_RIGHT and not is_self_test:
            data = {
                "right_upper": data.get("left_upper"),
                "right_fore":  data.get("left_fore"),
                "left_upper":  data.get("right_upper"),
                "left_fore":   data.get("right_fore"),
            }
            data = {k: v for k, v in data.items() if v is not None}

        # Upper arms MUST be processed before forearms.
        # Flush after each upper arm so the forearm sees the correct parent matrix.
        for key in ["right_upper", "left_upper"]:
            if key not in data:
                continue
            bone_name = BONES.get(key)
            if not bone_name:
                continue
            pbone = pose.bones.get(bone_name)
            if pbone is None:
                if do_debug:
                    print(f"[EKAGRATA] Pose bone '{bone_name}' (key='{key}') not found.")
                continue
            target_dir = Vector(data[key]).normalized()
            if do_debug:
                print(f"[EKAGRATA] rotate '{key}' -> '{bone_name}' dir={[round(v,3) for v in data[key]]}")
            _aim_bone(obj, pbone, key, target_dir, do_debug=do_debug)

        # Flush dep graph so forearm bones see the updated upper-arm parent
        bpy.context.view_layer.update()

        for key in ["right_fore", "left_fore"]:
            if key not in data:
                continue
            bone_name = BONES.get(key)
            if not bone_name:
                continue
            pbone = pose.bones.get(bone_name)
            if pbone is None:
                if do_debug:
                    print(f"[EKAGRATA] Pose bone '{bone_name}' (key='{key}') not found.")
                continue
            target_dir = Vector(data[key]).normalized()
            if do_debug:
                print(f"[EKAGRATA] rotate '{key}' -> '{bone_name}' dir={[round(v,3) for v in data[key]]}")
            _aim_bone(obj, pbone, key, target_dir, do_debug=do_debug)

        # Final flush
        bpy.context.view_layer.update()

    except Exception as e:
        import traceback
        print(f"[EKAGRATA] update_bones error: {e}")
        traceback.print_exc()


# --------------------------------------------------------------
# Viewport redraw
# --------------------------------------------------------------

def force_redraw():
    """Force every 3D viewport to redraw."""
    try:
        for wm in bpy.data.window_managers:
            for window in wm.windows:
                for area in window.screen.areas:
                    if area.type == 'VIEW_3D':
                        area.tag_redraw()
    except Exception as e:
        print(f"[EKAGRATA] redraw error: {e}")


# --------------------------------------------------------------
# Timer: runs on Blender's main thread (~30 Hz)
# --------------------------------------------------------------

def timer_tick():
    global latest_data, keep_running, _self_test_done, _self_test_start
    global _last_debug_time, _last_warn_time, _stream_start_time, last_data_time

    if not keep_running:
        print("[EKAGRATA] Timer stopped.")
        return None  # Unregister

    now = time.time()

    # Rate-limit DEBUG printing to once per second
    do_debug = False
    if DEBUG and (now - _last_debug_time >= 1.0):
        do_debug = True
        _last_debug_time = now

    # ----------------------------------------------------------
    # 1. Startup self-test (5 seconds)
    # ----------------------------------------------------------
    if SELF_TEST and not _self_test_done:
        elapsed = now - _self_test_start
        if elapsed < 5.0:
            frac = elapsed / 5.0  # 0.0 -> 1.0
            # Sweep right_upper from pointing down (0, 0, -1) to pointing up (0, 0, 1)
            # Arc in Y-Z plane maintaining unit vector length
            angle = math.pi * frac
            target_dir = [0.0, math.sin(angle), -math.cos(angle)]
            test_data = {"right_upper": target_dir}
            update_bones(test_data, is_self_test=True, do_debug=do_debug)
            force_redraw()
            return 0.033
        else:
            _self_test_done = True
            _stream_start_time = now
            print("[EKAGRATA] Self-test complete (5s elapsed). Switching back to socket data.")

    # ----------------------------------------------------------
    # 2. Check for socket data arrival and warn if no data for >= 2.0s
    # ----------------------------------------------------------
    with _lock:
        data = latest_data
        t_data = last_data_time

    time_since_data = (now - t_data) if t_data is not None else (now - _stream_start_time)

    if time_since_data >= 2.0:
        if (now - _last_warn_time) >= 2.0:
            _last_warn_time = now
            print(f"[EKAGRATA] WARNING: No data arrived for {time_since_data:.1f}s")

    # ----------------------------------------------------------
    # 3. Apply socket data if available
    # ----------------------------------------------------------
    if data is not None:
        update_bones(data, is_self_test=False, do_debug=do_debug)
        force_redraw()

    return 0.033  # ~30 FPS


# --------------------------------------------------------------
# Background thread: TCP server
# --------------------------------------------------------------

def server_loop():
    global keep_running, latest_data, last_data_time, _last_rx_debug_time
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.settimeout(1.0)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        server.bind((HOST, PORT))
        server.listen(1)
        print(f"[EKAGRATA] Listening on {HOST}:{PORT}")
    except OSError as e:
        print(f"[EKAGRATA] Bind failed: {e}")
        return

    while keep_running:
        try:
            client, addr = server.accept()
            print(f"[EKAGRATA] Connected: {addr}")
            client.settimeout(1.0)
            buf = ""
            while keep_running:
                try:
                    chunk = client.recv(4096)
                    if not chunk:
                        break
                    buf += chunk.decode('utf-8')
                    while '\n' in buf:
                        line, buf = buf.split('\n', 1)
                        line = line.strip()
                        if line:
                            try:
                                parsed = json.loads(line)
                                now = time.time()
                                if DEBUG and (now - _last_rx_debug_time >= 1.0):
                                    _last_rx_debug_time = now
                                    print(f"[EKAGRATA][rx] {parsed}")
                                with _lock:
                                    latest_data = parsed
                                    last_data_time = now
                            except json.JSONDecodeError as je:
                                print(f"[EKAGRATA] JSON error: {je} | line={line!r}")
                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"[EKAGRATA] Socket error: {e}")
                    break
            client.close()
            print("[EKAGRATA] Client disconnected.")
        except socket.timeout:
            continue

    server.close()
    print("[EKAGRATA] Server stopped.")


# --------------------------------------------------------------
# Register / Unregister
# --------------------------------------------------------------

def register():
    global keep_running, _self_test_start, _self_test_done, _stream_start_time
    global _last_debug_time, _last_rx_debug_time, _last_warn_time
    global latest_data, last_data_time

    keep_running = True
    now = time.time()
    _self_test_start = now
    _self_test_done = not SELF_TEST
    _stream_start_time = now
    _last_debug_time = 0.0
    _last_rx_debug_time = 0.0
    _last_warn_time = 0.0
    latest_data = None
    last_data_time = None

    # Unregister stale timer if it exists from a previous run
    if bpy.app.timers.is_registered(timer_tick):
        bpy.app.timers.unregister(timer_tick)

    # ----------------------------------------------------------------
    # Animation cleanup: detach action, mute NLA tracks, cancel playback
    # ----------------------------------------------------------------
    obj = bpy.data.objects.get(ARMATURE_NAME)
    if obj and obj.animation_data:
        if obj.animation_data.action:
            print(f"[EKAGRATA] Detaching action '{obj.animation_data.action.name}'")
            obj.animation_data.action = None
        if obj.animation_data.nla_tracks:
            count = len(obj.animation_data.nla_tracks)
            for track in obj.animation_data.nla_tracks:
                track.mute = True
            print(f"[EKAGRATA] Muted {count} NLA track(s).")

    try:
        if bpy.context.screen and bpy.context.screen.is_animation_playing:
            bpy.ops.screen.animation_cancel(restore_frame=False)
            print("[EKAGRATA] Stopped timeline playback.")
    except Exception:
        pass

    # Cache rest pose before starting the timer
    cache_rest_pose()

    bpy.app.timers.register(timer_tick, first_interval=0.1, persistent=True)
    print(f"[EKAGRATA] Timer registered: {bpy.app.timers.is_registered(timer_tick)}")

    t = threading.Thread(target=server_loop, daemon=True)
    t.start()
    print("[EKAGRATA] Receiver started.")


def unregister():
    global keep_running
    keep_running = False
    if bpy.app.timers.is_registered(timer_tick):
        bpy.app.timers.unregister(timer_tick)
    print("[EKAGRATA] Receiver stopped.")


if __name__ == "__main__":
    unregister()
    register()
