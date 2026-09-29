import numpy as np

def calculate_angle_3d(p1, p2, p3):
    """
    Calculates the 3D angle formed by three points (in degrees).
    p2 is the vertex.
    
    Args:
        p1, p2, p3: numpy arrays of shape (3,) representing [x, y, z].
        
    Returns:
        float: Angle in degrees between p1-p2 and p3-p2, ranging from 0 to 180.
    """
    # Vectors
    v1 = p1 - p2
    v2 = p3 - p2
    
    # Normalize vectors
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    
    if v1_norm == 0 or v2_norm == 0:
        return 0.0
        
    v1_u = v1 / v1_norm
    v2_u = v2 / v2_norm
    
    # Dot product
    dot_product = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
    
    # Angle in radians, then convert to degrees
    angle_rad = np.arccos(dot_product)
    return np.degrees(angle_rad)

def get_elbow_angle(landmarks):
    """
    Returns the angle of the right elbow.
    """
    if 'right_shoulder' not in landmarks or 'right_elbow' not in landmarks or 'right_wrist' not in landmarks:
        return None
    return calculate_angle_3d(
        landmarks['right_shoulder'],
        landmarks['right_elbow'],
        landmarks['right_wrist']
    )

def get_shoulder_angle(landmarks):
    """
    Returns the angle of the right shoulder relative to the hip.
    """
    if 'right_hip' not in landmarks or 'right_shoulder' not in landmarks or 'right_elbow' not in landmarks:
        return None
    return calculate_angle_3d(
        landmarks['right_hip'],
        landmarks['right_shoulder'],
        landmarks['right_elbow']
    )

def get_trunk_rotation(landmarks):
    """
    Returns a simplified trunk rotation angle by comparing the shoulder line to the horizontal plane.
    """
    if 'left_shoulder' not in landmarks or 'right_shoulder' not in landmarks:
        return None
        
    left_sh = landmarks['left_shoulder']
    right_sh = landmarks['right_shoulder']
    
    # Vector across shoulders
    shoulder_vec = left_sh - right_sh
    
    # Project onto XZ plane (ignore Y/height) to see rotation relative to camera depth
    shoulder_vec_xz = np.array([shoulder_vec[0], 0, shoulder_vec[2]])
    norm = np.linalg.norm(shoulder_vec_xz)
    
    if norm == 0:
        return 0.0
        
    shoulder_vec_xz = shoulder_vec_xz / norm
    
    # Angle relative to X axis
    ref_vec = np.array([1, 0, 0])
    dot = np.clip(np.dot(shoulder_vec_xz, ref_vec), -1.0, 1.0)
    
    return np.degrees(np.arccos(dot))

if __name__ == "__main__":
    # Simple unit tests
    p1 = np.array([1, 0, 0])
    p2 = np.array([0, 0, 0])
    p3 = np.array([0, 1, 0])
    print(f"90 Degree Test: {calculate_angle_3d(p1, p2, p3)} == 90.0")
    
    p3_180 = np.array([-1, 0, 0])
    print(f"180 Degree Test: {calculate_angle_3d(p1, p2, p3_180)} == 180.0")
