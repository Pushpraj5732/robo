import numpy as np

class PoseImuFusion:
    def __init__(self, alpha=0.98):
        """
        Initializes the complementary filter.
        alpha (float): Weight given to the IMU (high-frequency). (1-alpha) is given to Camera.
        """
        self.alpha = alpha
        self.fused_angle = None
        self.last_imu_timestamp = None
        self.last_imu_quat = None

    def _quat_to_euler_roll(self, qw, qx, qy, qz):
        """
        Helper to convert a quaternion to an approximate single-axis rotation (roll/extension)
        for the elbow/wrist. In a full system, this would do a proper coordinate mapping 
        from the sensor frame to the arm frame.
        """
        # Simplified: Extract roll (x-axis rotation) as a proxy for arm extension
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        return np.degrees(roll)

    def update(self, camera_angle, imu_data):
        """
        Fuses the camera-derived angle with IMU orientation data.
        
        Args:
            camera_angle (float): Angle computed from MediaPipe landmarks (degrees).
            imu_data (dict): Latest IMU quaternion data or None.
            
        Returns:
            float: The fused angle (degrees).
        """
        # Fallback to camera only if no IMU data or no previous state
        if imu_data is None or self.fused_angle is None:
            self.fused_angle = camera_angle
            if imu_data:
                self.last_imu_timestamp = imu_data.get('timestamp')
                self.last_imu_quat = imu_data
            return self.fused_angle

        # If we have a new IMU packet
        current_time = imu_data.get('timestamp')
        if current_time != self.last_imu_timestamp:
            
            # 1. Compute delta angle from IMU (gyro/quaternion integration proxy)
            # In a real rigorous filter, you would integrate gyro delta_t.
            # Here, we estimate the change in rotation angle from the quaternion difference.
            prev_roll = self._quat_to_euler_roll(self.last_imu_quat['qw'], self.last_imu_quat['qx'],
                                                 self.last_imu_quat['qy'], self.last_imu_quat['qz'])
            
            curr_roll = self._quat_to_euler_roll(imu_data['qw'], imu_data['qx'],
                                                 imu_data['qy'], imu_data['qz'])
            
            # Simple delta (assuming no gimbal lock crossing in this fast interval)
            delta_angle = curr_roll - prev_roll
            
            # 2. Complementary Filter formula:
            # Angle = alpha * (Previous Angle + Gyro_Delta) + (1 - alpha) * Camera_Angle
            predicted_angle = self.fused_angle + delta_angle
            
            self.fused_angle = self.alpha * predicted_angle + (1.0 - self.alpha) * camera_angle
            
            self.last_imu_timestamp = current_time
            self.last_imu_quat = imu_data
        else:
            # No new IMU data, just slowly pull towards camera
            self.fused_angle = self.alpha * self.fused_angle + (1.0 - self.alpha) * camera_angle

        return self.fused_angle
