import time
from collections import deque
import numpy as np

class SmashDetector:
    def __init__(self, config):
        """
        config (dict): scoring configuration from yaml
        """
        self.config = config.get('scoring', {})
        self.vel_threshold = self.config.get('velocity_threshold', 400.0)
        self.weights = self.config.get('weights', {})
        self.ideal_elbow = self.config.get('ideal_elbow_extension', 170.0)
        self.ideal_trunk = self.config.get('ideal_trunk_rotation', 45.0)
        
        self.angle_history = deque(maxlen=10)
        self.time_history = deque(maxlen=10)
        
        self.smash_detected = False
        self.last_score = None
        self.last_metrics = None
        
        self.cooldown = 0
        
    def update(self, fused_elbow_angle, trunk_angle):
        """
        Checks for a smash impact and calculates a technique score.
        Call this every frame.
        
        Args:
            fused_elbow_angle (float): The current fused elbow angle.
            trunk_angle (float): The current trunk rotation angle.
            
        Returns:
            dict: The smash score and metrics if an impact was just detected, else None.
        """
        current_time = time.time()
        
        if fused_elbow_angle is None or trunk_angle is None:
            return None
            
        self.angle_history.append(fused_elbow_angle)
        self.time_history.append(current_time)
        
        if self.cooldown > 0:
            self.cooldown -= 1
            return None
            
        if len(self.angle_history) < 2:
            return None
            
        # Calculate angular velocity (degrees per second)
        delta_angle = self.angle_history[-1] - self.angle_history[-2]
        delta_time = self.time_history[-1] - self.time_history[-2]
        
        if delta_time == 0:
            return None
            
        angular_vel = abs(delta_angle / delta_time)
        
        # Detect Impact (Peak velocity threshold crossed)
        # In a real system, you'd look for a local maximum, but a simple threshold works for demo.
        if angular_vel > self.vel_threshold:
            score, metrics = self._calculate_score(fused_elbow_angle, angular_vel, trunk_angle)
            self.last_score = score
            self.last_metrics = metrics
            self.smash_detected = True
            
            # Cooldown to avoid multiple triggers for one smash
            self.cooldown = 30 
            
            return {
                'score': score,
                'metrics': metrics,
                'is_suggested': True # Explicitly flag as suggested score
            }
            
        return None

    def _calculate_score(self, elbow_angle, wrist_speed, trunk_angle):
        """
        Calculates a 0-100 technique score based on ideal angles.
        NOTE: This is a suggested score - validate with a coach.
        """
        # 1. Elbow Extension Score (higher is better, up to ideal)
        elbow_error = abs(self.ideal_elbow - elbow_angle)
        elbow_score = max(0, 100 - (elbow_error * 1.5)) 
        
        # 2. Speed Score (cap at 100)
        speed_score = min(100, (wrist_speed / 1000.0) * 100) # Assume 1000 deg/s is a 'perfect' speed
        
        # 3. Trunk Rotation Score
        trunk_error = abs(self.ideal_trunk - trunk_angle)
        trunk_score = max(0, 100 - (trunk_error * 2.0))
        
        w_e = self.weights.get('elbow_extension', 0.3)
        w_s = self.weights.get('wrist_speed', 0.5)
        w_t = self.weights.get('trunk_rotation', 0.2)
        
        final_score = (elbow_score * w_e) + (speed_score * w_s) + (trunk_score * w_t)
        
        metrics = {
            'elbow_score': elbow_score,
            'speed_score': speed_score,
            'trunk_score': trunk_score,
            'raw_elbow': elbow_angle,
            'raw_speed': wrist_speed,
            'raw_trunk': trunk_angle
        }
        
        return round(final_score), metrics
