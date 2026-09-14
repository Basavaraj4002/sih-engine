import numpy as np
import sys
from .types import EngineInput

def get_rotation_matrix_from_vectors(vec1, vec2):
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    
    if s < 1e-6:
        if c > 0:
            return np.eye(3)
        else:
            return np.diag([1, -1, -1])
            
    vX = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
    
    R = np.eye(3) + vX + np.dot(vX, vX) * ((1 - c) / (s ** 2))
    return R

def align_imu_to_gravity(accel, gyro, gravity):
    N = accel.shape[0]
    aligned_accel = np.zeros_like(accel)
    aligned_gyro = np.zeros_like(gyro)
    
    canonical_z = np.array([0, 0, 1.0])
    
    for i in range(N):
        g = gravity[i]
        if np.linalg.norm(g) < 1e-4 or np.any(np.isnan(g)):
            aligned_accel[i] = accel[i]
            aligned_gyro[i] = gyro[i]
            continue
            
        R = get_rotation_matrix_from_vectors(g, canonical_z)
        aligned_accel[i] = (R @ accel[i].reshape(3, 1)).reshape(3)
        aligned_gyro[i] = (R @ gyro[i].reshape(3, 1)).reshape(3)
        
    return aligned_accel, aligned_gyro

class IMUPreprocessor:
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.smoothed_g = None
        
    def process(self, accel: np.ndarray, gyro: np.ndarray, dt: float = 0.025) -> np.ndarray:
        """
        Takes raw [ax, ay, az] and [gx, gy, gz] and normalizes it to gravity frame.
        Uses a simple low-pass filter on the accelerometer to estimate gravity 
        (since true gravity isn't provided directly in streaming EngineInput).
        """
        if self.smoothed_g is None:
            self.smoothed_g = np.copy(accel)
            
        # Update gravity estimate
        self.smoothed_g = self.alpha * accel + (1 - self.alpha) * self.smoothed_g
        
        # Align
        a_align, g_align = align_imu_to_gravity(
            accel.reshape(1, 3), 
            gyro.reshape(1, 3), 
            self.smoothed_g.reshape(1, 3)
        )
        
        # Returns [ax, ay, az, gx, gy, gz] aligned
        return np.concatenate([a_align, g_align], axis=1).flatten()
