import numpy as np
from dataclasses import dataclass

# WGS84 Constants
R_EARTH = 6378137.0

def latlon_to_xy(lat: float, lon: float, origin_lat: float, origin_lon: float) -> tuple[float, float]:
    """Convert Lat/Lon to local cartesian (meters) using equirectangular projection"""
    dlat = np.radians(lat - origin_lat)
    dlon = np.radians(lon - origin_lon)
    x = dlon * np.cos(np.radians(origin_lat)) * R_EARTH
    y = dlat * R_EARTH
    return x, y

def xy_to_latlon(x: float, y: float, origin_lat: float, origin_lon: float) -> tuple[float, float]:
    """Convert local cartesian (meters) back to Lat/Lon"""
    dlat = y / R_EARTH
    dlon = x / (np.cos(np.radians(origin_lat)) * R_EARTH)
    lat = origin_lat + np.degrees(dlat)
    lon = origin_lon + np.degrees(dlon)
    return lat, lon

class MadgwickAHRS:
    """Basic Madgwick filter for standalone 6DOF orientation if missing from input"""
    def __init__(self, sampleperiod=0.025, beta=0.1):
        self.sampleperiod = sampleperiod
        self.beta = beta
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        
    def update(self, gyro, accel):
        q = self.q
        
        # Normalize accelerometer
        a_norm = np.linalg.norm(accel)
        if a_norm == 0:
            return
        a = accel / a_norm
        
        # Gradient descent step
        f = np.array([
            2.0*(q[1]*q[3] - q[0]*q[2]) - a[0],
            2.0*(q[0]*q[1] + q[2]*q[3]) - a[1],
            2.0*(0.5 - q[1]**2 - q[2]**2) - a[2]
        ])
        
        J = np.array([
            [-2.0*q[2],  2.0*q[3], -2.0*q[0],  2.0*q[1]],
            [ 2.0*q[1],  2.0*q[0],  2.0*q[3],  2.0*q[2]],
            [ 0.0,      -4.0*q[1], -4.0*q[2],  0.0     ]
        ])
        
        step = J.T @ f
        step_norm = np.linalg.norm(step)
        if step_norm > 0:
            step /= step_norm
            
        # Compute rate of change of quaternion
        qDot = 0.5 * np.array([
            -q[1]*gyro[0] - q[2]*gyro[1] - q[3]*gyro[2],
             q[0]*gyro[0] + q[2]*gyro[2] - q[3]*gyro[1],
             q[0]*gyro[1] - q[1]*gyro[2] + q[3]*gyro[0],
             q[0]*gyro[2] + q[1]*gyro[1] - q[2]*gyro[0]
        ]) - self.beta * step
        
        # Integrate to yield quaternion
        q = q + qDot * self.sampleperiod
        self.q = q / np.linalg.norm(q)
        
    def get_euler(self):
        # returns roll, pitch, yaw
        q = self.q
        roll = np.arctan2(2.0*(q[0]*q[1] + q[2]*q[3]), 1.0 - 2.0*(q[1]**2 + q[2]**2))
        pitch = np.arcsin(2.0*(q[0]*q[2] - q[3]*q[1]))
        yaw = np.arctan2(2.0*(q[0]*q[3] + q[1]*q[2]), 1.0 - 2.0*(q[2]**2 + q[3]**2))
        return roll, pitch, yaw
