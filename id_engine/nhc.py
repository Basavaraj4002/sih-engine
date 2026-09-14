import sys
import numpy as np
from .config import EngineConfig
from .types import Status

def get_nhc_measurement_matrix(yaw):
    H = np.zeros((1, 6))
    H[0, 2] = -np.sin(yaw)
    H[0, 3] = np.cos(yaw)
    return H

def apply_nhc_update(ekf, yaw, R_nhc=0.1**2, reject_threshold=25.0):
    H = get_nhc_measurement_matrix(yaw)
    z = np.array([0.0]) 
    y_inv = z - H @ ekf.x
    
    S = H @ ekf.P @ H.T + np.array([[R_nhc]])
    
    nis_val = y_inv.T @ np.linalg.inv(S) @ y_inv
    nis = float(np.squeeze(nis_val))
    
    if nis > reject_threshold:
        return False, nis
        
    K = ekf.P @ H.T @ np.linalg.inv(S)
    ekf.x = ekf.x + K @ y_inv
    ekf.P = (np.eye(6) - K @ H) @ ekf.P
    ekf.P = 0.5 * (ekf.P + ekf.P.T)
    
    return True, nis
    
class NHCEngine:
    def __init__(self, config: EngineConfig):
        self.config = config
        
    def apply(self, ekf_wrapper, yaw) -> Status:
        if not self.config.use_nhc:
            return Status.DISABLED
            
        success, _ = apply_nhc_update(ekf_wrapper.ekf, yaw, self.config.R_nhc, self.config.reject_nhc)
        return Status.ACCEPTED if success else Status.REJECTED
