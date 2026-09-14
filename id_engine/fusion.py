import sys
import numpy as np
import warnings
from .config import EngineConfig

warnings.filterwarnings('ignore')

class CausalHybridEKF:
    def __init__(self, dt=0.025, init_pos=(0.0, 0.0), init_vel=(0.0, 0.0)):
        # State: [px, py, vx, vy, bax, bay]
        self.x = np.zeros(6)
        self.x[0] = init_pos[0]
        self.x[1] = init_pos[1]
        self.x[2] = init_vel[0]
        self.x[3] = init_vel[1]
        
        self.dt = dt
        self.P = np.eye(6) * 1.0
        self.P[4:, 4:] = 0.1 # bias uncertainty
        
        # Process Noise (tuned for 40Hz)
        self.Q = np.zeros((6, 6))
        self.Q[0:2, 0:2] = np.eye(2) * (0.5 * dt**2)**2 * 1.0
        self.Q[2:4, 2:4] = np.eye(2) * (dt)**2 * 1.0
        self.Q[4:6, 4:6] = np.eye(2) * 1e-6
        
        # Measurement Models
        self.H_pos = np.zeros((2, 6))
        self.H_pos[0, 0] = 1.0
        self.H_pos[1, 1] = 1.0
        
        self.H_vel = np.zeros((2, 6))
        self.H_vel[0, 2] = 1.0
        self.H_vel[1, 3] = 1.0
        
    def predict(self, ax, ay, yaw):
        ax_b = ax - self.x[4]
        ay_b = ay - self.x[5]
        
        # ENU projection
        a_east = ax_b * np.cos(yaw - np.pi/2) + ay_b * np.cos(yaw)
        a_north = ax_b * np.sin(yaw - np.pi/2) + ay_b * np.sin(yaw)
        
        # State transition
        px = self.x[0] + self.x[2] * self.dt + 0.5 * a_east * self.dt**2
        py = self.x[1] + self.x[3] * self.dt + 0.5 * a_north * self.dt**2
        vx = self.x[2] + a_east * self.dt
        vy = self.x[3] + a_north * self.dt
        
        # Jacobian F
        F = np.eye(6)
        F[0, 2] = self.dt
        F[1, 3] = self.dt
        
        F[0, 4] = -0.5 * np.cos(yaw - np.pi/2) * self.dt**2
        F[1, 4] = -0.5 * np.sin(yaw - np.pi/2) * self.dt**2
        F[2, 4] = -np.cos(yaw - np.pi/2) * self.dt
        F[3, 4] = -np.sin(yaw - np.pi/2) * self.dt
        
        F[0, 5] = -0.5 * np.cos(yaw) * self.dt**2
        F[1, 5] = -0.5 * np.sin(yaw) * self.dt**2
        F[2, 5] = -np.cos(yaw) * self.dt
        F[3, 5] = -np.sin(yaw) * self.dt
        
        self.x[0] = px
        self.x[1] = py
        self.x[2] = vx
        self.x[3] = vy
        
        self.P = F @ self.P @ F.T + self.Q
        self.P = 0.5 * (self.P + self.P.T)
        
    def update_gnss(self, gnss_x, gnss_y, R_cov, reject_threshold=None):
        z = np.array([gnss_x, gnss_y])
        y_inv = z - self.H_pos @ self.x
        
        S = self.H_pos @ self.P @ self.H_pos.T + R_cov
        
        if reject_threshold is not None:
            nis = y_inv.T @ np.linalg.inv(S) @ y_inv
            if nis > reject_threshold:
                return False
                
        K = self.P @ self.H_pos.T @ np.linalg.inv(S)
        self.x = self.x + K @ y_inv
        self.P = (np.eye(6) - K @ self.H_pos) @ self.P
        self.P = 0.5 * (self.P + self.P.T)
        return True
        
    def update_zupt(self, R_zupt_cov=1e-4):
        z = np.array([0.0, 0.0])
        y_inv = z - self.H_vel @ self.x
        
        R = np.eye(2) * R_zupt_cov
        S = self.H_vel @ self.P @ self.H_vel.T + R
        K = self.P @ self.H_vel.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y_inv
        self.P = (np.eye(6) - K @ self.H_vel) @ self.P
        self.P = 0.5 * (self.P + self.P.T)
        
    def update_ai(self, ai_delta_v, v_history_5s, yaw, R_ai=1.0, reject_threshold=25.0):
        H_fwd = np.zeros((1, 6))
        H_fwd[0, 2] = np.cos(yaw)
        H_fwd[0, 3] = np.sin(yaw)
        
        v_fwd_history = v_history_5s[0] * np.cos(yaw) + v_history_5s[1] * np.sin(yaw)
        
        z = np.array([v_fwd_history + ai_delta_v])
        y_inv = z - H_fwd @ self.x
        
        S = H_fwd @ self.P @ H_fwd.T + np.array([[R_ai]])
        
        nis_val = y_inv.T @ np.linalg.inv(S) @ y_inv
        nis = float(np.squeeze(nis_val))
        if nis > reject_threshold:
            return False, nis
            
        K = self.P @ H_fwd.T @ np.linalg.inv(S)
        self.x = self.x + K @ y_inv
        self.P = (np.eye(6) - K @ H_fwd) @ self.P
        self.P = 0.5 * (self.P + self.P.T)
        
        return True, nis
    
class EngineEKF:
    def __init__(self, config: EngineConfig):
        self.config = config
        self.ekf = CausalHybridEKF(dt=config.dt, init_pos=(0.0, 0.0), init_vel=(0.0, 0.0))
        
    def reset(self, init_pos, init_vel):
        self.ekf = CausalHybridEKF(dt=self.config.dt, init_pos=init_pos, init_vel=init_vel)
        
    def predict(self, ax, ay, yaw):
        self.ekf.predict(ax, ay, yaw)
        
    def update_gnss(self, px, py, R_cov, reject_threshold):
        return self.ekf.update_gnss(px, py, R_cov, reject_threshold=reject_threshold)
        
    def update_ai(self, ai_delta_v, prev_vel, yaw, R_ai, reject_threshold):
        return self.ekf.update_ai(ai_delta_v, prev_vel, yaw, R_ai, reject_threshold=reject_threshold)
        
    def get_state(self):
        return self.ekf.x
        
    def get_cov(self):
        return self.ekf.P
