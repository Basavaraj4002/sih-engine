import time
import numpy as np
from collections import deque
from .types import EngineInput, EngineOutput, EngineMode, GNSSQuality, Status
from .config import EngineConfig
from .preprocessing import IMUPreprocessor
from .gnss_quality import CausalGNSSQualityEngine
from .ai_velocity import AIVelocityEngine
from .fusion import EngineEKF
from .nhc import NHCEngine
from .map_matching import MapMatchingEngine
from .utils import latlon_to_xy, xy_to_latlon, MadgwickAHRS

class IDREngine:
    def __init__(self, config: EngineConfig = None):
        self.config = config if config else EngineConfig()
        
        # Sub-systems
        self.preprocessor = IMUPreprocessor(alpha=0.05)
        self.gnss_quality = CausalGNSSQualityEngine(self.config)
        self.ai = AIVelocityEngine(self.config)
        self.fusion = EngineEKF(self.config)
        self.nhc = NHCEngine(self.config)
        self.map_matching = MapMatchingEngine(self.config)
        self.ahrs = MadgwickAHRS()
        
        self.reset()
        
    def reset(self, initial_lat: float = None, initial_lon: float = None):
        self.initialized = False
        
        # History buffers
        self.imu_history = deque(maxlen=self.config.ai_history_steps)
        self.v_history = deque(maxlen=self.config.ai_history_steps)
        
        self.step_count = 0
        self.last_gnss_time = 0.0
        
        if initial_lat is not None and initial_lon is not None:
            self.config.origin_lat = initial_lat
            self.config.origin_lon = initial_lon
            self.fusion.reset(init_pos=(0.0, 0.0), init_vel=(0.0, 0.0))
            self.initialized = True
            
        self.current_yaw = 0.0
        self.last_altitude = None
            
    def load_map(self, session_name: str, split: str = "test"):
        self.map_matching.load_map(session_name, split)
        
    def update(self, input_data: EngineInput) -> EngineOutput:
        t0 = time.perf_counter()
        self.step_count += 1
        
        # 1. Orientation
        if input_data.yaw is not None:
            self.current_yaw = input_data.yaw
        else:
            acc = np.array([input_data.accel_x, input_data.accel_y, input_data.accel_z])
            gyr = np.array([input_data.gyro_x, input_data.gyro_y, input_data.gyro_z])
            self.ahrs.update(gyr, acc)
            _, _, self.current_yaw = self.ahrs.get_euler()
            
        # 2. Preprocessing
        accel = np.array([input_data.accel_x, input_data.accel_y, input_data.accel_z])
        gyro = np.array([input_data.gyro_x, input_data.gyro_y, input_data.gyro_z])
        ai_frame = self.preprocessor.process(accel, gyro)
        self.imu_history.append(ai_frame)
        
        # Determine initialization from first GNSS
        has_gnss = input_data.gnss_lat is not None and input_data.gnss_lon is not None
        if not self.initialized and has_gnss:
            self.config.origin_lat = input_data.gnss_lat
            self.config.origin_lon = input_data.gnss_lon
            self.fusion.reset(init_pos=(0.0, 0.0), init_vel=(0.0, 0.0))
            self.initialized = True
            
        if not self.initialized:
            # Cannot do anything meaningful yet
            return self._build_output(input_data, EngineMode.DR, GNSSQuality.LOST, Status.DISABLED, Status.DISABLED, 0.0, 0.0)
            
        # 3. Predict Step (INS physics)
        self.fusion.predict(ai_frame[0], ai_frame[1], self.current_yaw)
        self.v_history.append(np.copy(self.fusion.ekf.x[2:4]))
        
        # 4. GNSS Quality & Update
        gnss_q = self.gnss_quality.evaluate(input_data.timestamp, has_gnss, input_data.gnss_accuracy)
        
        gnss_available = (gnss_q in [GNSSQuality.GOOD, GNSSQuality.DEGRADED, GNSSQuality.RECOVERING])
        mode = EngineMode.HYBRID
        
        if gnss_available and has_gnss:
            gx, gy = latlon_to_xy(input_data.gnss_lat, input_data.gnss_lon, self.config.origin_lat, self.config.origin_lon)
            
            R_val = self.config.R_gnss_good if gnss_q == GNSSQuality.GOOD else self.config.R_gnss_degraded
            if gnss_q == GNSSQuality.RECOVERING:
                R_val = self.config.R_gnss_good * 5.0 # Smooth recovery
                mode = EngineMode.RECOVERING
                
            reject_thresh = self.config.reject_gnss_good
            if gnss_q == GNSSQuality.DEGRADED:
                reject_thresh = self.config.reject_gnss_degraded
            elif gnss_q == GNSSQuality.RECOVERING:
                reject_thresh = 100.0 # Looser for recovery snap
                
            self.fusion.update_gnss(gx, gy, np.eye(2)*R_val, reject_threshold=reject_thresh)
            mode = EngineMode.GNSS if gnss_q == GNSSQuality.GOOD else mode
        else:
            mode = EngineMode.DR
            
        # 5. AI Update
        ai_dv = 0.0
        ai_speed = 0.0
        if len(self.imu_history) == self.config.ai_history_steps:
            if self.step_count % self.config.ai_update_rate == 0:
                imu_buf = np.array(self.imu_history)
                # Expand dims for model (batch=1, seq=200, features=6)
                imu_batch = np.expand_dims(imu_buf, axis=0)
                ai_dv_array = self.ai.predict(imu_batch) # Returns shape (1,) or scalar
                ai_dv = float(np.squeeze(ai_dv_array))
                
                # Apply AI pseudo-measurement
                hist_vel = self.v_history[0] if len(self.v_history) > 0 else np.array([0.0, 0.0])
                self.fusion.update_ai(ai_dv, hist_vel, self.current_yaw, self.config.R_ai_velocity, self.config.reject_ai_velocity)
                
                v_curr = self.fusion.ekf.x[2:4]
                ai_speed = np.linalg.norm(hist_vel) + ai_dv
                
        # 6. NHC Update (Only during outages)
        nhc_status = Status.DISABLED
        if mode == EngineMode.DR:
            nhc_status = self.nhc.apply(self.fusion, self.current_yaw)
            
        # 7. Map Matching Update (Only during outages)
        map_status = Status.DISABLED
        if mode == EngineMode.DR and self.step_count % self.config.map_update_rate == 0:
            map_status = self.map_matching.apply(self.fusion, self.current_yaw)
            
        # 8. Build Output
        t1 = time.perf_counter()
        dt_ms = (t1 - t0) * 1000.0
        return self._build_output(input_data, mode, gnss_q, nhc_status, map_status, ai_dv, dt_ms, ai_speed)
        
    def _build_output(self, inp: EngineInput, mode: EngineMode, gnss_q: GNSSQuality, nhc_status: Status, map_status: Status, ai_dv: float, dt_ms: float, ai_speed: float = 0.0) -> EngineOutput:
        x = self.fusion.get_state()
        P = self.fusion.get_cov()
        
        px, py = x[0], x[1]
        vx, vy = x[2], x[3]
        
        lat, lon = xy_to_latlon(px, py, self.config.origin_lat, self.config.origin_lon)
        speed = np.hypot(vx, vy)
        
        pos_u = float(np.sqrt(P[0,0] + P[1,1]))
        vel_u = float(np.sqrt(P[2,2] + P[3,3]))
        
        if inp.gnss_alt is not None:
            self.last_altitude = inp.gnss_alt
            
        return EngineOutput(
            timestamp=inp.timestamp,
            latitude=lat,
            longitude=lon,
            altitude=self.last_altitude,
            velocity=(vx, vy),
            speed=speed,
            heading=self.current_yaw,
            mode=mode.value,
            gnss_available=(inp.gnss_lat is not None),
            gnss_quality=gnss_q.value,
            position_uncertainty=pos_u,
            velocity_uncertainty=vel_u,
            map_status=map_status.value,
            nhc_status=nhc_status.value,
            ai_velocity=ai_speed,
            ai_delta_velocity=ai_dv,
            processing_time_ms=dt_ms
        )
