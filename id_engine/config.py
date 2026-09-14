import os
from dataclasses import dataclass

@dataclass
class EngineConfig:
    # IMU / Processing
    dt: float = 0.025 # 40Hz
    ai_history_steps: int = 200 # For Custom GRU (5 seconds of 40Hz data)
    ai_update_rate: int = 4 # Perform AI update every N steps (10Hz)
    map_update_rate: int = 10 # Perform map update every N steps (4Hz)
    
    # Origin for local Cartesian projection (will be set dynamically)
    origin_lat: float = 0.0
    origin_lon: float = 0.0
    
    # Measurement Noise (R matrices equivalents)
    R_gnss_good: float = 1.0**2
    R_gnss_degraded: float = 10.0**2
    R_ai_velocity: float = 1.0**2
    R_nhc: float = 0.1**2
    R_map: float = 5.0**2
    
    # Innovation Gating (Mahalanobis chi-square reject thresholds)
    reject_gnss_good: float = 25.0
    reject_gnss_degraded: float = 50.0
    reject_ai_velocity: float = 25.0
    reject_nhc: float = 25.0
    reject_map: float = 25.0
    
    # GNSS Quality Hysteresis settings
    gnss_good_acc_thresh: float = 15.0 # meters
    gnss_loss_timeout_sec: float = 2.0
    gnss_recovery_samples_required: int = 4 # Must have 4 good samples to switch back to GNSS mode
    
    # Maps
    use_map_matching: bool = True
    map_cache_dir: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
    
    # Enable/Disable constraints
    use_nhc: bool = True
