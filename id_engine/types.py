from dataclasses import dataclass
from typing import Optional, List, Tuple
from enum import Enum

class EngineMode(Enum):
    GNSS = "GNSS"
    HYBRID = "HYBRID"
    DR = "DR"
    RECOVERING = "RECOVERING"

class GNSSQuality(Enum):
    GOOD = "GOOD"
    DEGRADED = "DEGRADED"
    LOST = "LOST"
    RECOVERING = "RECOVERING"
    
class Status(Enum):
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    AMBIGUOUS = "AMBIGUOUS"
    DISABLED = "DISABLED"

@dataclass
class EngineInput:
    timestamp: float
    accel_x: float
    accel_y: float
    accel_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float
    
    # Optional Orientation (if AHRS exists externally)
    yaw: Optional[float] = None
    pitch: Optional[float] = None
    roll: Optional[float] = None
    
    # Optional GNSS
    gnss_lat: Optional[float] = None
    gnss_lon: Optional[float] = None
    gnss_alt: Optional[float] = None
    gnss_speed: Optional[float] = None
    gnss_accuracy: Optional[float] = None
    gnss_satellites: Optional[int] = None

@dataclass
class EngineOutput:
    timestamp: float
    
    # Core state
    latitude: float
    longitude: float
    altitude: float
    velocity: Tuple[float, float]
    speed: float
    heading: float
    
    # Mode and status
    mode: str
    gnss_available: bool
    gnss_quality: str
    
    # Uncertainty (e.g. from EKF covariance)
    position_uncertainty: float
    velocity_uncertainty: float
    
    # Sub-system status
    map_status: str
    nhc_status: str
    
    # AI info
    ai_velocity: Optional[float] = None
    ai_delta_velocity: Optional[float] = None
    
    # Metrics
    processing_time_ms: float = 0.0
