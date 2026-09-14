from pydantic import BaseModel, Field
from typing import Optional

class Vector3(BaseModel):
    x: float
    y: float
    z: float

class Orientation(BaseModel):
    yaw: Optional[float] = None
    pitch: Optional[float] = None
    roll: Optional[float] = None

class GNSSData(BaseModel):
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude: Optional[float] = None
    speed: Optional[float] = None
    accuracy: Optional[float] = None
    satellites: Optional[int] = None

class UpdateRequest(BaseModel):
    timestamp: float
    accel: Vector3
    gyro: Vector3
    orientation: Optional[Orientation] = None
    gnss: Optional[GNSSData] = None

class UpdateResponse(BaseModel):
    timestamp: float
    latitude: float
    longitude: float
    altitude: Optional[float] = None
    velocity: Vector3
    speed: float
    heading: float
    mode: str
    gnss_available: bool
    gnss_quality: str
    position_uncertainty: float
    velocity_uncertainty: float
    ai_delta_velocity: Vector3  # We map the 1D ai_delta_v scalar to the forward axis (x) or null.
    map_status: str
    nhc_status: str
    processing_time_ms: float
