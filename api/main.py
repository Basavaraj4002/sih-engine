from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import sys
import os

# Ensure the root project path is available for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from id_engine.engine import IDREngine
from id_engine.types import EngineInput
from id_engine.config import EngineConfig
from api.schemas import UpdateRequest, UpdateResponse, Vector3

app = FastAPI(title="SIH IDR Engine API", version="1.0.0")

# Enable CORS for Android App integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global persistent engine instance
config = EngineConfig()
config.use_map_matching = False # Usually requires offline map packs on the device
engine = IDREngine(config)

@app.get("/health")
def health_check():
    """Returns the basic health status of the IDR engine."""
    return {
        "status": "ok",
        "engine": "IDR",
        "model_loaded": torch.cuda.is_available() or hasattr(engine.ai, 'model')
    }

@app.post("/engine/reset")
def reset_engine():
    """Resets the IDR engine history and internal Kalman filters."""
    engine.reset()
    return {"status": "ok", "message": "IDR Engine reset successfully."}

@app.get("/engine/state")
def get_state():
    """Returns the current raw state vector of the engine."""
    return {
        "initialized": engine.initialized,
        "step_count": engine.step_count,
        "state_vector": engine.fusion.get_state().tolist() if engine.initialized else None
    }

@app.post("/engine/update", response_model=UpdateResponse)
def update_engine(req: UpdateRequest):
    """Processes a single causal sensor frame (IMU/GNSS)."""
    try:
        # Map Pydantic request to internal EngineInput
        yaw, pitch, roll = None, None, None
        if req.orientation:
            yaw = req.orientation.yaw
            pitch = req.orientation.pitch
            roll = req.orientation.roll
            
        gnss_lat, gnss_lon, gnss_alt = None, None, None
        gnss_speed, gnss_acc, gnss_sats = None, None, None
        
        if req.gnss:
            gnss_lat = req.gnss.latitude
            gnss_lon = req.gnss.longitude
            gnss_alt = req.gnss.altitude
            gnss_speed = req.gnss.speed
            gnss_acc = req.gnss.accuracy
            gnss_sats = req.gnss.satellites
            
        inp = EngineInput(
            timestamp=req.timestamp,
            accel_x=req.accel.x,
            accel_y=req.accel.y,
            accel_z=req.accel.z,
            gyro_x=req.gyro.x,
            gyro_y=req.gyro.y,
            gyro_z=req.gyro.z,
            yaw=yaw, pitch=pitch, roll=roll,
            gnss_lat=gnss_lat, gnss_lon=gnss_lon, gnss_alt=gnss_alt,
            gnss_speed=gnss_speed, gnss_accuracy=gnss_acc, gnss_satellites=gnss_sats
        )
        
        # Step the engine
        out = engine.update(inp)
        
        # Map output to API response
        ai_dv_val = out.ai_delta_velocity if out.ai_delta_velocity is not None else 0.0
        
        return UpdateResponse(
            timestamp=out.timestamp,
            latitude=out.latitude,
            longitude=out.longitude,
            altitude=out.altitude,
            velocity=Vector3(x=out.velocity[0], y=out.velocity[1], z=0.0),
            speed=out.speed,
            heading=out.heading,
            mode=out.mode,
            gnss_available=out.gnss_available,
            gnss_quality=out.gnss_quality,
            position_uncertainty=out.position_uncertainty,
            velocity_uncertainty=out.velocity_uncertainty,
            ai_delta_velocity=Vector3(x=ai_dv_val, y=0.0, z=0.0),
            map_status=out.map_status,
            nhc_status=out.nhc_status,
            processing_time_ms=out.processing_time_ms
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Engine processing failure: {str(e)}")
