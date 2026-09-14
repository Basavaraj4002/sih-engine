import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["engine"] == "IDR"

def test_engine_reset():
    response = client.post("/engine/reset")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_engine_causal_workflow():
    # 1. Reset engine
    client.post("/engine/reset")
    
    # 2. Send one GNSS update to initialize
    payload_gnss = {
        "timestamp": 0.0,
        "accel": {"x": 0.0, "y": 0.0, "z": 9.81},
        "gyro": {"x": 0.0, "y": 0.0, "z": 0.0},
        "orientation": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
        "gnss": {
            "latitude": 12.9716,
            "longitude": 77.5946,
            "altitude": 900.0,
            "speed": 10.0,
            "accuracy": 5.0,
            "satellites": 12
        }
    }
    
    res = client.post("/engine/update", json=payload_gnss)
    assert res.status_code == 200
    data = res.json()
    assert data["latitude"] == 12.9716
    assert data["mode"] == "RECOVERING"
    assert data["gnss_available"] == True
    
    # 3. Send IMU updates (simulating dead reckoning between GNSS)
    for i in range(1, 4):
        payload_imu = {
            "timestamp": i * 0.1,
            "accel": {"x": 0.0, "y": 0.0, "z": 9.81},
            "gyro": {"x": 0.0, "y": 0.0, "z": 0.0},
            "orientation": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
            "gnss": None # No GNSS provided
        }
        res = client.post("/engine/update", json=payload_imu)
        assert res.status_code == 200
        data = res.json()
        assert data["gnss_available"] == False
        assert data["altitude"] == 900.0 # Altitude should be retained
        
    # 4. Simulate a long GNSS outage forcing it to drop to LOST/DR mode
    # According to our hysteresis it drops to LOST after 2 seconds
    for i in range(4, 25):
        payload_imu["timestamp"] = i * 0.1
        res = client.post("/engine/update", json=payload_imu)
        assert res.status_code == 200
        data = res.json()
        assert data["gnss_available"] == False
        assert data["altitude"] == 900.0 # Altitude should be retained
        
    # The last one should be DR
    assert res.json()["mode"] == "DR"
    assert res.json()["gnss_quality"] == "LOST"
