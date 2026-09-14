import pytest
import numpy as np
from id_engine import IDREngine, EngineConfig, EngineInput, EngineMode, GNSSQuality
from id_engine.simulator import Simulator

def test_engine_initialization():
    config = EngineConfig()
    engine = IDREngine(config)
    
    assert not engine.initialized
    
    # Send purely IMU data, should remain uninitialized
    inp = EngineInput(
        timestamp=0.0,
        accel_x=0.0, accel_y=0.0, accel_z=9.81,
        gyro_x=0.0, gyro_y=0.0, gyro_z=0.0
    )
    
    out = engine.update(inp)
    assert out.mode == EngineMode.DR.value
    assert out.gnss_available == False

def test_engine_gnss_initialization():
    config = EngineConfig()
    engine = IDREngine(config)
    
    # Send 4 GNSS data points to pass recovery hysteresis (4 required)
    for i in range(4):
        inp = EngineInput(
            timestamp=i*1.0,
            accel_x=0.0, accel_y=0.0, accel_z=9.81,
            gyro_x=0.0, gyro_y=0.0, gyro_z=0.0,
            gnss_lat=12.0, gnss_lon=77.0, gnss_accuracy=5.0
        )
        out = engine.update(inp)
        
    assert engine.initialized
    assert engine.config.origin_lat == 12.0
    assert engine.config.origin_lon == 77.0
    assert out.gnss_quality == GNSSQuality.GOOD.value
    assert out.mode == EngineMode.GNSS.value
    assert out.mode == EngineMode.GNSS.value

def test_simulator_outage_transition():
    config = EngineConfig()
    config.use_map_matching = False
    engine = IDREngine(config)
    sim = Simulator()
    
    # 5 seconds of good gnss, then outage for 5 seconds
    stream = sim.generate_stream(duration=10.0, outage_start=5.0, outage_duration=5.0)
    
    modes = []
    for inp in stream:
        out = engine.update(inp)
        if inp.timestamp > 4.0 and inp.timestamp < 6.0:
            modes.append((inp.timestamp, out.mode))
            
    # Should see GNSS switch to DR
    assert modes[0][1] == EngineMode.GNSS.value or modes[0][1] == EngineMode.HYBRID.value
    assert modes[-1][1] == EngineMode.DR.value
