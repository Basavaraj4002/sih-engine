import numpy as np
from .types import EngineInput

class Simulator:
    def __init__(self, start_lat=12.9716, start_lon=77.5946, speed=15.0, heading_deg=45.0):
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.speed = speed
        self.heading = np.radians(heading_deg)
        
        self.t = 0.0
        self.dt = 0.025 # 40Hz
        
        # Current local coords
        self.x = 0.0
        self.y = 0.0
        
    def get_latlon(self):
        # 1 deg lat ~ 111km
        lat = self.start_lat + (self.y / 111139.0)
        lon = self.start_lon + (self.x / (111139.0 * np.cos(np.radians(self.start_lat))))
        return lat, lon
        
    def generate_stream(self, duration=100.0, outage_start=30.0, outage_duration=40.0):
        """Yields EngineInput sequentially"""
        steps = int(duration / self.dt)
        
        for i in range(steps):
            self.t += self.dt
            
            # Kinematics
            self.x += self.speed * np.cos(self.heading) * self.dt
            self.y += self.speed * np.sin(self.heading) * self.dt
            
            # Mock IMU (No acceleration since constant speed, just add gravity to Z)
            noise_acc = np.random.normal(0, 0.1, 3)
            ax, ay, az = noise_acc[0], noise_acc[1], noise_acc[2] + 9.81
            
            noise_gyr = np.random.normal(0, 0.01, 3)
            gx, gy, gz = noise_gyr[0], noise_gyr[1], noise_gyr[2]
            
            # Simulated True orientation
            yaw = self.heading
            
            # GNSS Data (1Hz)
            has_gnss = (i % int(1.0 / self.dt) == 0)
            in_outage = (outage_start <= self.t <= outage_start + outage_duration)
            
            gnss_lat = None
            gnss_lon = None
            gnss_acc = None
            gnss_sats = None
            gnss_speed = None
            
            if has_gnss and not in_outage:
                # Add 2m noise
                nx = self.x + np.random.normal(0, 2.0)
                ny = self.y + np.random.normal(0, 2.0)
                gnss_lat = self.start_lat + (ny / 111139.0)
                gnss_lon = self.start_lon + (nx / (111139.0 * np.cos(np.radians(self.start_lat))))
                
                gnss_acc = 5.0 + np.random.normal(0, 1.0)
                gnss_sats = 12
                gnss_speed = self.speed + np.random.normal(0, 0.5)
                
            inp = EngineInput(
                timestamp=self.t,
                accel_x=ax, accel_y=ay, accel_z=az,
                gyro_x=gx, gyro_y=gy, gyro_z=gz,
                yaw=yaw, pitch=0.0, roll=0.0,
                gnss_lat=gnss_lat, gnss_lon=gnss_lon, gnss_alt=0.0,
                gnss_speed=gnss_speed, gnss_accuracy=gnss_acc, gnss_satellites=gnss_sats
            )
            yield inp
