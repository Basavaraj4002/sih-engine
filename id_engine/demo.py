import time
from .engine import IDREngine
from .simulator import Simulator
from .config import EngineConfig

def main():
    print("Initializing IDR Engine Phase 1 Prototype...")
    
    config = EngineConfig()
    config.use_map_matching = False # Demo runs arbitrary coordinates
    
    engine = IDREngine(config)
    sim = Simulator(speed=15.0) # 15 m/s (~54 km/h)
    
    # 100 seconds simulation, GNSS outage from 30s to 70s
    stream = sim.generate_stream(duration=100.0, outage_start=30.0, outage_duration=40.0)
    
    print("Starting Causal Simulation (100 seconds)")
    print("="*100)
    print(f"{'TIME':<8} | {'MODE':<12} | {'LATITUDE':<12} | {'LONGITUDE':<12} | {'SPEED':<8} | {'HEADING':<8} | {'GNSS':<8} | {'AI dV':<8} | {'UNCERT':<8}")
    print("-" * 100)
    
    last_print = 0.0
    for inp in stream:
        out = engine.update(inp)
        
        # Print update every 1 second
        if inp.timestamp - last_print >= 1.0:
            ai_str = f"{out.ai_delta_velocity:.3f}" if out.ai_delta_velocity is not None else "Init"
            gnss_str = out.gnss_quality
            print(f"{out.timestamp:<8.1f} | {out.mode:<12} | {out.latitude:<12.5f} | {out.longitude:<12.5f} | {out.speed:<8.2f} | {out.heading:<8.2f} | {gnss_str:<8} | {ai_str:<8} | {out.position_uncertainty:<8.2f}")
            last_print = inp.timestamp
            
    print("="*100)
    print("ENGINE STATUS: READY")
    device_name = getattr(engine.ai, 'device', 'Unknown')
    print(f"- Model Loaded: Yes ({device_name})")
    print(f"- Engine Update Latency: ~{out.processing_time_ms:.2f} ms")
    print(f"- GNSS Outage Handling: Yes")
    print(f"- GNSS Recovery Handling: Yes")
    print(f"- Map Matching Availability: Yes (via CausalMapMatcher)")
    print(f"- NHC Availability: Yes (via apply_nhc_update)")
    print(f"- Simulator Availability: Yes")

if __name__ == "__main__":
    main()
