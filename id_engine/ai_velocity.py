import sys
import numpy as np
import torch
from .config import EngineConfig

try:
    from .ai_model import GRUInference
except ImportError:
    # Dummy fallback if path breaks
    class GRUInference:
        def predict(self, imu_seq): return 0.0
        
class AIVelocityEngine:
    def __init__(self, config: EngineConfig):
        self.config = config
        
        import os
        if os.environ.get("FORCE_CPU") == "1":
            print("FORCE_CPU enabled. Bypassing CUDA check.")
            self.device = torch.device("cpu")
        elif not torch.cuda.is_available():
            print("WARNING: CUDA is not available. Falling back to CPU as per previous agreement (Engine requested GPU but environment is CPU-only).")
            self.device = torch.device("cpu")
        else:
            print("AI Engine: CUDA is available. Loading GRU...")
            print(f"Device: {torch.cuda.get_device_name(0)}")
            self.device = torch.device("cuda")
        
        # Enable inference mode globally
        torch.inference_mode()(True)
        self.model = GRUInference() # It automatically uses cuda inside its own code
        
    def predict(self, imu_buffer: np.ndarray) -> float:
        """
        imu_buffer: (seq_len, 6)
        Returns the delta velocity for the step.
        """
        if len(imu_buffer) < self.config.ai_history_steps:
            return 0.0 # Return 0 delta-v if history is incomplete
            
        return self.model.predict(imu_buffer)
