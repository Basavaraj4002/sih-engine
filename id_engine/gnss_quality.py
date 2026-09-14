from .types import GNSSQuality
from .config import EngineConfig

class CausalGNSSQualityEngine:
    def __init__(self, config: EngineConfig):
        self.config = config
        self.state = GNSSQuality.LOST
        self.last_valid_time = 0.0
        self.good_samples_in_a_row = 0
        
    def evaluate(self, timestamp: float, has_gnss: bool, accuracy: float = None) -> GNSSQuality:
        # Determine raw frame validity
        raw_valid = False
        if has_gnss:
            acc = accuracy if accuracy is not None else 10.0
            if acc < self.config.gnss_good_acc_thresh:
                raw_valid = True
                
        # Time since last valid
        time_since_last = timestamp - self.last_valid_time
        
        # State Machine Transitions
        if self.state == GNSSQuality.GOOD:
            if not raw_valid and has_gnss:
                self.state = GNSSQuality.DEGRADED
            elif time_since_last > self.config.gnss_loss_timeout_sec:
                self.state = GNSSQuality.LOST
            elif raw_valid:
                self.last_valid_time = timestamp
                
        elif self.state == GNSSQuality.DEGRADED:
            if time_since_last > self.config.gnss_loss_timeout_sec:
                self.state = GNSSQuality.LOST
            elif raw_valid:
                self.state = GNSSQuality.GOOD
                self.last_valid_time = timestamp
                
        elif self.state == GNSSQuality.LOST:
            if raw_valid:
                self.state = GNSSQuality.RECOVERING
                self.good_samples_in_a_row = 1
                self.last_valid_time = timestamp
                
        elif self.state == GNSSQuality.RECOVERING:
            if raw_valid:
                self.good_samples_in_a_row += 1
                self.last_valid_time = timestamp
                if self.good_samples_in_a_row >= self.config.gnss_recovery_samples_required:
                    self.state = GNSSQuality.GOOD
            elif time_since_last > self.config.gnss_loss_timeout_sec:
                # Failed recovery timeout
                self.state = GNSSQuality.LOST
                self.good_samples_in_a_row = 0
                
        return self.state
