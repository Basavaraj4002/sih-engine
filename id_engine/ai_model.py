import os
import torch
import torch.nn as nn
import numpy as np

class DeltaVGRU(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU()
        )
        
        self.gru = nn.GRU(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        x_proj = self.input_proj(x)
        out, _ = self.gru(x_proj)
        last_out = out[:, -1, :]
        delta_v = self.head(last_out)
        return delta_v

class GRUInference:
    def __init__(self):
        if os.environ.get("FORCE_CPU") == "1":
            self.device = torch.device('cpu')
            print("FORCE_CPU enabled. Running inference on CPU.")
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        self.model = DeltaVGRU(input_size=6, hidden_size=128, num_layers=2, dropout=0.0).to(self.device)
        
        # Default to relative path from this script
        default_chkpt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
        chkpt_dir = os.environ.get("IDR_MODEL_DIR", default_chkpt)
        
        weights_path = os.path.join(chkpt_dir, "best_deltav_gru.pt")
        stats_path = os.path.join(chkpt_dir, "deltav_norm_stats.pt")
        
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()
        
        stats = torch.load(stats_path, map_location=self.device)
        self.mean = stats['mean'].to(self.device)
        self.std = stats['std'].to(self.device)
        
    def predict(self, window_6d):
        if len(window_6d) < 200:
            return 0.0 
            
        x_tensor = torch.FloatTensor(window_6d).to(self.device)
        x = (x_tensor - self.mean) / (self.std + 1e-8)
        x = x.unsqueeze(0)
        
        with torch.no_grad():
            pred = self.model(x).item()
            
        return pred

    def predict_batch(self, ai_imu_full):
        """
        Fast batched prediction for an entire trajectory.
        ai_imu_full: (N, 6) array
        Returns: (N,) array of delta_v predictions (0 for first 200)
        """
        N = len(ai_imu_full)
        out = np.zeros(N)
        if N < 200:
            return out
            
        # Create rolling windows (every 4th index starting from 200)
        indices = np.arange(200, N, 4)
        if len(indices) == 0:
            return out
            
        windows = np.zeros((len(indices), 200, 6), dtype=np.float32)
        for j, i in enumerate(indices):
            windows[j] = ai_imu_full[i-200:i]
            
        windows_tensor = torch.FloatTensor(windows).to(self.device)
        x = (windows_tensor - self.mean) / (self.std + 1e-8)
        
        # Batch inference
        with torch.no_grad():
            # Process in chunks to avoid OOM on huge files
            chunk_size = 10000
            preds = []
            for j in range(0, len(x), chunk_size):
                chunk = x[j:j+chunk_size]
                preds.append(self.model(chunk).cpu().numpy().squeeze())
            if len(preds) > 0:
                preds = np.concatenate(preds)
            else:
                preds = np.array([])
                
        for j, i in enumerate(indices):
            out[i] = preds[j]
            
        return out
