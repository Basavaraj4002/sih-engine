import numpy as np
import os
from .config import EngineConfig
from .types import Status
import pickle

def distance_point_to_segment(px, py, sx1, sy1, sx2, sy2):
    # Returns (distance, proj_x, proj_y)
    l2 = (sx2 - sx1)**2 + (sy2 - sy1)**2
    if l2 == 0:
        return np.hypot(px - sx1, py - sy1), sx1, sy1
    t = max(0, min(1, ((px - sx1) * (sx2 - sx1) + (py - sy1) * (sy2 - sy1)) / l2))
    proj_x = sx1 + t * (sx2 - sx1)
    proj_y = sy1 + t * (sy2 - sy1)
    return np.hypot(px - proj_x, py - proj_y), proj_x, proj_y

def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

class CausalMapMatcher:
    def __init__(self, map_file):
        self.segments = []
        self.kdtree = None
        self.points = []
        self.point_to_segment = []
        
        if os.path.exists(map_file):
            with open(map_file, 'rb') as f:
                data = pickle.load(f)
                self.segments = data['segments']
                self.kdtree = data['kdtree']
                self.points = data['points']
                self.point_to_segment = data['point_to_segment']
                
    def match(self, px, py, yaw, P_cov, base_radius=15.0):
        if self.kdtree is None:
            return None
            
        # Scale search radius by position uncertainty
        sigma_pos = np.sqrt(P_cov[0,0] + P_cov[1,1])
        search_radius = min(50.0, max(base_radius, 3.0 * sigma_pos))
        
        # KDTree search
        idx = self.kdtree.query_ball_point((px, py), search_radius)
        if not idx:
            return None
            
        # Get unique segments
        candidate_seg_ids = list(set([self.point_to_segment[i] for i in idx]))
        
        scored_candidates = []
        for sid in candidate_seg_ids:
            seg = self.segments[sid]
            x1, y1 = seg['start']
            x2, y2 = seg['end']
            
            # calculate geometry
            dist, proj_x, proj_y = distance_point_to_segment(px, py, x1, y1, x2, y2)
            
            # calculate road heading
            road_heading = np.arctan2(y2 - y1, x2 - x1)
            
            # calculate heading diff
            heading_diff = np.abs(wrap_angle(road_heading - yaw))
            
            # Handle two-way roads (can travel opposite direction)
            if not seg['oneway']:
                heading_diff = min(heading_diff, np.abs(wrap_angle(road_heading + np.pi - yaw)))
                
            # Discard if heading is completely wrong (> 45 deg)
            if heading_diff > np.pi/4:
                continue
                
            # Score function: distance + heading penalty
            # Weighting: 1 meter distance ~ 0.1 rad heading difference (~5.7 deg)
            score = dist + 10.0 * heading_diff
            
            scored_candidates.append({
                'score': score,
                'dist': dist,
                'proj_x': proj_x,
                'proj_y': proj_y,
                'seg_id': sid,
                'heading_diff': heading_diff
            })
            
        if not scored_candidates:
            return None
            
        scored_candidates.sort(key=lambda x: x['score'])
        
        # Ambiguity check
        if len(scored_candidates) > 1:
            best_score = scored_candidates[0]['score']
            second_best_score = scored_candidates[1]['score']
            if (second_best_score - best_score) < 2.0:
                # Too ambiguous, reject
                return None
                
        return scored_candidates[0]

class MapMatchingEngine:
    def __init__(self, config: EngineConfig):
        self.config = config
        self.matcher = None
        self.session_loaded = False
        
    def load_map(self, session_name: str, split: str = "test"):
        if not self.config.use_map_matching:
            return
            
        map_file = os.path.join(self.config.map_cache_dir, split, f"{session_name}_map.pkl")
        if os.path.exists(map_file):
            self.matcher = CausalMapMatcher(map_file)
            self.session_loaded = True
        else:
            print(f"Map engine warning: could not find map at {map_file}")
            
    def apply(self, ekf_wrapper, yaw) -> Status:
        if not self.config.use_map_matching or not self.session_loaded or self.matcher is None:
            return Status.DISABLED
            
        # Match using KDTree
        px, py = ekf_wrapper.ekf.x[0], ekf_wrapper.ekf.x[1]
        P_cov = ekf_wrapper.ekf.P
        
        res = self.matcher.match(px, py, yaw, P_cov)
        
        if res is None:
            return Status.AMBIGUOUS # Or simply didn't find a match
            
        # Construct the pseudo-measurement
        z = np.array([res['proj_x'], res['proj_y']])
        y_inv = z - ekf_wrapper.ekf.H_pos @ ekf_wrapper.ekf.x
        
        S = ekf_wrapper.ekf.H_pos @ ekf_wrapper.ekf.P @ ekf_wrapper.ekf.H_pos.T + np.eye(2) * self.config.R_map
        nis_val = y_inv.T @ np.linalg.inv(S) @ y_inv
        nis = float(np.squeeze(nis_val))
        
        if nis > self.config.reject_map:
            return Status.REJECTED
            
        # Apply the correction
        K = ekf_wrapper.ekf.P @ ekf_wrapper.ekf.H_pos.T @ np.linalg.inv(S)
        correction = K @ y_inv
        ekf_wrapper.ekf.x = ekf_wrapper.ekf.x + correction
        ekf_wrapper.ekf.P = (np.eye(6) - K @ ekf_wrapper.ekf.H_pos) @ ekf_wrapper.ekf.P
        ekf_wrapper.ekf.P = 0.5 * (ekf_wrapper.ekf.P + ekf_wrapper.ekf.P.T)
        
        return Status.ACCEPTED
