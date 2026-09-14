# SIH Intelligent Dead Reckoning (IDR) Engine Architecture

## 1. Overview
The IDR Engine is a Phase-1 prototype for real-time, sample-by-sample navigation. It fuses high-frequency IMU data with low-frequency GNSS using an Extended Kalman Filter (EKF), augmented with an AI-driven velocity estimator (GRU) and Non-Holonomic Constraints (NHC) during GNSS outages.

## 2. Directory Structure
```text
id_engine/
  ├── __init__.py          # Package exports
  ├── types.py             # Data schemas (EngineInput, EngineOutput, Enums)
  ├── config.py            # Hyperparameters and noise covariances (R matrices)
  ├── engine.py            # Core IDREngine orchestrator
  ├── preprocessing.py     # Causal IMU gravity alignment (from Exp 15A/22C)
  ├── gnss_quality.py      # Causal GNSS degradation state machine (from Exp 22A)
  ├── ai_velocity.py       # Custom GRU Inference wrapper (from Exp 22C)
  ├── fusion.py            # Hybrid EKF physics wrapper (from Exp 22C/22D)
  ├── nhc.py               # Non-Holonomic Constraints (from Exp 22D)
  ├── map_matching.py      # KDTree Map matching (from Exp 22D)
  ├── utils.py             # AHRS and Projection utilities
  ├── simulator.py         # Mock kinematic generator for testing
  └── demo.py              # CLI demonstrator
```

## 3. Causal Pipeline
The `engine.update()` method receives single `EngineInput` frames (40Hz expected for IMU). The internal workflow is strictly causal with no ground truth or future knowledge leakage:

1. **Orientation**: Uses `MadgwickAHRS` to update pitch/roll/yaw if not provided.
2. **Preprocessing**: IMU frames are normalized to a local gravity frame using a low-pass accelerometer estimate.
3. **EKF Prediction**: Classical kinematics integrate IMU rotation and velocity estimates to predict new position.
4. **GNSS Quality**: The `CausalGNSSQualityEngine` evaluates raw GNSS accuracy and applies time-based hysteresis to transition between `GOOD`, `DEGRADED`, `LOST`, and `RECOVERING`.
5. **GNSS Update**: If GNSS is trusted, position constraints are pushed to the EKF.
6. **AI Update**: If 200 history frames exist, the AI model estimates $\Delta V$. This acts as a pseudo-measurement in the EKF, constrained by the Mahalanobis $\chi^2$ gate.
7. **Constraints**: During outages (DR mode), NHC and Map Matching attempt to bind lateral velocity and position to valid geometry.
8. **Output**: The fused optimal state is returned.

## 4. Sub-system Rules
- **GPU Inference**: `torch.inference_mode(True)` is maintained to ensure low-latency model evaluation.
- **Outage Recovery**: After a GNSS outage, 4 consecutive high-quality GNSS samples must arrive to switch from `RECOVERING` to `GOOD`. This prevents teleporting to erroneous initial fixes.
- **Gating**: Any pseudo-measurement (AI, NHC, Map) whose innovation Mahalanobis distance exceeds $\chi^2 = 25.0$ is strictly rejected to prevent divergence.
