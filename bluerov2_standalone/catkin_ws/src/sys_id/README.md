### System Identification for the BlueROV

The data used to identify the dynamics model is collected using the following [script](../vandy_bluerov/nodes/vandy_sys_id.py)

### Relevant Scripts: 
- [blue_rov_sys_id.m (Black Box System Identification Ax + Bu)](blue_rov_sys_id.m)
- [runSysID.m (Black Box System Identification Ax + Bu) With Quaternion instead of Yaw](runSysID.m)
- [bluerov_sys_id_bicycle.m (Kinematic Bicycle Model)](bluerov_sys_id_bicycle.m)
- [bluerov_sys_id_bicycle_degraded.m (Kinematic Bicycle Model Degregaded Operation)](bluerov_sys_id_bicycle_degraded.m)

### Validation Scripts
- [validate_bluerov_bicycle.m (Validation Script for BlueROV Bicycle Model)](validate_bluerov_bicycle.m)
- [validate_blue_rov_model.m (Validation Script for Black Box SysID)](validate_blue_rov_model.m)
- [validate_quat_model.m (Validation Script for Black Box SysID Quaternion)](validate_quat_model.m)

### Simulation of reachable set propagations

- [simulate_model_bicycle.m](simulate_model_bicycle.m)
- [simulate_model_bicycle_reach_degraded.m](simulate_model_bicycle_reach_degraded.m)
- [simulate_model_bicycle_reach.m](simulate_model_bicycle_reach.m )