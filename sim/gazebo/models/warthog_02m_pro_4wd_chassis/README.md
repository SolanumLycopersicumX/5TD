# Warthog-02M-Pro 4WD Chassis Gazebo Model

This model wraps the provided STEP CAD with a simplified Gazebo-ready vehicle model.

- Visual mesh: `meshes/chassis_visual.stl`
- Mesh units: millimeters
- SDF/URDF mesh scale: `0.001 0.001 0.001`
- CAD mesh normalized length/width/height: `1.608 x 1.206 x 0.999 m`
- Collision geometry: simplified body box plus four cylinder wheels
- Wheel visual appearance: CAD mesh only; simplified wheel cylinders are collision/drive links, not rendered visuals
- Drive interface: Gazebo Sim DiffDrive plugin on `/cmd_vel`

The wheel cylinders are still simplified placeholders, but their centers were moved to the CAD wheel mounting plates: front x `0.343 m`, rear x `-0.498 m`, y `+/-0.500 m`, radius `0.285 m`, width `0.220 m`. Measure the real chassis before using these values for controller tuning.

## Flat Test World

Use the provided flat-ground world for drive testing so the model is spawned on a collision plane:

```bash
source /opt/ros/jazzy/setup.bash
export GZ_SIM_RESOURCE_PATH=/home/tomato/5TD/sim/gazebo/models:${GZ_SIM_RESOURCE_PATH}
export SDF_PATH=/home/tomato/5TD/sim/gazebo/models:${SDF_PATH}
gz sim -r /home/tomato/5TD/sim/gazebo/worlds/warthog_flat_test.sdf
```

The `-r` flag starts physics unpaused. Without a world that contains ground collision, the chassis will fall under gravity.

## Driver-Style Keyboard Control

After launching the flat test world, run this in another terminal to control Gazebo through a driver-style adapter instead of ROS teleop:

```bash
cd /home/tomato/5TD
source /opt/ros/jazzy/setup.bash
python3 tools/sim/gazebo_driver_keyboard.py --linear 0.10 --angular 0.30
```

Keys: `W/S` forward/backward, `A/D` rotate, `I/J/L` forward arc, `Space` or `K` stop, `Q` quit.
