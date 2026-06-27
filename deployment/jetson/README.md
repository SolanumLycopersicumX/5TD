# Jetson Vision Trajectory Deployment

This directory records the Jetson deployment path for the RGB road-recognition and image-space trajectory follower.

## Current Target

- Jetson IP on direct Ethernet: `192.168.137.248`
- Jetson user: `bqtec`
- Local Ethernet IP used during deployment: `192.168.137.1/24`
- Runtime directory on Jetson: `/home/bqtec/5TD`
- Jetson OS observed during deployment: Ubuntu 22.04, L4T R36.5, aarch64

## Installed Runtime Dependencies

The first native deployment installed these packages on the Jetson:

```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-opencv python3-serial python3-venv
python3 -m pip install --user --no-cache-dir torch==2.8.0
```

This gives a CPU PyTorch runtime: `torch 2.8.0+cpu`. CUDA acceleration is not active in this native setup. Use an NVIDIA Jetson PyTorch container or NVIDIA Jetson-specific PyTorch wheel later if GPU inference is required.

## Synced Runtime Files

The deployment copied this minimal runtime set to `/home/bqtec/5TD`:

- `1/`
- `src/`
- `tools/`
- `configs/`
- `tests/`
- `deployment/`
- `README.md`
- `runs/passable_ego/passable_ditch_artifact_v3_finetune/`
- `runs/passable_ego/boundary_wall_aux_v2_no_testvideo/`

## Perception And Trajectory Dry Run

Use this command on the Jetson to run camera inference and trajectory generation without opening RS232:

```bash
cd /home/bqtec/5TD
python3 tools/robot/vision_autodrive_trajectory.py \
  --camera /dev/video0 \
  --dry-run \
  --no-display \
  --cpu \
  --width 640 \
  --height 480 \
  --fps 5 \
  --rate-hz 1
```

The command should print `[STATE] ... cmd=(...)` lines. `low_passable safe=0.00` means the current camera scene was not recognized as safe road; it is not a runtime failure.

## Live Driver Command

Only use live driver mode after the RS232 device is connected, the physical emergency stop is ready, and the vehicle is restrained or lifted for first bring-up.

```bash
cd /home/bqtec/5TD
python3 tools/robot/vision_autodrive_trajectory.py \
  --camera /dev/video0 \
  --port /dev/ttyUSB0 \
  --linear 0.012 \
  --max-angular 0.08 \
  --enable-driver
```

Add `--release-estop` only when the external safety procedure says it is safe to clear the driver emergency-stop bit.

## Verified On Jetson

Commands run successfully during deployment:

```bash
python3 -m py_compile tools/passable_segmentation/live_webcam_fused_preview.py tools/robot/vision_autodrive_trajectory.py tools/robot/vision_autodrive_forward.py src/tunnel_nav/vision_trajectory.py src/tunnel_nav/vision_autodrive.py
python3 -m unittest tests/test_vision_autodrive.py tests/test_vision_trajectory.py tests/test_vision_autodrive_forward_script.py tests/test_vision_autodrive_trajectory_script.py tests/test_live_webcam_fused_preview.py -v
```

A timed dry-run of `vision_autodrive_trajectory.py` also ran successfully against `/dev/video0`.
