# BEV / DWA Plan Feasibility Comparison

Date: 2026-06-24

Compared files:

- External proposal: `tunnel_ugv_plan_A_BEV_DWA_trajectory.md`
- Current local plan: `docs/superpowers/plans/2026-06-24-rs232-navigation-bridge.md`

## Short Conclusion

The external BEV / Risk Grid / DWA proposal is directionally correct and is a better medium-term navigation architecture than pure image-space planning.

It should not replace the current local plan immediately. The current local plan is still the right first step because it is smaller, testable now, does not require camera calibration, and keeps RS232 in dry-run mode.

Recommended merge:

```text
Current local plan
→ Stage 0: saved fused masks → image-space candidate path → command JSON / overlay → RS232 dry-run

External BEV / DWA plan
→ Stage 1: calibrated BEV / IPM → occupancy grid → risk grid
→ Stage 2: DWA over risk grid → safety state machine
→ Stage 3: live RS232 only after validation
```

## Feasibility of the External Proposal

The proposal is feasible as the eventual engineering route, but it assumes several capabilities are already production-ready. In the current repository, they are not.

### Feasible Now

- Use the current segmentation outputs as planning inputs:
  - `safe_passable`
  - `ditch`
  - `tunnel_wall`
  - `left_barrier`
- Export fused masks for downstream planning.
- Build an occupancy-like forbidden mask from `ditch | tunnel_wall | ~safe_passable`.
- Build command overlays for manual review.
- Keep learning outputs behind a safety layer.
- Keep OpenPilot-style waypoint prediction as a later candidate generator, not final control.

### Not Ready Yet

- Calibrated BEV / IPM:
  - requires camera height, pitch, intrinsic calibration, and ground reference points.
  - without this, meter-level clearance and vehicle footprint checks are not trustworthy.
- Production Risk Grid:
  - needs reliable mapping from semantic masks to vehicle-coordinate cells.
  - needs dilation by vehicle half-width and safety margin.
- Full DWA:
  - needs a metric grid and vehicle kinematic model.
  - must use conservative speeds for bring-up, not the proposal's example `0.2-1.5 m/s` range.
- Safety State Machine:
  - good idea, but it should be built after the command and risk interfaces are stable.
- Object detections:
  - the proposal mentions workers, vehicles, hanging objects, debris, and boxes.
  - current active perception work is mainly road / ditch / wall segmentation, so detection fields should be optional at first.

## Comparison

| Topic | Current Local Plan | External BEV / DWA Proposal | Assessment |
|---|---|---|---|
| Immediate feasibility | High | Medium | Current plan can be implemented with saved masks and tests now. |
| Safety for first bring-up | High | Medium | Current plan defaults to no serial access. External plan is safe conceptually but larger and easier to miswire early. |
| Final navigation quality | Medium | High | BEV + Risk Grid + DWA is the stronger long-term planner. |
| Calibration need | Low | High | Current image-space bridge avoids calibration initially. BEV needs physical camera calibration. |
| Vehicle footprint / clearance | Weak | Strong | BEV is required for trustworthy meter-scale clearance. |
| Explainability | Medium | High | DWA trajectories over risk grids are easier to audit once BEV is calibrated. |
| Development risk | Low | Medium-High | External plan has more interfaces and assumptions. |
| RS232 readiness | High for dry-run | Medium | Current plan explicitly handles Modbus registers and angular sign. |
| Legacy-code dependency | Avoided | Ambiguous | External plan says modules already exist, but current integrated package is still empty. |

## Advantages of the External Proposal

- It describes the correct long-term architecture:
  - perception.
  - BEV / IPM.
  - occupancy grid.
  - risk grid.
  - DWA.
  - safety state machine.
  - control adapter.
- It handles safety in the right place: learning is not allowed to directly control the vehicle.
- It makes vehicle footprint, trench margin, and hard-boundary checks first-class concepts.
- It naturally supports future LiDAR, right-side range sensors, and learned waypoint proposals.
- It gives a clear research boundary: OpenPilot-inspired trajectory prediction is a candidate proposal module only.

## Weaknesses of the External Proposal

- It overstates current implementation readiness. The current `src/tunnel_nav` package is still a future integration package, while BEV / occupancy / DWA code mostly exists only in the legacy baseline.
- It depends on calibrated BEV before the project has measured camera geometry.
- It introduces many components at once, which makes early debugging harder.
- It mentions detection outputs that are not part of the current active segmentation pipeline.
- Its example DWA speeds are too high for first RS232 bring-up. Initial live tests should be closer to `0.05-0.10 m/s`.
- It uses steering-centric language in places, while the selected RS232 VCU accepts linear velocity and angular velocity. The bridge should use `linear_mps` and `angular_radps` internally.
- It does not emphasize the current angular-sign conflict enough. That remains a blocking safety validation item.

## Advantages of the Current Local Plan

- It is immediately implementable with current artifacts.
- It keeps all new runtime code under `src/tunnel_nav`.
- It does not import the legacy baseline.
- It starts from saved masks, so unit tests can use synthetic data.
- It produces command JSON and overlays before any vehicle control.
- It explicitly handles RS232 dry-run conversion:
  - register `1040` for linear velocity.
  - register `1041` for angular velocity.
  - `angular_sign` configuration.
  - no serial opening by default.
- It is small enough to debug end to end.

## Weaknesses of the Current Local Plan

- Image-space planning is not enough for real driving.
- It cannot reliably enforce meter-scale clearance.
- It does not model the vehicle footprint properly.
- It may depend too much on the camera mounting position.
- Candidate paths in image coordinates are only an interim planning proxy.
- Before live driving, it must be upgraded or wrapped by BEV / Risk Grid / DWA safety checks.

## Recommended Revision to the Current Plan

Keep the current plan, but explicitly mark it as Stage 0.

### Stage 0: Offline Command Bridge

Purpose:

- verify mask-to-command data flow.
- verify command JSON schema.
- verify overlays.
- verify RS232 dry-run register conversion.
- verify angular sign configurability.

No live vehicle motion.

### Stage 1: BEV and Risk Grid

Add:

- camera calibration notes and measured parameters.
- homography / IPM module.
- local grid metadata:
  - x left/right.
  - y forward.
  - resolution.
  - grid origin.
- occupancy grid:
  - `ditch`, `tunnel_wall`, and outside `safe_passable` are occupied.
- risk grid:
  - hard boundaries = 1.0.
  - near-boundary dilation = high risk.
  - safe road center = low risk.

### Stage 2: DWA and Safety State Machine

Add:

- DWA candidate simulation over metric grid.
- collision and risk checks.
- trench margin threshold.
- steering / angular rate smoothing.
- stop / cautious / slowdown / manual takeover states.

### Stage 3: Live RS232 Bring-Up

Only after Stages 0-2:

- confirm serial port.
- confirm Modbus node address.
- confirm emergency stop.
- confirm angular sign with restrained or lifted vehicle.
- cap speed to `0.05-0.10 m/s`.

## Practical Decision

The best route is not "current plan vs external plan".

The best route is:

```text
Current plan first for testable offline bridge
External BEV / DWA plan next for real navigation safety
```

Do not skip the current plan, because without it the project has no verified command schema, no RS232 dry-run path, and no easy way to inspect per-frame decisions.

Do not stop at the current plan, because image-space planning alone is not enough for safe real vehicle motion near a drainage channel.

