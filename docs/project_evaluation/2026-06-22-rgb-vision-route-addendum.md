# RGB Pure-Vision Route Addendum

Date: 2026-06-22

## Context

The project owner clarified that the original RGB-camera pure-vision route can continue. The main dust concern applies to the civil-construction period. When the UGV is allowed to enter the tunnel for this project, the civil work is expected to be complete, so heavy construction dust is not the dominant operating condition.

## Impact on Technical Direction

This changes the project framing from a single preferred LiDAR-RGB fusion route to a dual-route plan:

- RGB-only engineering route: continue the existing pure-vision/HBD-Net-RT direction as the near-term MVP and runnable baseline.
- LiDAR-RGB fusion route: keep LiDAR, Transformer fusion, RL, and diffusion as enhanced research or safety-upgrade tracks, especially if the trench boundary or obstacle-distance requirements exceed what RGB can verify safely.

The RGB-only route should not be treated merely as legacy code. It remains an active engineering path that can produce a working demo and training/evaluation baseline.

## Updated Engineering Position

Recommended near-term order:

1. Continue the RGB-only baseline with free-space, hard-boundary, obstacle, person, and engineering-vehicle perception.
2. Evaluate the RGB-only baseline under the actual post-civil tunnel conditions: lighting, reflections, water stains, repeating textures, tunnel wall appearance, and right-side trench visibility.
3. Add a right-side distance sensor, LiDAR, or ToF if RGB-only trench-margin estimation is not reliable enough for safety acceptance.
4. Use Transformer first for temporal RGB consistency or LiDAR-RGB fusion experiments, not for direct motor control.
5. Keep RL and diffusion as safety-filtered trajectory proposal modules after a costmap/planner baseline exists.

## Repository Implications

- `baselines/hbdnet_rt/` should be an active RGB-only engineering baseline.
- `baselines/vision_obstacle_avoidance_legacy/` remains a legacy/reference baseline unless specific utilities are reused.
- `research/rgb_vision/` should exist for pure-RGB perception experiments that go beyond the preserved baseline.
- `research/transformer_fusion/` remains useful, but it is no longer the only primary research track.
- Shared `src/tunnel_nav/` should support both RGB-only and sensor-fusion adapters.

## Remaining Questions

- What are the actual lighting conditions after civil construction ends?
- Is the right-side trench edge visually clear enough for RGB-only hard-boundary detection?
- What minimum trench safety distance is required by the project owner?
- Is a low-cost right-side ToF/LiDAR allowed as an independent safety channel even if the main route remains RGB-only?
