# Multi-Robot Visual Localization and ML Navigation Demo

This repository contains a Python/OpenCV simulation for a multi-robot navigation challenge without LiDAR. The system simulates two ground robots, **ANYmal** and **Husky**, moving through a 2D warehouse-like environment while using camera-based perception to support navigation, localization, and collision avoidance.

The main executable file is:

```bash
live_multi_robot_visual.py
```

The demo shows a live dashboard with:

- a top-down map of the environment,
- ANYmal's robot point-of-view camera,
- Husky's robot point-of-view camera,
- an overhead RGB camera used for visual localization,
- Hough line detection overlays,
- estimated steering commands,
- task assignments,
- replanning count,
- avoided collision count,
- final saved video and dashboard image.

---

## Project Goal

The goal is to demonstrate multi-robot coordination using **vision-only perception**. No LiDAR, sonar, GPS, or ideal external position sensor is used as the perception source.

The robots must:

1. Move inside a shared warehouse map.
2. Follow visual guide lines from a camera perspective.
3. Estimate robot positions using a simulated overhead RGB camera.
4. Use those visual estimates for planning and collision avoidance.
5. Complete pick-up and drop-off tasks.
6. Show camera views and algorithm overlays during execution.

---

## Main Features

### 1. Multi-Robot Coordination

The simulation includes two robots:

- **ANYmal**
- **Husky**

The robots receive logistics-style tasks from stations to drop-off zones:

```python
stations = {"S1", "S2", "S3"}
dropoffs = {"D1", "D2"}
tasks = [("S1", "D1"), ("S2", "D2"), ("S3", "D1")]
```

Tasks are assigned using a nearest-task strategy. Each robot plans a route and moves through the shared environment while avoiding the other robot.

### 2. A* Path Planning

The environment is represented as an occupancy grid:

- `0` = free space
- `1` = wall or obstacle

The robots use **A\*** to generate paths from their current estimated position to the next station or drop-off.

### 3. Vision-Based Line Following

Each robot has a simulated forward-facing RGB camera. The camera view contains visual guide lines representing the current navigation path.

The perception pipeline uses:

- grayscale conversion,
- Canny edge detection,
- Hough line detection,
- lane-center estimation,
- center-error calculation,
- steering command estimation.

The camera overlay shows:

- detected Hough lines,
- estimated lane center,
- image center,
- steering command `omega`,
- detected landmarks,
- detected robot markers,
- current motion mode.

### 4. End-of-Line Turning Behavior

The latest version avoids unrealistic instant 90-degree turns.

The robot now:

1. moves straight while the line is visible in front,
2. slows/stops at the end of a line segment,
3. rotates gradually toward the next path segment,
4. resumes forward motion once aligned.

This creates a smoother animation and a more realistic turn behavior.

### 5. Visual Localization with Overhead Camera

The simulation includes an overhead RGB camera that observes the map from above.

Robots are detected through colored visual markers:

- ANYmal marker: red
- Husky marker: blue

The overhead camera converts detected marker pixels back into estimated grid coordinates. These estimated positions are used by the planner and collision-avoidance logic.

This makes the simulation explicitly use visual information instead of relying only on internal ground-truth positions.

### 6. Machine Learning Steering

The script supports three steering modes:

| Mode | Description |
|---|---|
| `rule` | Rule-based steering from lane center error |
| `gb` | Gradient Boosting Regressor |
| `xgb` | XGBoost Regressor |

The ML models are trained on synthetic Hough-derived features generated inside the script. Features include:

- number of detected lines,
- center error,
- left/right line balance,
- mean line length,
- slope statistics,
- visual confidence indicators.

The predicted output is an angular correction `omega`.

---

## Requirements

Recommended Python version:

```bash
Python 3.10+
```

Required packages:

```bash
numpy
opencv-python
scipy
scikit-learn
matplotlib
pillow
```

Optional package for XGBoost mode:

```bash
xgboost
```

---

## Installation

### Recommended: use a virtual environment

```bash
python3 -m venv hackenv
source hackenv/bin/activate
python -m pip install --upgrade pip
python -m pip install "numpy>=1.23.5,<2.0" "opencv-python<4.10" scipy scikit-learn matplotlib pillow xgboost
```

The NumPy version is pinned below 2.0 because some OpenCV builds may fail with NumPy 2.x.

---

## How to Run

### Gradient Boosting mode

```bash
python3 live_multi_robot_visual.py --ml gb
```

### XGBoost mode

```bash
python3 live_multi_robot_visual.py --ml xgb
```

### Rule-based mode

```bash
python3 live_multi_robot_visual.py --ml rule
```

---

## Recommended Demo Command

For a smooth and readable animation:

```bash
python3 live_multi_robot_visual.py --ml gb --speed 0.08 --turn-rate 0.025 --fps 15
```

For XGBoost:

```bash
python3 live_multi_robot_visual.py --ml xgb --speed 0.08 --turn-rate 0.025 --fps 15
```

---

## Command-Line Arguments

| Argument | Options / Type | Description |
|---|---:|---|
| `--ml` | `rule`, `gb`, `xgb` | Steering method to use |
| `--speed` | float | Robot linear speed in grid cells per frame. Smaller is slower |
| `--turn-rate` | float | Robot angular speed in radians per frame. Smaller makes turns slower |
| `--fps` | int | Display and video frame rate |
| `--steps` | int | Maximum number of simulation steps |
| `--headless` | flag | Run without opening an OpenCV window; useful for Colab or remote machines |
| `--no-video` | flag | Do not save the MP4 output |

---

## Live Controls

When the OpenCV window is active:

| Key | Action |
|---|---|
| `space` | Pause / resume simulation |
| `q` | Quit |
| `ESC` | Quit |

---

## Output Files

The script creates an output folder:

```bash
outputs_live/
```

Generated files:

```bash
outputs_live/live_map_and_povs.mp4
outputs_live/final_dashboard.png
```

The MP4 is the main video artifact for the assignment. It shows the top-down map and camera views at the same time.

---

## What the Dashboard Shows

### Left side: Top-down map

Shows:

- warehouse walls and obstacles,
- stations,
- drop-off points,
- robot positions,
- planned routes,
- visual localization estimates,
- collision-avoidance status,
- replanning count.

### Right side: Robot POV cameras

Shows:

- ANYmal camera view,
- Husky camera view,
- Hough-detected line segments,
- lane center estimate,
- steering command,
- robot and landmark detections,
- motion state.

### Bottom-right: Overhead visual localization camera

Shows:

- simulated RGB overhead view,
- colored robot markers,
- visual position estimates,
- localization annotations.

---

## How the Algorithm Works

At each simulation step:

1. The overhead RGB camera renders the map and robot markers.
2. The visual localization module detects robot markers from the overhead image.
3. Pixel detections are converted into estimated grid positions.
4. A* plans or replans paths using the visual estimates.
5. Each robot renders its own POV camera.
6. Canny and Hough detect navigation lines in the POV image.
7. Hough features are extracted.
8. The selected steering method predicts `omega`.
9. The robot moves forward if the line is visible and aligned.
10. If the robot reaches a line end or corner, it rotates gradually toward the next segment.
11. Collision avoidance checks prevent robots from occupying unsafe nearby positions.
12. The dashboard frame is rendered and optionally saved to video.

---

## Changes From Earlier Versions

This fixed version includes these improvements:

- replaced instant 90-degree turns with gradual animated turns,
- added end-of-line behavior,
- reduced constant turning during normal line following,
- cleaned the dashboard panel layout,
- fixed overlapping text in the camera panels,
- fixed the `NameError: n_lines is not defined` issue,
- added visual localization from an overhead RGB camera,
- added Gradient Boosting and XGBoost steering support,
- added cleaner metrics and output video generation.

---

## Troubleshooting

### Error: `unrecognized arguments: -- gb`

Use:

```bash
python3 live_multi_robot_visual.py --ml gb
```

Not:

```bash
python3 live_multi_robot_visual.py -- gb
```

### Error: `_ARRAY_API not found` or `numpy.core.multiarray failed to import`

This is a NumPy/OpenCV binary mismatch. Reinstall with NumPy below 2.0:

```bash
python3 -m pip uninstall -y numpy opencv-python opencv-contrib-python
python3 -m pip install "numpy>=1.23.5,<2.0" "opencv-python<4.10" scipy scikit-learn matplotlib pillow
```

Then test:

```bash
python3 - <<'PY'
import numpy as np
import cv2
import sklearn
print("numpy:", np.__version__)
print("cv2:", cv2.__version__)
print("sklearn:", sklearn.__version__)
PY
```

### XGBoost not installed

Install it:

```bash
python3 -m pip install xgboost
```

Then run:

```bash
python3 live_multi_robot_visual.py --ml xgb
```

### The robots move too fast

Use a lower speed:

```bash
python3 live_multi_robot_visual.py --ml gb --speed 0.08 --turn-rate 0.025 --fps 15
```

Avoid values like:

```bash
--speed 0.6 --fps 100
```

Those make the motion hard to read and may look unstable.

---

## Suggested GitHub Repository Structure

```bash
.
├── README.md
├── live_multi_robot_visual.py
├── outputs_live/                 # generated after running the script
│   ├── live_map_and_povs.mp4
│   └── final_dashboard.png
└── requirements.txt              # optional
```

Optional `requirements.txt`:

```txt
numpy>=1.23.5,<2.0
opencv-python<4.10
scipy
scikit-learn
matplotlib
pillow
xgboost
```

---

## Assignment Summary

This project demonstrates a camera-only multi-robot navigation system. The robots use visual perception to detect route lines, estimate their location, plan paths, avoid collisions, and complete station-to-dropoff tasks. The system integrates classical computer vision, geometric reasoning, A* path planning, visual localization, and optional Machine Learning steering through Gradient Boosting or XGBoost.
