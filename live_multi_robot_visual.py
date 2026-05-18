#!/usr/bin/env python3
"""
live_multi_robot_visual_localization_ml.py

Live multi-robot visual navigation demo for the HackReto.
Shows, at the same time:
  - top-down warehouse map
  - ANYmal robot POV with Hough lane detection
  - Husky robot POV with Hough lane detection
  - overhead RGB camera that visually estimates robot positions
  - task state, replans, avoided collisions, FPS, omega command
  - optional ML steering with Gradient Boosting or XGBoost

Run locally:
    python live_multi_robot_visual_localization_ml.py

Headless / Colab-safe run that only records video:
    python live_multi_robot_visual_localization_ml.py --headless

Controls in live mode:
    q or ESC: quit
    space: pause/resume
"""

from __future__ import annotations

import argparse
import heapq
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from sklearn.ensemble import GradientBoostingRegressor
    SKLEARN_OK = True
except Exception:
    GradientBoostingRegressor = None
    SKLEARN_OK = False

try:
    from xgboost import XGBRegressor
    XGBOOST_OK = True
except Exception:
    XGBRegressor = None
    XGBOOST_OK = False

# -----------------------------
# Scenario configuration
# -----------------------------
GRID_W, GRID_H = 72, 34
CELL = 0.18
MAX_STEPS = 4800
SAFE_DIST = 4.0
DEFAULT_LINEAR_SPEED = 0.11   # grid cells per frame; lower = slower translation
DEFAULT_TURN_RATE = 0.040     # rad per frame; lower = slower animated turns
ALIGN_TOLERANCE = 0.08       # rad; robot must face the next line before moving
REACH_TOLERANCE = 0.045      # grid cells; distance considered reached
WINDOW_NAME = "HackReto live: map + robot POV cameras + end-of-line turns"

POV_W, POV_H = 460, 300
HORIZON_Y = 105
HALF_TRACK_WIDTH = 0.42
LOOKAHEAD_METERS = 5.0
SAMPLE_STEP = 0.20
K_STEER = 1.6

OUT = Path("outputs_live")
OUT.mkdir(exist_ok=True)

# 0 = free, 1 = obstacle/wall
occ = np.zeros((GRID_H, GRID_W), dtype=np.uint8)
occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = 1
occ[6:8, 8:42] = 1
occ[16:18, 16:56] = 1
occ[24:26, 5:48] = 1
occ[4:14, 52:55] = 1
occ[20:31, 60:63] = 1
# gaps
occ[6:8, 26:31] = 0
occ[16:18, 36:42] = 0
occ[24:26, 20:26] = 0

stations = {"S1": (7, 4), "S2": (10, 21), "S3": (45, 11)}
dropoffs = {"D1": (66, 5), "D2": (66, 28)}
robots_init = {"ANYmal": (4, 29), "Husky": (4, 12)}
tasks = [("S1", "D1"), ("S2", "D2"), ("S3", "D1")]

ROBOT_COLORS = {
    "ANYmal": (70, 160, 255),   # BGR orange-ish
    "Husky": (255, 110, 80),    # BGR blue-ish
}


# Separate marker colors used by the simulated overhead camera for visual localization.
# The planner can use these estimated positions instead of ideal ground-truth positions.
LOCALIZATION_COLORS = {
    "ANYmal": (0, 0, 255),      # red marker in BGR
    "Husky": (255, 0, 0),       # blue marker in BGR
}


def grid_to_overhead_px(p: Tuple[int, int], w: int, h: int) -> Tuple[int, int]:
    """Map a grid coordinate to a pixel in the overhead RGB camera."""
    x, y = p
    u = int(round(x / max(1, GRID_W - 1) * (w - 1)))
    v = int(round((GRID_H - 1 - y) / max(1, GRID_H - 1) * (h - 1)))
    return u, v


def overhead_px_to_grid(u: float, v: float, w: int, h: int) -> Tuple[int, int]:
    """Invert the known overhead-camera homography into a grid estimate."""
    x = int(round(u / max(1, w - 1) * (GRID_W - 1)))
    y = int(round(GRID_H - 1 - v / max(1, h - 1) * (GRID_H - 1)))
    return int(np.clip(x, 0, GRID_W - 1)), int(np.clip(y, 0, GRID_H - 1))


def render_overhead_camera(robots: Dict[str, "MobileRobot"], cam_w: int = 720, cam_h: int = 340) -> np.ndarray:
    """Synthetic fixed RGB camera looking down at the whole workcell."""
    img = np.full((cam_h, cam_w, 3), 238, dtype=np.uint8)
    # Draw occupancy map as the visual scene seen by the camera.
    for y in range(GRID_H):
        for x in range(GRID_W):
            u0, v0 = grid_to_overhead_px((x, y), cam_w, cam_h)
            u1, v1 = grid_to_overhead_px((x + 1, y - 1), cam_w, cam_h)
            xlo, xhi = sorted([u0, u1])
            ylo, yhi = sorted([v0, v1])
            if occ[y, x] == 1:
                cv2.rectangle(img, (xlo, ylo), (xhi, yhi), (65, 65, 65), -1)
    for name, p in stations.items():
        u, v = grid_to_overhead_px(p, cam_w, cam_h)
        cv2.rectangle(img, (u - 8, v - 8), (u + 8, v + 8), (0, 180, 0), -1)
        cv2.putText(img, name, (u + 9, v - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 80, 0), 1)
    for name, p in dropoffs.items():
        u, v = grid_to_overhead_px(p, cam_w, cam_h)
        cv2.circle(img, (u, v), 9, (180, 0, 0), -1)
        cv2.putText(img, name, (u + 9, v - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 0, 0), 1)
    for name, r in robots.items():
        u, v = grid_to_overhead_px(r.xy(), cam_w, cam_h)
        cv2.circle(img, (u, v), 11, LOCALIZATION_COLORS[name], -1)
        cv2.circle(img, (u, v), 14, (255, 255, 255), 2)
        cv2.putText(img, name, (u + 13, v + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1)
    cv2.putText(img, "overhead RGB localization camera", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (25, 25, 25), 2)
    return img


def estimate_robot_positions_from_overhead(raw: np.ndarray) -> Tuple[Dict[str, Tuple[int, int]], np.ndarray]:
    """Detect colored robot markers and convert pixel centroids into grid coordinates."""
    h, w = raw.shape[:2]
    hsv = cv2.cvtColor(raw, cv2.COLOR_BGR2HSV)
    masks = {
        "ANYmal": cv2.bitwise_or(
            cv2.inRange(hsv, (0, 80, 80), (10, 255, 255)),
            cv2.inRange(hsv, (170, 80, 80), (180, 255, 255)),
        ),
        "Husky": cv2.inRange(hsv, (100, 80, 80), (130, 255, 255)),
    }
    ann = raw.copy()
    estimates: Dict[str, Tuple[int, int]] = {}
    for name, mask in masks.items():
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) > 60]
        if not contours:
            continue
        c = max(contours, key=cv2.contourArea)
        m = cv2.moments(c)
        if abs(m["m00"]) < 1e-9:
            continue
        cx = float(m["m10"] / m["m00"])
        cy = float(m["m01"] / m["m00"])
        est = overhead_px_to_grid(cx, cy, w, h)
        estimates[name] = est
        x, y, bw, bh = cv2.boundingRect(c)
        cv2.rectangle(ann, (x, y), (x + bw, y + bh), (0, 255, 255), 2)
        cv2.circle(ann, (int(cx), int(cy)), 5, (0, 255, 255), -1)
        cv2.putText(ann, f"{name} est={est}", (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 120, 120), 2)
    cv2.putText(ann, "visual pose estimates feed planning / collision avoidance", (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (20, 20, 20), 2)
    return estimates, ann

# -----------------------------
# Planning and task assignment
# -----------------------------
def astar(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int], extra_blocked=None) -> List[Tuple[int, int]]:
    extra_blocked = set(extra_blocked or [])
    start = tuple(map(int, start))
    goal = tuple(map(int, goal))
    h, w = grid.shape

    def free(p: Tuple[int, int]) -> bool:
        x, y = p
        return 0 <= x < w and 0 <= y < h and grid[y, x] == 0 and p not in extra_blocked

    if not free(start) or not free(goal):
        return [start]

    def heuristic(p):
        return abs(p[0] - goal[0]) + abs(p[1] - goal[1])

    openq = [(heuristic(start), 0, start)]
    came = {}
    g = {start: 0}
    moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    while openq:
        _, cost, p = heapq.heappop(openq)
        if p == goal:
            path = [p]
            while p in came:
                p = came[p]
                path.append(p)
            return path[::-1]
        for dx, dy in moves:
            q = (p[0] + dx, p[1] + dy)
            if not free(q):
                continue
            ng = cost + 1
            if ng < g.get(q, 1e9):
                g[q] = ng
                came[q] = p
                heapq.heappush(openq, (ng + heuristic(q), ng, q))
    return [start]


def assign_tasks_nearest(robot_positions, task_list):
    remaining = list(task_list)
    assigned = {r: [] for r in robot_positions}
    pos = {r: robot_positions[r] for r in robot_positions}
    while remaining:
        best = None
        for robot_name, rp in pos.items():
            for task in remaining:
                s, _ = task
                sx, sy = stations[s]
                score = abs(rp[0] - sx) + abs(rp[1] - sy)
                if best is None or score < best[0]:
                    best = (score, robot_name, task)
        _, robot_name, task = best
        assigned[robot_name].append(task)
        pos[robot_name] = dropoffs[task[1]]
        remaining.remove(task)
    return assigned


@dataclass
class MobileRobot:
    name: str
    pos: Tuple[int, int]
    assigned: List[Tuple[str, str]]
    route: List[Tuple[int, int]] = field(default_factory=list)
    carrying: bool = False
    current_task: Optional[Tuple[str, str]] = None
    completed: int = 0
    theta: float = 0.0
    log: List[Tuple[float, float]] = field(default_factory=list)

    # Continuous animation state. The planner still works on grid cells, but the
    # robot body moves and rotates gradually between cells.
    x: float = 0.0
    y: float = 0.0
    target_cell: Optional[Tuple[int, int]] = None
    motion_mode: str = "line_follow"
    turn_reason: str = ""

    def __post_init__(self):
        self.x = float(self.pos[0])
        self.y = float(self.pos[1])
        self.theta = float(self.theta)

    def xy(self) -> Tuple[float, float]:
        return float(self.x), float(self.y)

    def choose_next_goal(self):
        if self.current_task is None:
            if not self.assigned:
                return None
            self.current_task = self.assigned.pop(0)
            self.carrying = False
        s, d = self.current_task
        return dropoffs[d] if self.carrying else stations[s]

    def step_to_next_cell(self, linear_speed: float = DEFAULT_LINEAR_SPEED, turn_rate: float = DEFAULT_TURN_RATE):
        """Line-following-style motion with discrete end-of-line turns.

        The previous smooth version continuously curved toward the next A* cell.
        That looked like the robot never stopped turning. This version behaves more
        like a real line follower:

        1. While a line/segment is in front, drive straight along it.
        2. At the end of the segment, if the next segment changes direction,
           stop translating and rotate gradually.
        3. Once aligned with the new segment, drive straight again.

        The turn is still animated over multiple frames; it is just gated so it
        only happens at line ends/corners, not continuously during every step.
        """
        if len(self.route) <= 1 and self.target_cell is None:
            self.motion_mode = "idle"
            self.turn_reason = "no route"
            return False

        if self.target_cell is None:
            self.target_cell = self.route[1]

        tx, ty = float(self.target_cell[0]), float(self.target_cell[1])
        dx = tx - self.x
        dy = ty - self.y
        dist = math.hypot(dx, dy)

        if dist < REACH_TOLERANCE:
            self.x, self.y = tx, ty
            self.pos = (int(round(tx)), int(round(ty)))
            if len(self.route) > 1 and self.route[1] == self.target_cell:
                self.route.pop(0)
            self.target_cell = None
            self.motion_mode = "line_end"
            self.turn_reason = "reached segment end"
            return True

        desired_theta = math.atan2(dy, dx)
        angle_error = wrap_angle(desired_theta - self.theta)

        # Turn only when the next line segment is not in front of the robot.
        # This simulates: camera does not see the guide line straight ahead, or
        # the robot reached the end/corner of the current line.
        if abs(angle_error) > ALIGN_TOLERANCE:
            dtheta = float(np.clip(angle_error, -turn_rate, turn_rate))
            self.theta = wrap_angle(self.theta + dtheta)
            self.motion_mode = "turning"
            self.turn_reason = "no line ahead / segment end"
            return True

        # Once aligned, hold heading and move straight. No continuous steering.
        step = min(linear_speed, dist)
        self.x += step * math.cos(self.theta)
        self.y += step * math.sin(self.theta)
        self.motion_mode = "line_follow"
        self.turn_reason = "line visible ahead"
        return True

    def update_task_state(self):
        if self.current_task is None:
            return
        s, d = self.current_task
        if not self.carrying and self.pos == stations[s]:
            self.carrying = True
        elif self.carrying and self.pos == dropoffs[d]:
            self.completed += 1
            self.current_task = None
            self.carrying = False


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi

# -----------------------------
# POV camera and vision pipeline
# -----------------------------
def grid_to_world(cell) -> np.ndarray:
    return np.array([float(cell[0]) * CELL, float(cell[1]) * CELL], dtype=float)


def body_transform(robot_xy: np.ndarray, theta: float, world_xy: np.ndarray) -> Tuple[float, float]:
    dx, dy = world_xy[0] - robot_xy[0], world_xy[1] - robot_xy[1]
    xb = math.cos(theta) * dx + math.sin(theta) * dy
    yb = -math.sin(theta) * dx + math.cos(theta) * dy
    return xb, yb


def project_ground_point(robot_xy: np.ndarray, theta: float, world_xy: np.ndarray):
    xb, yb = body_transform(robot_xy, theta, world_xy)
    if xb <= 0.05:
        return None
    f = 190.0
    u = POV_W / 2 + f * (yb / max(xb, 1e-3))
    v = HORIZON_Y + 160.0 / max(xb, 0.18)
    return int(round(u)), int(round(v)), xb, yb


def sample_route_world(route: List[Tuple[int, int]], lookahead=LOOKAHEAD_METERS, step=SAMPLE_STEP):
    if not route:
        return []
    pts = [grid_to_world(c) for c in route]
    dense = []
    for a, b in zip(pts[:-1], pts[1:]):
        seg = b - a
        dist = float(np.linalg.norm(seg))
        n = max(1, int(np.ceil(dist / step)))
        for i in range(n):
            t = i / n
            dense.append(a * (1 - t) + b * t)
    dense.append(pts[-1])
    out = [dense[0]]
    total = 0.0
    for a, b in zip(dense[:-1], dense[1:]):
        total += float(np.linalg.norm(b - a))
        out.append(b)
        if total >= lookahead:
            break
    return out


def draw_route_track(img: np.ndarray, robot_xy: np.ndarray, theta: float, route):
    pts = sample_route_world(route)
    if len(pts) < 2:
        return img
    left_pts, right_pts, center_pts = [], [], []
    for i, p in enumerate(pts[:-1]):
        nxt = pts[min(i + 1, len(pts) - 1)]
        tangent = nxt - p
        norm = np.linalg.norm(tangent)
        if norm < 1e-6:
            continue
        tangent = tangent / norm
        normal = np.array([-tangent[1], tangent[0]])
        for world_pt, dest in [
            (p + HALF_TRACK_WIDTH * normal, left_pts),
            (p - HALF_TRACK_WIDTH * normal, right_pts),
            (p, center_pts),
        ]:
            q = project_ground_point(robot_xy, theta, world_pt)
            if q is not None:
                u, v, _, _ = q
                if -200 <= u <= POV_W + 200 and HORIZON_Y - 50 <= v <= POV_H + 80:
                    dest.append((u, v))
    if len(left_pts) >= 2:
        cv2.polylines(img, [np.array(left_pts, dtype=np.int32)], False, (25, 25, 25), 7)
    if len(right_pts) >= 2:
        cv2.polylines(img, [np.array(right_pts, dtype=np.int32)], False, (25, 25, 25), 7)
    if len(center_pts) >= 2:
        for a, b in zip(center_pts[::2], center_pts[1::2]):
            cv2.line(img, a, b, (0, 230, 255), 3)
    return img


def render_pov(robot_name: str, robot: MobileRobot, robots: Dict[str, MobileRobot], goal):
    robot_xy = grid_to_world(robot.xy())
    img = np.full((POV_H, POV_W, 3), 235, dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (POV_W, HORIZON_Y), (232, 238, 244), -1)
    cv2.rectangle(img, (0, HORIZON_Y), (POV_W, POV_H), (205, 205, 205), -1)
    cv2.line(img, (0, HORIZON_Y), (POV_W, HORIZON_Y), (170, 170, 170), 2)

    draw_route_track(img, robot_xy, robot.theta, robot.route)

    # Landmarks: stations green, dropoffs blue
    for name, p in stations.items():
        q = project_ground_point(robot_xy, robot.theta, grid_to_world(p))
        if q is not None:
            u, v, xb, _ = q
            if -40 <= u <= POV_W + 40 and HORIZON_Y - 40 <= v <= POV_H + 40:
                s = int(np.clip(48 / max(xb, 0.4), 8, 24))
                cv2.rectangle(img, (u - s, v - s), (u + s, v + s), (0, 180, 0), -1)
                cv2.putText(img, name, (u - s, max(15, v - s - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 80, 0), 1)
    for name, p in dropoffs.items():
        q = project_ground_point(robot_xy, robot.theta, grid_to_world(p))
        if q is not None:
            u, v, xb, _ = q
            if -40 <= u <= POV_W + 40 and HORIZON_Y - 40 <= v <= POV_H + 40:
                s = int(np.clip(48 / max(xb, 0.4), 8, 24))
                cv2.rectangle(img, (u - s, v - s), (u + s, v + s), (180, 0, 0), -1)
                cv2.putText(img, name, (u - s, max(15, v - s - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 0, 0), 1)

    # Other robots as red dynamic obstacles
    for other_name, other in robots.items():
        if other_name == robot_name:
            continue
        q = project_ground_point(robot_xy, robot.theta, grid_to_world(other.xy()))
        if q is not None:
            u, v, xb, _ = q
            if -50 <= u <= POV_W + 50 and HORIZON_Y - 40 <= v <= POV_H + 50:
                radius = int(np.clip(62 / max(xb, 0.5), 10, 28))
                cv2.circle(img, (u, v), radius, (0, 0, 220), -1)
                cv2.putText(img, other_name, (u - radius, max(15, v - radius - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 100), 1)

    if goal is not None:
        q = project_ground_point(robot_xy, robot.theta, grid_to_world(goal))
        if q is not None:
            u, v, _, _ = q
            if -60 <= u <= POV_W + 60 and 0 <= v <= POV_H + 60:
                cv2.circle(img, (u, v), 8, (255, 255, 0), -1)
                cv2.putText(img, "goal", (u + 7, max(15, v - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 0), 1)

    cv2.putText(img, f"{robot_name} raw RGB camera", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (30, 30, 30), 2)
    cv2.putText(img, f"motion: {robot.motion_mode}", (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (30, 30, 30), 1)
    if robot.motion_mode == "turning":
        cv2.putText(img, "turning: no line ahead / end reached", (12, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 180), 2)
    return img



# -----------------------------
# ML steering model: Gradient Boosting / XGBoost
# -----------------------------
FEATURE_NAMES = [
    "n_lines", "left_count", "right_count", "center_error_norm",
    "mean_abs_slope", "std_slope", "mean_line_length", "line_balance"
]


def build_hough_features(lines, left_x, right_x, center_error_px: float) -> np.ndarray:
    lengths, slopes = [], []
    if lines is not None:
        for ln in lines[:80]:
            x1, y1, x2, y2 = ln[0]
            dx = x2 - x1
            dy = y2 - y1
            lengths.append(float(np.hypot(dx, dy)))
            slopes.append(float(0.0 if abs(dx) < 1e-6 else dy / dx))
    n_lines = 0 if lines is None else int(len(lines))
    left_count = len(left_x)
    right_count = len(right_x)
    center_error_norm = float(center_error_px / (POV_W / 2))
    mean_abs_slope = float(np.mean(np.abs(slopes))) if slopes else 0.0
    std_slope = float(np.std(slopes)) if slopes else 0.0
    mean_line_length = float(np.mean(lengths)) if lengths else 0.0
    line_balance = float((right_count - left_count) / max(1, left_count + right_count))
    return np.array([[n_lines, left_count, right_count, center_error_norm,
                      mean_abs_slope, std_slope, mean_line_length, line_balance]], dtype=float)


def train_ml_steering_model(method: str = "gb"):
    """Train a small synthetic steering model from Hough-style geometric variables."""
    method = method.lower()
    if method == "rule":
        return None, "rule"
    if method == "xgb" and not XGBOOST_OK:
        print("XGBoost not installed; falling back to Gradient Boosting.")
        method = "gb"
    if method == "gb" and not SKLEARN_OK:
        print("scikit-learn not installed; falling back to rule-based steering.")
        return None, "rule"

    rng = np.random.default_rng(42)
    X, y = [], []
    for _ in range(2500):
        center_error_norm = rng.uniform(-0.85, 0.85)
        left_count = int(rng.integers(1, 18))
        right_count = int(rng.integers(1, 18))
        if center_error_norm > 0.15:
            right_count += int(rng.integers(0, 7))
        elif center_error_norm < -0.15:
            left_count += int(rng.integers(0, 7))
        n_lines = left_count + right_count + int(rng.integers(0, 7))
        mean_abs_slope = rng.uniform(0.45, 3.5)
        std_slope = rng.uniform(0.02, 1.1)
        mean_line_length = rng.uniform(25, 155)
        line_balance = (right_count - left_count) / max(1, left_count + right_count)
        # teacher signal: classic geometric controller, with small nonlinear corrections
        omega = -K_STEER * center_error_norm - 0.22 * line_balance + 0.04 * np.sin(2.5 * center_error_norm)
        omega += rng.normal(0, 0.025)
        omega = float(np.clip(omega, -1.6, 1.6))
        X.append([n_lines, left_count, right_count, center_error_norm,
                  mean_abs_slope, std_slope, mean_line_length, line_balance])
        y.append(omega)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if method == "xgb":
        model = XGBRegressor(
            n_estimators=160, learning_rate=0.06, max_depth=4,
            reg_lambda=1.0, objective="reg:squarederror", random_state=42, n_jobs=-1
        )
        model.fit(X, y)
        return model, "xgb"

    model = GradientBoostingRegressor(
        n_estimators=120, learning_rate=0.08, max_depth=3, random_state=42
    )
    model.fit(X, y)
    return model, "gb"


def detect_lane_and_annotate(raw: np.ndarray, steering_model=None, steering_method: str = "rule"):
    """Detect the floor guide line from a robot POV and annotate the camera image.

    This version keeps all overlay variables local to the function, so it cannot
    raise NameError from n_lines/center_error_px. Text is compact to avoid panel
    overlap after resizing.
    """
    t0 = time.perf_counter()
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    roi = gray[HORIZON_Y:, :]
    edges = cv2.Canny(roi, 70, 170)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=28, minLineLength=28, maxLineGap=18)

    ann = raw.copy()
    left_x, right_x = [], []
    n_lines = 0

    if lines is not None:
        n_lines = int(len(lines))
        for ln in lines[:70]:
            x1, y1, x2, y2 = ln[0]
            y1 += HORIZON_Y
            y2 += HORIZON_Y
            dx = x2 - x1
            dy = y2 - y1
            slope = 1e6 if abs(dx) < 1 else dy / dx
            cv2.line(ann, (x1, y1), (x2, y2), (255, 0, 255), 2)

            y_eval = POV_H - 1
            if abs(dy) > 4:
                x_eval = x1 + (y_eval - y1) * dx / dy
                if slope < -0.25:
                    left_x.append(x_eval)
                elif slope > 0.25:
                    right_x.append(x_eval)

    image_center = POV_W / 2
    lane_center = image_center
    if left_x and right_x:
        lane_center = 0.5 * (np.median(left_x) + np.median(right_x))
    elif left_x:
        lane_center = np.median(left_x) + 70
    elif right_x:
        lane_center = np.median(right_x) - 70

    center_error_px = float(lane_center - image_center)
    omega_rule = float(-K_STEER * center_error_px / (POV_W / 2))

    hough_feature_row = build_hough_features(lines, left_x, right_x, center_error_px)
    if steering_model is not None and steering_method in {"gb", "xgb"}:
        omega = float(np.clip(steering_model.predict(hough_feature_row)[0], -1.6, 1.6))
    else:
        omega = omega_rule

    # Visual steering guides.
    cv2.line(ann, (int(image_center), POV_H), (int(image_center), HORIZON_Y), (255, 255, 0), 2)
    cv2.line(ann, (int(lane_center), POV_H), (int(lane_center), HORIZON_Y + 20), (0, 255, 0), 3)
    cv2.circle(ann, (int(lane_center), POV_H - 18), 6, (0, 255, 0), -1)
    arrow_end = (int(image_center + 80 * np.clip(-omega, -1.0, 1.0)), POV_H - 35)
    cv2.arrowedLine(ann, (int(image_center), POV_H - 35), arrow_end, (0, 140, 255), 4, tipLength=0.25)

    # Color detections: other robots and landmarks.
    hsv = cv2.cvtColor(raw, cv2.COLOR_BGR2HSV)
    masks = {
        "robots": cv2.bitwise_or(
            cv2.inRange(hsv, (0, 60, 50), (10, 255, 255)),
            cv2.inRange(hsv, (170, 60, 50), (180, 255, 255)),
        ),
        "green": cv2.inRange(hsv, (45, 45, 45), (85, 255, 255)),
        "blue": cv2.inRange(hsv, (95, 45, 45), (135, 255, 255)),
    }
    counts = {"robots": 0, "green": 0, "blue": 0}
    for key, mask in masks.items():
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        count = 0
        for c in contours:
            if cv2.contourArea(c) < 55:
                continue
            x, y, w, h = cv2.boundingRect(c)
            color = (0, 0, 255) if key == "robots" else ((0, 255, 0) if key == "green" else (255, 0, 0))
            cv2.rectangle(ann, (x, y), (x + w, y + h), color, 2)
            count += 1
        counts[key] = count

    fps = 1.0 / max(1e-9, time.perf_counter() - t0)
    model_label = {"gb": "GradientBoost", "xgb": "XGBoost", "rule": "rule"}.get(steering_method, steering_method)

    # Compact overlay. This avoids the huge text blocks that were overlapping.
    overlay = [
        f"lines={n_lines}  err={center_error_px:.0f}px",
        f"omega={omega:.2f}  {model_label}",
        f"robots={counts['robots']}  lm={counts['green'] + counts['blue']}  fps={fps:.0f}",
    ]
    cv2.rectangle(ann, (6, 6), (315, 78), (245, 245, 245), -1)
    cv2.rectangle(ann, (6, 6), (315, 78), (180, 180, 180), 1)
    cv2.putText(ann, "algorithm overlay", (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (30, 30, 30), 1)
    y = 44
    for txt in overlay:
        cv2.putText(ann, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1)
        y += 16

    return ann, {
        "n_lines": n_lines,
        "center_error_px": center_error_px,
        "omega": omega,
        "omega_rule": omega_rule,
        "steering_method": steering_method,
        "fps": fps,
    }

# -----------------------------
# Dashboard rendering
# -----------------------------
def draw_topdown(robots: Dict[str, MobileRobot], step: int, assignments, replans: int, avoided: int, last_metrics, visual_positions=None) -> np.ndarray:
    scale = 13
    h, w = GRID_H * scale, GRID_W * scale
    img = np.full((h, w, 3), 245, dtype=np.uint8)

    # map cells
    for y in range(GRID_H):
        for x in range(GRID_W):
            x0, y0 = x * scale, (GRID_H - 1 - y) * scale
            if occ[y, x] == 1:
                cv2.rectangle(img, (x0, y0), (x0 + scale, y0 + scale), (70, 70, 70), -1)
            else:
                cv2.rectangle(img, (x0, y0), (x0 + scale, y0 + scale), (235, 235, 235), 1)

    def to_px(p):
        return int(p[0] * scale + scale / 2), int((GRID_H - 1 - p[1]) * scale + scale / 2)

    for name, p in stations.items():
        px = to_px(p)
        cv2.rectangle(img, (px[0] - 7, px[1] - 7), (px[0] + 7, px[1] + 7), (0, 180, 0), -1)
        cv2.putText(img, name, (px[0] + 8, px[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 90, 0), 1)
    for name, p in dropoffs.items():
        px = to_px(p)
        pts = np.array([[px[0], px[1] - 10], [px[0] - 10, px[1] + 8], [px[0] + 10, px[1] + 8]], dtype=np.int32)
        cv2.fillPoly(img, [pts], (180, 0, 0))
        cv2.putText(img, name, (px[0] + 8, px[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 0, 0), 1)

    for name, robot in robots.items():
        color = ROBOT_COLORS.get(name, (0, 120, 255))
        if len(robot.log) >= 2:
            pts = np.array([to_px(p) for p in robot.log], dtype=np.int32)
            cv2.polylines(img, [pts], False, color, 2)
        if len(robot.route) >= 2:
            pts = np.array([to_px(p) for p in robot.route], dtype=np.int32)
            cv2.polylines(img, [pts], False, (0, 210, 210), 1)
        px = to_px(robot.xy())
        cv2.circle(img, px, 8, color, -1)
        end = (int(px[0] + 16 * math.cos(-robot.theta)), int(px[1] + 16 * math.sin(-robot.theta)))
        cv2.arrowedLine(img, px, end, (20, 20, 20), 2, tipLength=0.3)
        status = "carry" if robot.carrying else "go"
        cv2.putText(img, f"{name} {status}", (px[0] + 10, px[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1)
        cv2.putText(img, f"{robot.motion_mode}", (px[0] + 10, px[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (30, 30, 30), 1)

    if visual_positions:
        for name, p in visual_positions.items():
            px = to_px(p)
            cv2.drawMarker(img, px, (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=22, thickness=2)
            cv2.putText(img, f"vision est {name}", (px[0] + 10, px[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 120, 120), 1)

    panel_h = 125
    panel = np.full((panel_h, w, 3), 250, dtype=np.uint8)
    cv2.putText(panel, "Top-down map: A* routes + visual localization + collision avoidance", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (25, 25, 25), 2)
    cv2.putText(panel, f"step={step} replans={replans} avoided_collisions={avoided}", (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (25, 25, 25), 1)
    cv2.putText(panel, f"assignments={assignments}", (12, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (25, 25, 25), 1)

    short_metrics = {}
    for robot_name, m in last_metrics.items():
        short_metrics[robot_name] = {
            "omega": round(float(m.get("omega", 0.0)), 2),
            "lines": int(m.get("n_lines", 0)),
            "fps": round(float(m.get("fps", 0.0)), 0),
        }
    cv2.putText(panel, f"POV metrics={short_metrics}", (12, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (25, 25, 25), 1)
    return np.vstack([panel, img])


def compose_dashboard(topdown: np.ndarray, anymal_ann: np.ndarray, husky_ann: np.ndarray, overhead_ann: np.ndarray) -> np.ndarray:
    """Create one clean dashboard without text overlapping camera panels."""
    topdown_resized = cv2.resize(topdown, (720, 660), interpolation=cv2.INTER_AREA)

    right_w = 560
    label_h = 28
    panel_h = 185
    gap = 10
    right_h = 660

    right = np.full((right_h, right_w, 3), 245, dtype=np.uint8)
    anymal_panel = cv2.resize(anymal_ann, (right_w, panel_h), interpolation=cv2.INTER_AREA)
    husky_panel = cv2.resize(husky_ann, (right_w, panel_h), interpolation=cv2.INTER_AREA)
    overhead_panel = cv2.resize(overhead_ann, (right_w, panel_h), interpolation=cv2.INTER_AREA)

    y0 = 0
    cv2.putText(right, "ANYmal POV camera", (12, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2)
    right[y0 + label_h:y0 + label_h + panel_h, 0:right_w] = anymal_panel

    y1 = y0 + label_h + panel_h + gap
    cv2.putText(right, "Husky POV camera", (12, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2)
    right[y1 + label_h:y1 + label_h + panel_h, 0:right_w] = husky_panel

    y2 = y1 + label_h + panel_h + gap
    cv2.putText(right, "Overhead visual localization", (12, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2)
    right[y2 + label_h:y2 + label_h + panel_h, 0:right_w] = overhead_panel

    return np.hstack([topdown_resized, right])

# -----------------------------
# Simulation loop
# -----------------------------
def run(live=True, save_video=True, max_steps=MAX_STEPS, fps=15, ml_method="gb", linear_speed=DEFAULT_LINEAR_SPEED, turn_rate=DEFAULT_TURN_RATE):
    assigned = assign_tasks_nearest(robots_init, tasks)
    assignments_text = {k: list(v) for k, v in assigned.items()}
    robots = {name: MobileRobot(name, pos, list(assigned[name])) for name, pos in robots_init.items()}
    steering_model, steering_method = train_ml_steering_model(ml_method)
    print(f"Steering method: {steering_method} ({'ML model' if steering_model is not None else 'geometric rule'})")
    replans = 0
    avoided = 0
    paused = False

    video = None
    out_path = OUT / "live_map_and_povs.mp4"
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(str(out_path), fourcc, fps, (1280, 660))

    if live:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1280, 660)

    last_metrics = {}
    visual_positions = dict(robots_init)
    overhead_ann = np.full((340, 720, 3), 235, dtype=np.uint8)
    step = 0
    while step < max_steps:
        if not paused:
            all_done = all((not r.assigned and r.current_task is None) for r in robots.values())
            if all_done:
                break

            # Planning/collision avoidance uses visual pose estimates from the previous overhead camera frame.
            # This avoids relying on ideal LiDAR/GPS-style perfect localization.
            positions = {n: visual_positions.get(n, r.pos) for n, r in robots.items()}
            proposed = {}
            for name, r in robots.items():
                goal = r.choose_next_goal()
                if goal is None:
                    proposed[name] = r.pos
                    continue
                blocked = {p for oname, p in positions.items() if oname != name}
                if not r.route or r.route[-1] != goal or r.pos != r.route[0]:
                    r.route = astar(occ, r.pos, goal, extra_blocked=blocked)
                    replans += 1
                proposed[name] = r.route[1] if len(r.route) > 1 else r.pos

            used = set()
            for name in sorted(robots.keys()):
                p = proposed[name]
                too_close = any((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 < SAFE_DIST ** 2 for q in used)
                if p in used or too_close:
                    proposed[name] = robots[name].pos
                    avoided += 1
                else:
                    used.add(p)

            for name, r in robots.items():
                # Continue an ongoing continuous turn/move until the next cell is reached.
                # If collision avoidance blocks a robot before it starts moving, it waits.
                if (proposed[name] != r.pos and len(r.route) > 1) or r.target_cell is not None:
                    if len(r.route) > 1:
                        r.route[0] = r.pos
                    r.step_to_next_cell(linear_speed=linear_speed, turn_rate=turn_rate)
                r.update_task_state()
                r.log.append(r.xy())

        # Render both POVs every display frame.
        anymal_goal = robots["ANYmal"].choose_next_goal()
        husky_goal = robots["Husky"].choose_next_goal()
        anymal_raw = render_pov("ANYmal", robots["ANYmal"], robots, anymal_goal)
        husky_raw = render_pov("Husky", robots["Husky"], robots, husky_goal)
        anymal_ann, m1 = detect_lane_and_annotate(anymal_raw, steering_model, steering_method)
        husky_ann, m2 = detect_lane_and_annotate(husky_raw, steering_model, steering_method)
        overhead_raw = render_overhead_camera(robots)
        detected_positions, overhead_ann = estimate_robot_positions_from_overhead(overhead_raw)
        if detected_positions:
            visual_positions.update(detected_positions)
        last_metrics = {
            "ANYmal": m1,
            "Husky": m2,
        }
        topdown = draw_topdown(robots, step, assignments_text, replans, avoided, last_metrics, visual_positions=visual_positions)
        dashboard = compose_dashboard(topdown, anymal_ann, husky_ann, overhead_ann)

        if video is not None:
            video.write(dashboard)

        if live:
            cv2.imshow(WINDOW_NAME, dashboard)
            key = cv2.waitKey(int(1000 / fps)) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord(" "):
                paused = not paused
        else:
            if step % 20 == 0:
                print(f"step={step} replans={replans} avoided={avoided} metrics={last_metrics}")

        if not paused:
            step += 1

    if video is not None:
        video.release()
    if live:
        cv2.destroyAllWindows()

    final_png = OUT / "final_dashboard.png"
    cv2.imwrite(str(final_png), dashboard)
    print("Saved:", out_path if save_video else "video disabled")
    print("Saved:", final_png)
    print("Final metrics:", {"steps": step, "replans": replans, "avoided_collisions": avoided})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="Do not open a live OpenCV window; only render/save video.")
    parser.add_argument("--no-video", action="store_true", help="Do not save mp4 output.")
    parser.add_argument("--steps", type=int, default=MAX_STEPS)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--speed", type=float, default=DEFAULT_LINEAR_SPEED,
                        help="Robot linear speed in grid cells per frame. Smaller is slower.")
    parser.add_argument("--turn-rate", type=float, default=DEFAULT_TURN_RATE,
                        help="Robot angular speed in radians per frame. Smaller makes end-of-line turns take longer.")
    parser.add_argument("--ml", choices=["rule", "gb", "xgb"], default="gb",
                        help="Steering predictor: rule, gb=Gradient Boosting, xgb=XGBoost.")
    args = parser.parse_args()
    run(live=not args.headless, save_video=not args.no_video, max_steps=args.steps, fps=args.fps, ml_method=args.ml, linear_speed=args.speed, turn_rate=args.turn_rate)


if __name__ == "__main__":
    main()
