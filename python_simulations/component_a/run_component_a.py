#!/usr/bin/env python3
"""
Component A – Offline reproducible simulation
Perception (LiDAR+camera fusion), Dijkstra vs A*, PID path tracking.
Produces all required plots under ../../plots/
"""
import os, time, heapq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "plots")
os.makedirs(OUT, exist_ok=True)
np.random.seed(42)

WORLD, CELL = 10.0, 0.2
GS = int(WORLD / CELL)  # 50

# ---- Environment ----
OBSTACLES = [(2,2,1.5,3),(5.5,1,1.2,2.5),(3,6,3,1),(7,5.5,1.5,2),(1,7.5,2,1.2)]
START, GOAL = (0.5, 0.5), (9.0, 9.0)

def occupied(x, y, r=0.18):
    if x < 0 or y < 0 or x > WORLD or y > WORLD:
        return True
    for ox, oy, w, h in OBSTACLES:
        if ox - r < x < ox + w + r and oy - r < y < oy + h + r:
            return True
    return False

def build_grid():
    g = np.zeros((GS, GS), dtype=np.uint8)
    for i in range(GS):
        for j in range(GS):
            if occupied(j*CELL+CELL/2, i*CELL+CELL/2, 0.2):
                g[i, j] = 1
    return g

# ---- Sensors ----
def lidar_scan(pose, n=180, max_r=4.0):
    x, y, th = pose
    ranges = np.full(n, max_r)
    angs = np.linspace(-np.pi, np.pi, n, endpoint=False)
    for i, a in enumerate(angs):
        for r in np.linspace(0.05, max_r, 80):
            if occupied(x + r*np.cos(th+a), y + r*np.sin(th+a), 0.05):
                ranges[i] = r
                break
    return angs, ranges

def camera_img(pose, sz=128):
    img = np.ones((sz, sz), np.uint8) * 255
    sc = sz / WORLD
    for ox, oy, w, h in OBSTACLES:
        img[int(oy*sc):int((oy+h)*sc), int(ox*sc):int((ox+w)*sc)] = 0
    rx, ry = int(pose[0]*sc), int(pose[1]*sc)
    if 0 <= rx < sz and 0 <= ry < sz:
        img[max(0,ry-2):ry+3, max(0,rx-2):rx+3] = 128
    return img

def fuse(angs, ranges, img, pose):
    grid = np.full((GS, GS), 0.5)
    x, y, th = pose
    for a, rng in zip(angs, ranges):
        if rng >= 3.9:
            continue
        for r in np.linspace(0, rng, 25):
            fx, fy = x + r*np.cos(th+a), y + r*np.sin(th+a)
            gi, gj = int(fy/CELL), int(fx/CELL)
            if 0 <= gi < GS and 0 <= gj < GS:
                grid[gi, gj] = 1.0 if r > rng - 0.15 else 0.0
    sc = GS / img.shape[0]
    for i in range(GS):
        for j in range(GS):
            if img[int(i/sc), int(j/sc)] < 50:
                grid[i, j] = 1.0
    return grid

# ---- Path planning ----
def neighbours(r, c, grid):
    out = []
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
        nr, nc = r+dr, c+dc
        if 0 <= nr < GS and 0 <= nc < GS and grid[nr, nc] == 0:
            out.append((nr, nc, 1.414 if dr and dc else 1.0))
    return out

def dijkstra(grid, start, goal):
    openq, came, gscore, explored = [], {}, {start: 0.0}, 0
    heapq.heappush(openq, (0.0, start))
    while openq:
        _, cur = heapq.heappop(openq)
        explored += 1
        if cur == goal:
            break
        for nr, nc, cost in neighbours(*cur, grid):
            ng = gscore[cur] + cost
            if ng < gscore.get((nr, nc), 1e18):
                gscore[(nr, nc)] = ng
                came[(nr, nc)] = cur
                heapq.heappush(openq, (ng, (nr, nc)))
    path = []
    node = goal if goal in came or goal == start else None
    while node is not None:
        path.append(node)
        node = came.get(node)
    return path[::-1], explored

def astar(grid, start, goal):
    def h(a, b):
        return np.hypot(a[0]-b[0], a[1]-b[1])
    openq, came, gscore, explored = [], {}, {start: 0.0}, 0
    heapq.heappush(openq, (h(start, goal), start))
    while openq:
        _, cur = heapq.heappop(openq)
        explored += 1
        if cur == goal:
            break
        for nr, nc, cost in neighbours(*cur, grid):
            ng = gscore[cur] + cost
            if ng < gscore.get((nr, nc), 1e18):
                gscore[(nr, nc)] = ng
                came[(nr, nc)] = cur
                heapq.heappush(openq, (ng + h((nr, nc), goal), (nr, nc)))
    path = []
    node = goal if goal in came or goal == start else None
    while node is not None:
        path.append(node)
        node = came.get(node)
    return path[::-1], explored

def to_world(path):
    return np.array([[c*CELL+CELL/2, r*CELL+CELL/2] for r, c in path])

# ---- PID tracking ----
class Robot:
    def __init__(self, x, y, th=0.0):
        self.x, self.y, self.th = x, y, th
        self.traj = [(x, y, th)]
    def step(self, v, w, dt=0.05):
        self.x += v * np.cos(self.th) * dt
        self.y += v * np.sin(self.th) * dt
        self.th = (self.th + w*dt + np.pi) % (2*np.pi) - np.pi
        self.traj.append((self.x, self.y, self.th))

def track(path_w, Kp=2.5, Ki=0.3, Kd=0.8, steps=2500):
    robot = Robot(*START)
    integral = prev = 0.0
    errors, t = [], 0.0
    idx = 0
    for _ in range(steps):
        while idx < len(path_w)-1 and np.hypot(path_w[idx,0]-robot.x, path_w[idx,1]-robot.y) < 0.4:
            idx += 1
        tx, ty = path_w[min(idx, len(path_w)-1)]
        des = np.arctan2(ty-robot.y, tx-robot.x)
        err = (des - robot.th + np.pi) % (2*np.pi) - np.pi
        integral += err * 0.05
        deriv = (err - prev) / 0.05
        w = (Kp*err + Ki*integral + Kd*deriv)
        v = 0.45 * (1 - min(abs(err)/np.pi, 1)) + 0.15
        robot.step(v, w)
        errors.append((t, err))
        t += 0.05
        prev = err
        if np.hypot(robot.x-path_w[-1,0], robot.y-path_w[-1,1]) < 0.25:
            break
    return robot, errors

def draw_env(ax, title=""):
    ax.set_xlim(0, WORLD); ax.set_ylim(0, WORLD); ax.set_aspect("equal")
    for ox, oy, w, h in OBSTACLES:
        ax.add_patch(Rectangle((ox, oy), w, h, color="0.35", alpha=0.85))
    ax.plot(*START, "go", ms=10, label="Start")
    ax.plot(*GOAL, "r*", ms=14, label="Goal")
    ax.set_title(title); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.legend(loc="upper left", fontsize=8)

def main():
    print("=== Component A offline simulation ===")
    grid = build_grid()
    sc = (int(START[0]/CELL), int(START[1]/CELL))
    gc = (int(GOAL[0]/CELL), int(GOAL[1]/CELL))

    # Perception
    pose = (*START, 0.0)
    angs, ranges = lidar_scan(pose)
    img = camera_img(pose)
    fused = fuse(angs, ranges, img, pose)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    ax[0].imshow(img, cmap="gray", origin="lower"); ax[0].set_title("Camera"); ax[0].axis("off")
    ax[1].plot(angs, ranges, ".", ms=2); ax[1].set_title("LiDAR ranges"); ax[1].set_xlabel("angle")
    ax[2].imshow(fused, cmap="RdYlGn_r", origin="lower", vmin=0, vmax=1); ax[2].set_title("Fused occupancy"); ax[2].axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "A1_sensor_fusion.png"), dpi=140); plt.close()
    print("[A1] sensor fusion saved")

    # Planning
    t0 = time.perf_counter(); pd, ed = dijkstra(grid, sc, gc); td = time.perf_counter()-t0
    t0 = time.perf_counter(); pa, ea = astar(grid, sc, gc); ta = time.perf_counter()-t0
    wd, wa = to_world(pd), to_world(pa)
    print(f"Dijkstra: {len(pd)} cells, {ed} explored, {td*1000:.1f} ms")
    print(f"A*:       {len(pa)} cells, {ea} explored, {ta*1000:.1f} ms")

    fig, ax = plt.subplots(figsize=(7, 7))
    draw_env(ax, "Dijkstra vs A*")
    if len(wd): ax.plot(wd[:,0], wd[:,1], "b-", lw=2, label=f"Dijkstra ({len(pd)})")
    if len(wa): ax.plot(wa[:,0], wa[:,1], "m--", lw=2, label=f"A* ({len(pa)})")
    ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "A2_path_comparison.png"), dpi=140); plt.close()

    # PID tuning
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    for ax, (name, kp, ki, kd) in zip(axes, [
        ("Under-damped", 4.5, 0.0, 0.1),
        ("Well-tuned", 2.5, 0.3, 0.8),
        ("Over-damped", 1.2, 0.1, 2.8),
    ]):
        _, errs = track(wa, kp, ki, kd)
        ax.plot([e[0] for e in errs], [e[1] for e in errs])
        ax.set_title(name); ax.set_xlabel("t (s)"); ax.axhline(0, color="k", ls="--", lw=0.5); ax.grid(alpha=0.3)
    axes[0].set_ylabel("heading error (rad)")
    plt.suptitle("PID tuning – heading error"); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "A3_pid_tuning.png"), dpi=140); plt.close()

    # Closed-loop
    robot, errs = track(wa, 2.5, 0.3, 0.8)
    traj = np.array(robot.traj)
    fig, ax = plt.subplots(figsize=(7, 7))
    draw_env(ax, "Planned vs actual trajectory")
    ax.plot(wa[:,0], wa[:,1], "m--", lw=2, label="Planned (A*)")
    ax.plot(traj[:,0], traj[:,1], "b-", lw=1.5, label="Actual")
    ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "A4_trajectory.png"), dpi=140); plt.close()

    plen = np.sum(np.linalg.norm(np.diff(wa, axis=0), axis=1))
    alen = np.sum(np.linalg.norm(np.diff(traj[:,:2], axis=0), axis=1))
    metrics = {
        "planned_m": round(plen, 3), "actual_m": round(alen, 3),
        "time_s": round(errs[-1][0], 2) if errs else 0,
        "mean_heading_err": round(float(np.mean(np.abs([e[1] for e in errs]))), 4),
        "dijkstra_nodes": ed, "astar_nodes": ea,
        "dijkstra_ms": round(td*1000, 2), "astar_ms": round(ta*1000, 2),
    }
    with open(os.path.join(OUT, "A_metrics.txt"), "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
            print(f"  {k}: {v}")
    print("Component A complete.")

if __name__ == "__main__":
    main()
