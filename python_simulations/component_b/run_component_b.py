#!/usr/bin/env python3
"""
Component B – Offline multi-agent swarm + ML decision-making
Average consensus, PSO coverage, neural navigation policy.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "plots")
os.makedirs(OUT, exist_ok=True)
np.random.seed(42)
torch.manual_seed(42)

WORLD, N = 10.0, 5

class Agent:
    def __init__(self, i, x, y):
        self.id, self.x, self.y = i, x, y
        self.state = np.array([x, y], float)
        self.traj = [(x, y)]
    def move_toward(self, tx, ty, gain=3.0, dt=0.05, vmax=0.55):
        dx, dy = tx - self.x, ty - self.y
        d = np.hypot(dx, dy) + 1e-9
        v = min(vmax, gain * d * dt)
        self.x = np.clip(self.x + v * dx / d, 0.3, WORLD-0.3)
        self.y = np.clip(self.y + v * dy / d, 0.3, WORLD-0.3)
        self.state = np.array([self.x, self.y])
        self.traj.append((self.x, self.y))

def consensus_step(agents, eps=0.12):
    states = np.array([a.state for a in agents])
    new = states.copy()
    n = len(agents)
    for i in range(n):
        for j in range(n):
            if i != j:
                new[i] += eps * (states[j] - states[i])
    for i, a in enumerate(agents):
        a.state = new[i]
        a.x = 0.65*a.x + 0.35*new[i,0]
        a.y = 0.65*a.y + 0.35*new[i,1]
        a.traj.append((a.x, a.y))

def run_consensus(steps=90):
    agents = [Agent(i, *np.random.uniform(0.5, 3.5, 2)) for i in range(N)]
    hist = []
    for _ in range(steps):
        consensus_step(agents)
        hist.append(np.array([a.state.copy() for a in agents]))
    return agents, np.array(hist)

class PSO:
    def __init__(self, n, bounds=(0.5, 9.5)):
        lo, hi = bounds
        self.pos = np.random.uniform(lo, hi, (n, 2))
        self.vel = np.zeros((n, 2))
        self.pbest = self.pos.copy()
        self.pbest_val = np.full(n, -np.inf)
        self.gbest = self.pos[0].copy()
        self.gbest_val = -np.inf
    def step(self, targets, covered):
        for i in range(len(self.pos)):
            score = 0.0
            for k, t in enumerate(targets):
                if not covered[k]:
                    score -= np.min(np.linalg.norm(self.pos[i:i+1] - t, axis=1))
            if score > self.pbest_val[i]:
                self.pbest_val[i] = score
                self.pbest[i] = self.pos[i].copy()
        bi = np.argmax(self.pbest_val)
        if self.pbest_val[bi] > self.gbest_val:
            self.gbest_val = self.pbest_val[bi]
            self.gbest = self.pbest[bi].copy()
        r1, r2 = np.random.rand(len(self.pos), 2), np.random.rand(len(self.pos), 2)
        self.vel = 0.65*self.vel + 1.5*r1*(self.pbest-self.pos) + 1.5*r2*(self.gbest-self.pos)
        self.pos = np.clip(self.pos + 0.12*self.vel, 0.3, WORLD-0.3)

def run_pso(iterations=150):
    agents = [Agent(i, 1.0 + 0.4*i, 1.0 + 0.3*i) for i in range(N)]
    targets = np.array([[2.0,7.5],[7.5,2.0],[8.0,8.0],[3.5,5.0],
                        [6.5,6.5],[4.0,3.0],[7.0,5.0],[5.0,8.5]], float)
    covered = np.zeros(len(targets), dtype=bool)
    # assign targets round-robin
    assign = {i: [] for i in range(N)}
    for k in range(len(targets)):
        assign[k % N].append(k)
    ptr = {i: 0 for i in range(N)}
    cov_hist = []
    for it in range(iterations):
        for i, a in enumerate(agents):
            my = assign[i]
            # advance pointer if current target covered
            while ptr[i] < len(my) and covered[my[ptr[i]]]:
                ptr[i] += 1
            if ptr[i] < len(my):
                t = targets[my[ptr[i]]]
                a.move_toward(t[0], t[1], gain=8.0, vmax=1.0)
            for k, t in enumerate(targets):
                if not covered[k] and np.hypot(a.x-t[0], a.y-t[1]) < 1.2:
                    covered[k] = True
        cov_hist.append(covered.mean())
    return agents, targets, covered, cov_hist

class NavNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 5)
        )
    def forward(self, x):
        return self.net(x)

def make_dataset(n=2800):
    X, y = [], []
    for _ in range(n):
        df, dl, dr = np.random.uniform(0.1, 3, 3)
        gdx, gdy = np.random.uniform(-2, 2, 2)
        herr = np.random.uniform(-np.pi, np.pi)
        feat = [df, dl, dr, gdx, gdy, herr]
        if df < 0.5:
            act = 4 if df < 0.25 else 3
        elif abs(herr) > 0.6:
            act = 1 if herr > 0 else 2
        elif dl < 0.4:
            act = 2
        elif dr < 0.4:
            act = 1
        else:
            act = 0
        X.append(feat); y.append(act)
    return np.array(X, np.float32), np.array(y, np.int64)

def train_nn(epochs=35):
    X, y = make_dataset()
    idx = np.random.permutation(len(X))
    sp = int(0.8*len(X))
    Xt, yt = X[idx[:sp]], y[idx[:sp]]
    Xv, yv = X[idx[sp:]], y[idx[sp:]]
    model = NavNet()
    opt = optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(Xt), torch.from_numpy(yt)),
                        batch_size=64, shuffle=True)
    losses = []
    for _ in range(epochs):
        el = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward(); opt.step()
            el += loss.item()
        losses.append(el/len(loader))
    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(Xv)).argmax(1).numpy()
    acc = (preds == yv).mean()
    # confusion
    cm = np.zeros((5, 5), int)
    for t, p in zip(yv, preds):
        cm[t, p] += 1
    return model, losses, acc, cm

def main():
    print("=== Component B offline simulation ===")
    # Consensus
    agents, hist = run_consensus()
    varx = hist[:,:,0].var(1); vary = hist[:,:,1].var(1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(varx, label="Var(x)"); axes[0].plot(vary, label="Var(y)")
    axes[0].set_title("Average consensus – variance"); axes[0].set_xlabel("iter")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    for a in agents:
        tr = np.array(a.traj)
        axes[1].plot(tr[:,0], tr[:,1], alpha=0.7)
        axes[1].plot(tr[-1,0], tr[-1,1], "o")
    axes[1].set_title("Agent trajectories"); axes[1].set_xlim(0,WORLD); axes[1].set_ylim(0,WORLD)
    axes[1].set_aspect("equal")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "B1_consensus.png"), dpi=140); plt.close()
    print("[B1] consensus saved")

    # PSO
    agents, targets, covered, cov_hist = run_pso()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(cov_hist); axes[0].set_title("PSO coverage progress")
    axes[0].set_xlabel("iter"); axes[0].set_ylabel("fraction covered"); axes[0].set_ylim(0,1.05)
    axes[0].grid(alpha=0.3)
    for a in agents:
        tr = np.array(a.traj)
        axes[1].plot(tr[:,0], tr[:,1], alpha=0.5)
        axes[1].plot(a.x, a.y, "bo")
    for i, t in enumerate(targets):
        axes[1].plot(t[0], t[1], "*", color="g" if covered[i] else "r", ms=12)
    axes[1].set_title("Final swarm (green*=covered)"); axes[1].set_xlim(0,WORLD); axes[1].set_ylim(0,WORLD)
    axes[1].set_aspect("equal")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "B2_pso_coverage.png"), dpi=140); plt.close()
    print(f"[B2] coverage {cov_hist[-1]*100:.0f}%")

    # NN
    model, losses, acc, cm = train_nn()
    print(f"[B3] NN accuracy {acc*100:.1f}%")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(losses); axes[0].set_title("Training loss"); axes[0].set_xlabel("epoch"); axes[0].grid(alpha=0.3)
    im = axes[1].imshow(cm, cmap="Blues")
    axes[1].set_title(f"Confusion matrix (acc={acc*100:.1f}%)")
    labels = ["Fwd","Left","Right","Slow","Stop"]
    axes[1].set_xticks(range(5)); axes[1].set_yticks(range(5))
    axes[1].set_xticklabels(labels); axes[1].set_yticklabels(labels)
    for i in range(5):
        for j in range(5):
            axes[1].text(j, i, cm[i,j], ha="center", va="center",
                         color="white" if cm[i,j] > cm.max()/2 else "black", fontsize=8)
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "B3_nn_performance.png"), dpi=140); plt.close()

    with open(os.path.join(OUT, "B_metrics.txt"), "w") as f:
        f.write(f"coverage: {cov_hist[-1]:.3f}\n")
        f.write(f"nn_accuracy: {acc:.4f}\n")
        f.write(f"consensus_final_var_x: {varx[-1]:.6f}\n")
        f.write(f"consensus_final_var_y: {vary[-1]:.6f}\n")
    print("Component B complete.")

if __name__ == "__main__":
    main()
