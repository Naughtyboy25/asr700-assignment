# ASR700 Assignment - Comprehensive Project Plan

**Module:** Autonomous Systems and Robotics 700
**Institution:** Richfield Graduate Institute of Technology
**Platform:** Webots 2025a (mandatory), Python with Matplotlib/Plotly/Seaborn
**Package Manager:** UV
**Version Control:** Git + GitHub CLI (gh) - Public repo, minimum 10 commits

---

## 1. Assignment Overview

### Component A (50 Marks): Single Robot Autonomous Navigation
| Section | Marks | Key Requirements |
|---------|-------|------------------|
| 1. Sensor Simulation & Perception Pipeline | 10 | LiDAR + Camera simulation, OpenCV processing, Sensor fusion (2+ modalities) |
| 2. Motion Planning & Pathfinding | 15 | Occupancy grid, Dijkstra, A* with heuristic, Comparison metrics, Visualization |
| 3. Control System Design | 15 | Kinematic model, PID controller, Gain tuning, Response curves, Stability analysis |
| 4. System Integration & Evaluation | 10 | Closed-loop integration, Planned vs actual trajectory, Performance table, Ethics discussion |

### Component B (50 Marks): Multi-Agent Swarm with Learning
| Section | Marks | Key Requirements |
|---------|-------|------------------|
| 1. Multi-Agent Simulation Setup | 10 | 3+ robots in Webots, Roles, Communication topology, Cooperative task |
| 2. Consensus & Swarm Intelligence | 10 | Consensus algorithm, PSO or ACO, Emergent behavior, Convergence evaluation |
| 3. ML for Navigation/Decision-Making | 10 | Neural network training, Metrics (accuracy, success rate, reward), Confusion matrix, Centralized vs distributed comparison |
| 4. Ethics, Risk, Responsible Autonomy | 20 | Bias/reliability, Explainability, Safety-critical risks, Fault tolerance, SA context |

### Deliverables (All in single ZIP ≤ 50MB)
1. **Simulation/Code Package** - Webots project + Python/.ipynb scripts
2. **Professional PDF Report** (3-9 pages) - Introduction, Methodology, Results, Discussion/Ethics, Conclusion
3. **Simulation Environment/Dataset** - Webots worlds, maps, sensor data
4. **README.txt** - Run instructions, versions, dependencies, runtime, hardware

---

## 2. Technical Architecture

### 2.1 Repository Structure
```
ASR700_Assignment/
├── .github/
│   └── workflows/
│       └── ci.yml                 # Optional: basic CI checks
├── webots_project/
│   ├── worlds/
│   │   ├── component_a.wbt        # Single robot world
│   │   └── component_b.wbt        # Multi-robot swarm world
│   ├── controllers/
│   │   ├── robot_controller_a/    # Component A controller
│   │   │   ├── robot_controller_a.py
│   │   │   ├── perception.py      # Sensor processing & fusion
│   │   │   ├── planning.py        # Dijkstra & A*
│   │   │   ├── control.py         # PID controller
│   │   │   └── kinematics.py      # Robot kinematic model
│   │   └── swarm_agents/          # Component B controllers
│   │       ├── leader_controller.py
│   │       ├── follower_controller.py
│   │       ├── consensus.py       # Consensus algorithm
│   │       ├── swarm_intelligence.py  # PSO/ACO
│   │       └── ml_policy.py       # Neural network policy
│   ├── protos/
│   │   └── custom_robot.proto     # Custom robot definition if needed
│   └── resources/
│       ├── maps/
│       │   ├── occupancy_grid.npy
│       │   └── map.png
│       └── models/
├── python_analysis/
│   ├── component_a_analysis.ipynb  # Matplotlib visualizations for A
│   ├── component_b_analysis.ipynb  # Matplotlib visualizations for B
│   ├── training/
│   │   ├── train_nn.py            # Neural network training script
│   │   ├── model.py               # PyTorch/TensorFlow model definition
│   │   └── dataset_gen.py         # Training data generation
│   ├── evaluation/
│   │   ├── metrics.py             # Performance metrics
│   │   ├── plotting.py            # All Matplotlib/Seaborn/Plotly plots
│   │   └── comparison.py          # Algorithm comparisons
│   └── utils/
│       ├── config.py              # Configuration constants
│       └── helpers.py             # Utility functions
├── report/
│   ├── report.tex                 # LaTeX source for PDF report
│   ├── figures/                   # Generated figures for report
│   └── bibliography.bib           # Harvard references
├── data/
│   ├── sensor_logs/               # Recorded sensor data
│   ├── trajectories/              # Planned vs actual paths
│   └── ml_datasets/               # Training/validation datasets
├── tests/
│   ├── test_planning.py           # Unit tests for planners
│   ├── test_control.py            # Unit tests for PID
│   └── test_perception.py         # Unit tests for perception
├── README.txt                     # Mandatory run instructions
├── requirements.txt               # UV dependencies (generated from pyproject.toml)
├── pyproject.toml                 # UV project configuration
├── uv.lock                        # Locked dependencies
├── .gitignore
└── LICENSE
```

### 2.2 Technology Stack
| Category | Technology | Version/Notes |
|----------|------------|---------------|
| Simulation | Webots | 2025a (mandatory) |
| Language | Python | 3.10+ (Webots 2025a compatible) |
| Package Manager | UV | Latest |
| Visualization | Matplotlib, Seaborn, Plotly | As required |
| Computer Vision | OpenCV (cv2) | For image processing |
| ML Framework | PyTorch or TensorFlow | For neural network |
| Numerical | NumPy, SciPy | Core computations |
| Path Planning | Custom implementation | Dijkstra, A* (no ROS) |
| Control | Custom PID implementation | |
| Version Control | Git + GitHub CLI | Public repo, 10+ commits |

### 2.3 Key Design Decisions

#### Component A - Robot Selection
- **Robot:** e-puck or Thymio II (built-in Webots models with LiDAR + camera)
- **Environment:** 10×10m bounded arena with static obstacles
- **Sensors:** LiDAR (270° FOV, 0.05m resolution) + Camera (640×480)
- **Sensor Fusion:** Occupancy grid from LiDAR + semantic features from camera (color landmarks)
- **Kinematics:** Differential drive model (unicycle approximation)

#### Component B - Swarm Configuration
- **Robots:** 4× e-puck robots (1 leader, 3 followers)
- **Task:** Distributed area coverage with cooperative search
- **Communication:** Broadcast within 2m range (simulated via supervisor)
- **Consensus:** Average consensus for position agreement
- **Swarm Intelligence:** Particle Swarm Optimization for coverage waypoints
- **ML:** Small CNN for obstacle avoidance policy (trained offline, deployed online)

---

## 3. UV Package Management Setup

### 3.1 pyproject.toml
```toml
[project]
name = "asr700-assignment"
version = "0.1.0"
description = "ASR700 Autonomous Systems and Robotics Assignment"
readme = "README.txt"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.26.0",
    "scipy>=1.11.0",
    "matplotlib>=3.8.0",
    "seaborn>=0.13.0",
    "plotly>=5.18.0",
    "opencv-python>=4.8.0",
    "pillow>=10.0.0",
    "pyyaml>=6.0",
    "tqdm>=4.66.0",
]

[project.optional-dependencies]
ml = [
    "torch>=2.1.0",
    "torchvision>=0.16.0",
]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
    "mypy>=1.6.0",
    "jupyter>=1.0.0",
    "ipykernel>=6.25.0",
]
docs = [
    "sphinx>=7.2.0",
    "sphinx-rtd-theme>=1.3.0",
]

[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[tool.uv]
dev-dependencies = [
    "pytest>=7.4.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
]

[tool.black]
line-length = 100
target-version = ['py310']

[tool.ruff]
line-length = 100
target-version = "py310"
```

### 3.2 UV Commands
```bash
# Initialize project
uv init --name asr700-assignment
cd asr700-assignment

# Add dependencies
uv add numpy scipy matplotlib seaborn plotly opencv-python pillow pyyaml tqdm
uv add --dev pytest pytest-cov black ruff mypy jupyter ipykernel
uv add --optional ml torch torchvision

# Install in editable mode
uv pip install -e .

# Generate requirements.txt for README
uv pip freeze > requirements.txt

# Run scripts
uv run python python_analysis/training/train_nn.py
uv run jupyter notebook python_analysis/component_a_analysis.ipynb
```

---

## 4. Git & GitHub Strategy

### 4.1 Repository Setup
```bash
# Create public GitHub repo
gh repo create asr700-assignment --public --description "ASR700 Autonomous Systems and Robotics Assignment - Webots simulation with Python analysis"

# Clone locally
gh repo clone <username>/asr700-assignment
cd asr700-assignment

# Initialize UV project
uv init --name asr700-assignment
```

### 4.2 Commit Strategy (Minimum 10 Meaningful Commits)
| Commit # | Message | Scope |
|----------|---------|-------|
| 1 | `feat: initialize UV project with dependencies and project structure` | Setup |
| 2 | `feat: create Webots world for Component A with e-puck robot and obstacles` | Component A - World |
| 3 | `feat: implement sensor perception pipeline (LiDAR + camera fusion)` | Component A - Perception |
| 4 | `feat: implement Dijkstra and A* path planners with visualization` | Component A - Planning |
| 5 | `feat: implement differential drive kinematics and PID controller` | Component A - Control |
| 6 | `feat: integrate perception, planning, control into closed-loop navigation` | Component A - Integration |
| 7 | `feat: create Webots multi-robot world for Component B (4 e-pucks)` | Component B - World |
| 8 | `feat: implement consensus algorithm and PSO for swarm coordination` | Component B - Swarm |
| 9 | `feat: train and evaluate neural network for obstacle avoidance` | Component B - ML |
| 10 | `feat: generate all Matplotlib visualizations and performance metrics` | Analysis/Visualization |
| 11 | `docs: write professional PDF report with Harvard references` | Report |
| 12 | `docs: add README.txt with run instructions and requirements` | Documentation |
| 13 | `chore: final cleanup, verify all deliverables, zip submission package` | Final |

### 4.3 Branching Strategy
- `main` - Protected, only merge via PR
- `feature/component-a-*` - Component A features
- `feature/component-b-*` - Component B features
- `feature/report` - Report writing
- Delete branches after merge

---

## 5. Development Timeline

### Phase 1: Foundation (Week 1)
- [ ] Repository setup, UV initialization, GitHub repo creation
- [ ] Webots 2025a installation verification
- [ ] Project structure creation
- [ ] Basic Webots world for Component A
- [ ] **Target: 2 commits**

### Phase 2: Component A - Perception & Planning (Week 2)
- [ ] LiDAR simulation and data extraction
- [ ] Camera image processing with OpenCV
- [ ] Sensor fusion → occupancy grid
- [ ] Occupancy grid representation
- [ ] Dijkstra implementation + visualization
- [ ] A* implementation with Euclidean heuristic + visualization
- [ ] Algorithm comparison (nodes explored, path length, compute time)
- [ ] **Target: 3 commits**

### Phase 3: Component A - Control & Integration (Week 3)
- [ ] Differential drive kinematic model derivation
- [ ] PID controller implementation
- [ ] PID tuning (Ziegler-Nichols + manual refinement)
- [ ] Response curve generation (overshoot, rise time, settling time, SSE)
- [ ] Closed-loop integration in Webots controller
- [ ] Trajectory comparison plots (planned vs actual)
- [ ] Performance evaluation table
- [ ] **Target: 2 commits**

### Phase 4: Component B - Multi-Agent Swarm (Week 4)
- [ ] Multi-robot Webots world (4 e-pucks)
- [ ] Supervisor-based communication topology
- [ ] Role definitions (leader/followers)
- [ ] Average consensus implementation
- [ ] PSO for coverage waypoint optimization
- [ ] Emergent behavior visualization
- [ ] Convergence analysis plots
- [ ] **Target: 2 commits**

### Phase 5: Component B - Machine Learning (Week 5)
- [ ] Dataset generation from Webots simulations
- [ ] CNN architecture for obstacle avoidance
- [ ] Training loop with validation
- [ ] Model evaluation (accuracy, success rate, reward curves)
- [ ] Confusion matrix generation
- [ ] Centralized vs distributed decision comparison
- [ ] Model deployment to Webots controllers
- [ ] **Target: 2 commits**

### Phase 6: Report & Finalization (Week 6)
- [ ] Professional PDF report (LaTeX → PDF)
- [ ] All figures generated and embedded
- [ ] Harvard referencing verification
- [ ] README.txt with complete instructions
- [ ] Final testing of all simulations
- [ ] ZIP package creation (< 50MB)
- [ ] **Target: 2 commits**

---

## 6. Detailed Implementation Checklist

### Component A Checklist
#### Sensor Perception (10 marks)
- [ ] Webots LiDAR enabled, range data extracted each timestep
- [ ] Webots camera enabled, images captured each timestep
- [ ] OpenCV: Canny edge detection / color segmentation for landmarks
- [ ] LiDAR → occupancy grid (log-odds update)
- [ ] Camera → semantic landmarks (color-coded cylinders)
- [ ] Fusion: Occupancy grid + landmark positions → unified map
- [ ] Design justification documented in code comments and report

#### Motion Planning (15 marks)
- [ ] Occupancy grid class with inflation radius
- [ ] Graph construction (4-connected or 8-connected)
- [ ] Dijkstra: priority queue, visited set, path reconstruction
- [ ] A*: g-cost, h-cost (Euclidean), f-cost, open/closed sets
- [ ] Visualization: grid, explored nodes, final path (Matplotlib)
- [ ] Comparison table: nodes expanded, path length, computation time
- [ ] Analysis of heuristic effectiveness

#### Control System (15 marks)
- [ ] Kinematic model: dx/dt = v cos(θ), dy/dt = v sin(θ), dθ/dt = ω
- [ ] PID for linear velocity (distance error)
- [ ] PID for angular velocity (heading error)
- [ ] Gain tuning procedure documented
- [ ] Step response plots for Kp, Ki, Kd variations
- [ ] Metrics: overshoot %, rise time, settling time, steady-state error
- [ ] Stability discussion (root locus / frequency domain concepts)

#### Integration (10 marks)
- [ ] Main control loop: perceive → plan → control → execute
- [ ] Replanning trigger (obstacle detected, large tracking error)
- [ ] Trajectory logging (planned waypoints, actual positions)
- [ ] Plot: planned path vs actual trajectory
- [ ] Table: path length, time-to-goal, max tracking error, RMSE
- [ ] Trade-off analysis: perception noise → planning errors → control effort
- [ ] Ethics paragraph: shared human environments

### Component B Checklist
#### Multi-Agent Setup (10 marks)
- [ ] World file with 4 e-pucks at distinct start positions
- [ ] Supervisor controller for communication simulation
- [ ] Roles: 1 leader (explorer), 3 followers (coverage)
- [ ] Communication: broadcast within radius, packet loss simulation
- [ ] Task: distributed coverage of 10×10m area
- [ ] Justification in report: why this topology, why these roles

#### Consensus & Swarm Intelligence (10 marks)
- [ ] Average consensus: x_i(t+1) = x_i(t) + α Σ(x_j - x_i)
- [ ] Convergence proof sketch / citation
- [ ] PSO: particles = robots, fitness = coverage metric
- [ ] PSO parameters: w, c1, c2 tuned
- [ ] Emergent behavior: visualize agent trajectories over time
- [ ] Convergence plots: consensus error vs time, PSO fitness vs iteration

#### Machine Learning (10 marks)
- [ ] Dataset: (LiDAR scan, camera) → action (v, ω)
- [ ] CNN: 2-3 conv layers, FC layers, output 2 continuous values
- [ ] Training: MSE loss, Adam optimizer, early stopping
- [ ] Metrics: test MSE, success rate in simulation, avg reward
- [ ] Confusion matrix: discretized actions (forward, left, right, stop)
- [ ] Centralized (supervisor decides) vs Distributed (local NN) comparison

#### Ethics (20 marks)
- [ ] Sensor bias: LiDAR specular reflection, camera lighting
- [ ] Simulation assumptions: perfect comms, no actuator noise
- [ ] NN explainability: saliency maps, decision boundaries
- [ ] Safety: emergency stop, human detection, fail-safe behaviors
- [ ] Fault tolerance: agent dropout, communication loss, sensor failure
- [ ] SA context: unemployment, inequality, regulation, cultural factors

---

## 7. Report Structure (LaTeX Template)

```latex
\documentclass[12pt,a4paper]{article}
\usepackage{times}
\usepackage{setspace}
\onehalfspacing
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{natbib}  % Harvard style

\title{Intelligent Autonomous Robotic Systems Project}
\author{Student Name \\ Student Number}
\date{\today}

\begin{document}
\maketitle

\tableofcontents
\newpage

\section{Introduction}
% Problem statement, objectives, scope

\section{Methodology}
\subsection{Component A: Single Robot Navigation}
\subsubsection{Sensor Perception Pipeline}
\subsubsection{Motion Planning Algorithms}
\subsubsection{Control System Design}
\subsubsection{System Integration}

\subsection{Component B: Multi-Agent Swarm}
\subsubsection{Multi-Agent Simulation Setup}
\subsubsection{Consensus and Swarm Intelligence}
\subsubsection{Machine Learning for Navigation}

\section{Results and Evaluation}
\subsection{Component A Results}
% Figures, tables, metrics
\subsection{Component B Results}
% Figures, tables, metrics

\section{Discussion and Ethical Considerations}
% Trade-offs, ethics, SA context

\section{Conclusion and Recommendations}

\bibliographystyle{agsm}  % Harvard
\bibliography{bibliography}

\end{document}
```

---

## 8. README.txt Template

```
ASR700 Assignment - Autonomous Systems and Robotics 700
=======================================================

Student: [Name]
Student Number: [Number]
Date: [Submission Date]

SOFTWARE VERSIONS
-----------------
Webots: 2025a (R2025a)
Python: 3.10.x (recommended 3.10 or 3.11)
UV: 0.4.x or later

PYTHON DEPENDENCIES
-------------------
See requirements.txt (generated via: uv pip freeze)
Key packages:
- numpy>=1.26.0
- scipy>=1.11.0
- matplotlib>=3.8.0
- seaborn>=0.13.0
- plotly>=5.18.0
- opencv-python>=4.8.0
- torch>=2.1.0 (for ML component)
- pillow>=10.0.0
- pyyaml>=6.0
- tqdm>=4.66.0

INSTALLATION
------------
1. Install Webots 2025a from https://cyberbotics.com/
2. Install UV: pip install uv (or via brew/choco)
3. Clone repository:
   git clone https://github.com/<username>/asr700-assignment.git
   cd asr700-assignment
4. Install Python dependencies:
   uv sync --all-extras
   # Or: uv pip install -e .[ml,dev]

RUNNING COMPONENT A (Single Robot Navigation)
---------------------------------------------
1. Open Webots 2025a
2. File > Open World > webots_project/worlds/component_a.wbt
3. Build controller: Select robot_controller_a > Build
4. Press Play to run simulation
5. For analysis plots:
   cd python_analysis
   uv run jupyter notebook component_a_analysis.ipynb
   Run all cells to generate visualizations

RUNNING COMPONENT B (Multi-Agent Swarm)
---------------------------------------
1. Open Webots 2025a
2. File > Open World > webots_project/worlds/component_b.wbt
3. Build all controllers: swarm_agents > Build
4. Press Play to run simulation
5. For analysis plots:
   cd python_analysis
   uv run jupyter notebook component_b_analysis.ipynb
   Run all cells to generate visualizations

TRAINING NEURAL NETWORK (Component B ML)
----------------------------------------
cd python_analysis/training
uv run python train_nn.py
# Trained model saved to webots_project/resources/models/

EXPECTED RUNTIME
----------------
Component A simulation: ~60-120 seconds
Component B simulation: ~120-180 seconds
Neural network training: ~10-30 minutes (CPU), ~2-5 minutes (GPU)
Analysis notebooks: ~30-60 seconds each

HARDWARE REQUIREMENTS
---------------------
- OS: Windows 11 (tested), Linux, or macOS
- RAM: 8 GB minimum, 16 GB recommended
- CPU: 4+ cores recommended
- GPU: Optional but recommended for ML training (CUDA compatible)
- Disk: 2 GB free space
- Display: Required for Webots GUI (headless mode available for CI)

TROUBLESHOOTING
---------------
- Webots controller build fails: Ensure Python path in Webots preferences matches uv environment
- Import errors: Run 'uv sync' to ensure all dependencies installed
- OpenCV errors: Ensure opencv-python-headless not used (need GUI for Webots)
- Memory issues: Reduce grid resolution or swarm size

CONTACT
-------
[Student Email]
GitHub: https://github.com/<username>/asr700-assignment
```

---

## 9. Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Webots 2025a compatibility issues | Medium | High | Test early, use built-in robot models |
| UV dependency conflicts | Low | Medium | Lock versions in uv.lock, test in clean env |
| Git commit history insufficient | Low | High | Commit daily, use conventional commits |
| ZIP file > 50MB | Medium | High | Exclude build folders, compress video, optimize images |
| Simulation non-determinism | Medium | Medium | Set random seeds, log all parameters |
| ML training fails to converge | Medium | Medium | Start simple, use proven architectures, monitor loss |
| Report formatting issues | Low | Medium | Use LaTeX template, compile early and often |

---

## 10. Quality Gates

### Before Each Commit
- [ ] Code runs without errors (`uv run python -m py_compile <file>`)
- [ ] Black formatting applied (`uv run black .`)
- [ ] Ruff linting passes (`uv run ruff check .`)
- [ ] Type hints added for new functions
- [ ] Docstrings for public functions/classes

### Before Each Merge to Main
- [ ] All tests pass (`uv run pytest tests/`)
- [ ] Simulation runs end-to-end in Webots
- [ ] Analysis notebooks execute without errors
- [ ] New visualizations generated and verified
- [ ] Commit message follows conventional format

### Final Submission Verification
- [ ] ZIP file < 50MB
- [ ] All 4 deliverables present
- [ ] Webots worlds open and run
- [ ] Python scripts execute via `uv run`
- [ ] Report PDF compiles, 3-9 pages, Harvard refs
- [ ] README.txt complete and accurate
- [ ] GitHub repo public, 10+ commits, descriptive messages
- [ ] No ROS/Gazebo references anywhere

---

## 11. References & Resources

### Core References (Harvard Style)
- Siegwart, R., Nourbakhsh, I.R. and Scaramuzza, D. (2011) *Introduction to Autonomous Mobile Robots*. 2nd edn. Cambridge, MA: MIT Press.
- Corke, P. (2017) *Robotics, Vision and Control: Fundamental Algorithms in MATLAB*. 2nd edn. Cham: Springer.
- Bullo, F., Cortés, J. and Martínez, S. (2009) *Distributed Control of Robotic Networks*. Princeton: Princeton University Press.
- Kennedy, J. and Eberhart, R. (1995) 'Particle swarm optimization', *Proceedings of ICNN'95*, 4, pp. 1942-1948.
- Dorigo, M. and Stützle, T. (2004) *Ant Colony Optimization*. Cambridge, MA: MIT Press.
- Åström, K.J. and Hägglund, T. (1995) *PID Controllers: Theory, Design, and Tuning*. 2nd edn. Research Triangle Park, NC: ISA.

### Webots Resources
- Webots 2025a Documentation: https://cyberbotics.com/doc/guide
- e-puck Robot: https://cyberbotics.com/doc/guide/epuck
- Supervisor Programming: https://cyberbotics.com/doc/guide/supervisor-programming
- Python Controller API: https://cyberbotics.com/doc/guide/controller-programming

### Python Libraries
- NumPy: https://numpy.org/doc/
- SciPy: https://docs.scipy.org/doc/scipy/
- Matplotlib: https://matplotlib.org/stable/
- OpenCV: https://docs.opencv.org/4.x/
- PyTorch: https://pytorch.org/docs/stable/

---

## 12. Next Steps (Immediate Actions)

1. **Verify Webots 2025a installation** - Download from Cyberbotics, test basic world
2. **Initialize GitHub repository** - `gh repo create asr700-assignment --public`
3. **Set up UV project** - `uv init`, add dependencies, create structure
4. **Create first commit** - Project structure + pyproject.toml + README.txt skeleton
5. **Build Component A world** - Basic e-puck in arena with obstacles
6. **Begin perception pipeline** - LiDAR + camera data extraction

---

*Plan created: 2026-10-21*
*Last updated: 2026-10-21*
*Status: Ready for execution*