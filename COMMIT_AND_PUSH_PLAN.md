# ASR700 Assignment - Commit & Push Plan

**Based on:** PROJECT_PLAN.md Section 4 (Git & GitHub Strategy)  
**Repository:** `asr700-assignment` (public)  
**Minimum Commits:** 13 meaningful commits  
**Branch Strategy:** Feature branches → PR → main  

---

## 1. Repository Initialization

### 1.1 Create GitHub Repository (One-time)
```bash
# Authenticate with GitHub CLI (if not done)
gh auth login

# Create public repository
gh repo create asr700-assignment \
  --public \
  --description "ASR700 Autonomous Systems and Robotics Assignment - Webots simulation with Python analysis" \
  --source=. \
  --remote=origin \
  --push
```

### 1.2 Alternative: Manual Setup
```bash
# If repo already exists locally
git remote add origin https://github.com/<username>/asr700-assignment.git
git branch -M main
git push -u origin main
```

---

## 2. Branching Strategy

| Branch | Purpose | Lifetime |
|--------|---------|----------|
| `main` | Protected, production-ready | Permanent |
| `feature/component-a-world` | Component A Webots world | Delete after merge |
| `feature/component-a-perception` | LiDAR + Camera fusion | Delete after merge |
| `feature/component-a-planning` | Dijkstra & A* planners | Delete after merge |
| `feature/component-a-control` | Kinematics + PID | Delete after merge |
| `feature/component-a-integration` | Closed-loop navigation | Delete after merge |
| `feature/component-b-world` | Multi-robot Webots world | Delete after merge |
| `feature/component-b-swarm` | Consensus + PSO/ACO | Delete after merge |
| `feature/component-b-ml` | Neural network training | Delete after merge |
| `feature/analysis-visualization` | Matplotlib/Plotly plots | Delete after merge |
| `feature/report` | LaTeX PDF report | Delete after merge |
| `feature/documentation` | README.txt + docs | Delete after merge |
| `chore/final-submission` | Final cleanup & ZIP | Delete after merge |

### 2.1 Branch Naming Convention
```
<type>/<component>-<feature>
```
- `feat/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation
- `chore/` - Maintenance
- `refactor/` - Code restructuring

---

## 3. Detailed Commit Plan (13 Commits)

### Commit 1: Project Initialization
```bash
git checkout -b feat/project-init

# Files to commit:
# - pyproject.toml
# - uv.lock
# - .gitignore
# - README.txt (skeleton)
# - PROJECT_PLAN.md
# - COMMIT_AND_PUSH_PLAN.md
# - python_analysis/utils/config.py
# - python_analysis/utils/helpers.py

git add pyproject.toml uv.lock .gitignore README.txt PROJECT_PLAN.md COMMIT_AND_PUSH_PLAN.md python_analysis/utils/
git commit -m "feat: initialize UV project with dependencies and project structure

- Configure pyproject.toml with core, ML, and dev dependencies
- Add UV lock file for reproducible builds
- Create comprehensive .gitignore for Python, Webots, and data files
- Initialize README.txt with run instructions template
- Add project planning documents
- Set up shared config and helper utilities"

git push -u origin feat/project-init
# Create PR: gh pr create --title "feat: initialize UV project" --body "Initial project setup with UV, dependencies, and structure"
```

### Commit 2: Component A - Webots World
```bash
git checkout -b feat/component-a-world main

# Files:
# - webots_project/worlds/component_a.wbt
# - webots_project/controllers/robot_controller_a/__init__.py (empty)

git add webots_project/worlds/component_a.wbt webots_project/controllers/robot_controller_a/
git commit -m "feat: create Webots world for Component A with e-puck robot and obstacles

- Design 10x10m bounded arena with walls
- Add 6 static obstacles (cylinders and boxes) for navigation challenge
- Place 4 colored landmarks (red, green, blue, yellow) for camera detection
- Configure e-puck robot at start position (1,1) with differential drive
- Set up lighting and viewpoints for simulation observation"

git push -u origin feat/component-a-world
# PR: gh pr create --title "feat: Component A Webots world" --body "World file with obstacles and landmarks for single robot navigation"
```

### Commit 3: Component A - Sensor Perception Pipeline
```bash
git checkout -b feat/component-a-perception main

# Files:
# - webots_project/controllers/robot_controller_a/perception.py
# - webots_project/controllers/robot_controller_a/__init__.py (updated)

git add webots_project/controllers/robot_controller_a/perception.py
git commit -m "feat: implement sensor perception pipeline (LiDAR + camera fusion)

- LiDARProcessor: range filtering, point cloud conversion, grid cell mapping
- CameraProcessor: HSV color segmentation for landmark detection, Canny edge detection
- SensorFusion: log-odds occupancy grid updates + persistent landmark memory
- PerceptionPipeline: unified interface for sensor data processing
- Support for 270° LiDAR (0.5° resolution) and 640x480 camera"

git push -u origin feat/component-a-perception
# PR: gh pr create --title "feat: Component A perception pipeline" --body "LiDAR and camera fusion with occupancy grid and landmark detection"
```

### Commit 4: Component A - Motion Planning
```bash
git checkout -b feat/component-a-planning main

# Files:
# - webots_project/controllers/robot_controller_a/planning.py

git add webots_project/controllers/robot_controller_a/planning.py
git commit -m "feat: implement Dijkstra and A* path planners with visualization

- OccupancyGridMap: collision checking, neighbor generation, world/grid transforms
- Dijkstra: uniform cost search with priority queue, 8-connected grid support
- A*: Euclidean/octile/Manhattan/Chebyshev heuristics, weighted A* support
- PlannerStats: nodes explored, path length, computation time, success tracking
- compare_planners(): statistical comparison over multiple runs
- Path smoothing (gradient descent) and shortcutting post-processing"

git push -u origin feat/component-a-planning
# PR: gh pr create --title "feat: Component A path planners" --body "Dijkstra and A* implementations with performance comparison utilities"
```

### Commit 5: Component A - Kinematics & PID Control
```bash
git checkout -b feat/component-a-control main

# Files:
# - webots_project/controllers/robot_controller_a/kinematics.py
# - webots_project/controllers/robot_controller_a/control.py

git add webots_project/controllers/robot_controller_a/kinematics.py webots_project/controllers/robot_controller_a/control.py
git commit -m "feat: implement differential drive kinematics and PID controller

- DifferentialDriveKinematics: forward/inverse kinematics, exact odometry integration
- KinematicController: pure pursuit path tracking with lookahead distance
- DualPIDController: separate linear/angular PID loops with output limits
- PathTrackingController: waypoint sequencing with lookahead target selection
- PID analysis: overshoot, rise time, settling time, steady-state error metrics
- Ziegler-Nichols tuning helper functions"

git push -u origin feat/component-a-control
# PR: gh pr create --title "feat: Component A kinematics and PID control" --body "Differential drive model and dual-loop PID with tuning analysis"
```

### Commit 6: Component A - System Integration
```bash
git checkout -b feat/component-a-integration main

# Files:
# - webots_project/controllers/robot_controller_a/robot_controller_a.py

git add webots_project/controllers/robot_controller_a/robot_controller_a.py
git commit -m "feat: integrate perception, planning, control into closed-loop navigation

- RobotControllerA: main Webots controller class
- Sensor reading loop: LiDAR, camera, GPS, IMU
- Perception → occupancy map update → periodic replanning
- PathTrackingController execution with PID velocity commands
- Trajectory logging for planned vs actual comparison
- Performance metrics: path length, time-to-goal, tracking error (RMSE)"

git push -u origin feat/component-a-integration
# PR: gh pr create --title "feat: Component A closed-loop integration" --body "Full autonomous navigation loop with replanning and performance logging"
```

### Commit 7: Component B - Multi-Robot Webots World
```bash
git checkout -b feat/component-b-world main

# Files:
# - webots_project/worlds/component_b.wbt

git add webots_project/worlds/component_b.wbt
git commit -m "feat: create Webots multi-robot world for Component B (4 e-pucks)

- 10x10m arena with central obstacle cluster and corner obstacles
- Leader robot (e-puck_leader) at bottom-left
- 3 Follower robots (e-puck_follower_1/2/3) at other corners
- Emitter/Receiver devices for inter-robot communication (2m range)
- 4 colored landmarks for visual reference
- Shared controller structure for leader/follower roles"

git push -u origin feat/component-b-world
# PR: gh pr create --title "feat: Component B multi-robot world" --body "Swarm environment with 4 e-pucks and communication setup"
```

### Commit 8: Component B - Consensus & Swarm Intelligence
```bash
git checkout -b feat/component-b-swarm main

# Files:
# - webots_project/controllers/swarm_agents/consensus.py
# - webots_project/controllers/swarm_agents/swarm_intelligence.py

git add webots_project/controllers/swarm_agents/consensus.py webots_project/controllers/swarm_agents/swarm_intelligence.py
git commit -m "feat: implement consensus algorithm and PSO/ACO for swarm coordination

- ConsensusProtocol: average/max/min consensus with neighbor detection
- LeaderElection: bully, random, closest-to-target, highest-connectivity
- Convergence analysis: spectral gap, eigenvalue stability
- ParticleSwarmOptimization: coverage waypoint optimization with bounds
- AntColonyOptimization: TSP solver and CoverageACO for area coverage
- PSO parameter studies: inertia, cognitive, social weight effects"

git push -u origin feat/component-b-swarm
# PR: gh pr create --title "feat: Component B consensus and swarm intelligence" --body "Average consensus, leader election, PSO and ACO for distributed coordination"
```

### Commit 9: Component B - Machine Learning
```bash
git checkout -b feat/component-b-ml main

# Files:
# - webots_project/controllers/swarm_agents/ml_policy.py
# - python_analysis/training/train_nn.py
# - python_analysis/training/model.py (if separate)
# - python_analysis/training/dataset_gen.py (if separate)

git add webots_project/controllers/swarm_agents/ml_policy.py python_analysis/training/
git commit -m "feat: train and evaluate neural network for obstacle avoidance

- NavigationCNN: dual-branch (LiDAR 1D Conv + Camera 2D Conv) fusion architecture
- NavigationMLP: LiDAR-only baseline for comparison
- DiscreteActionCNN: 5-class classification (forward, left, right, stop, backward)
- PolicyTrainer: MSE loss, Adam optimizer, ReduceLROnPlateau scheduling
- Synthetic data generation: proportional control targets from random poses
- Evaluation: MSE, MAE, confusion matrix, per-class precision/recall/F1
- ONNX export for Webots deployment"

git push -u origin feat/component-b-ml
# PR: gh pr create --title "feat: Component B neural network policy" --body "CNN for navigation with training pipeline and evaluation metrics"
```

### Commit 10: Component B - Swarm Controllers
```bash
git checkout -b feat/component-b-controllers main

# Files:
# - webots_project/controllers/swarm_agents/leader_controller.py
# - webots_project/controllers/swarm_agents/follower_controller.py

git add webots_project/controllers/swarm_agents/leader_controller.py webots_project/controllers/swarm_agents/follower_controller.py
git commit -m "feat: implement leader and follower swarm controllers

- LeaderController: PSO coverage planning, consensus coordination, target assignment
- FollowerController: consensus participation, local obstacle avoidance, target tracking
- Emitter/Receiver communication: state broadcast, leader command reception
- PID control for both roles with avoidance behavior
- Trajectory logging for emergent behavior visualization"

git push -u origin feat/component-b-controllers
# PR: gh pr create --title "feat: Component B swarm controllers" --body "Leader and follower controllers with consensus-based coordination"
```

### Commit 11: Analysis & Visualization
```bash
git checkout -b feat/analysis-visualization main

# Files:
# - python_analysis/component_a_analysis/analysis_a.ipynb
# - python_analysis/component_b_analysis/analysis_b.ipynb
# - report/figures/*.png (generated)

git add python_analysis/component_a_analysis/ python_analysis/component_b_analysis/ report/figures/
git commit -m "feat: generate all Matplotlib visualizations and performance metrics

Component A:
- Perception pipeline: LiDAR polar/cartesian, occupancy grid
- Planner comparison: Dijkstra vs A* paths, node exploration, timing
- PID response: gain tuning effects (overshoot, rise time, settling)
- Trajectory: planned vs actual with tracking error over time
- Trade-off analysis: perception noise → planning → control

Component B:
- Swarm trajectories with communication range visualization
- Consensus convergence: topology comparison, gain stability
- PSO/ACO convergence curves and parameter sensitivity
- NN training curves, confusion matrix, per-class metrics
- Centralized vs distributed decision-making comparison
- Emergent behavior animation (GIF)"

git push -u origin feat/analysis-visualization
# PR: gh pr create --title "feat: analysis visualizations" --body "Complete visualization suite for both components with performance tables"
```

### Commit 12: Professional PDF Report
```bash
git checkout -b feat/report main

# Files:
# - report/report.tex
# - report/bibliography.bib
# - report/figures/ (referenced)
# - report/ASR700_Report.pdf (compiled)

git add report/report.tex report/bibliography.bib report/ASR700_Report.pdf
git commit -m "docs: write professional PDF report with Harvard references

- Introduction: problem statement, objectives, scope
- Methodology: Component A (perception, planning, control, integration)
- Methodology: Component B (swarm setup, consensus, PSO/ACO, ML)
- Results: all figures, tables, metrics from analysis notebooks
- Discussion: trade-offs, ethics (bias, explainability, safety, SA context)
- Conclusion: findings, limitations, future work
- Harvard (AGSM) references throughout
- 3-9 pages, Times New Roman 12pt, 1.5 line spacing"

git push -u origin feat/report
# PR: gh pr create --title "docs: professional PDF report" --body "Complete report with Harvard referencing, 3-9 pages"
```

### Commit 13: Documentation & Final Submission
```bash
git checkout -b feat/documentation main

# Files:
# - README.txt (complete)
# - requirements.txt (final)
# - Any missing docs

git add README.txt requirements.txt
git commit -m "docs: add README.txt with run instructions and requirements

- Software versions: Webots 2025a, Python 3.10+, UV
- Complete dependency list from uv pip freeze
- Step-by-step installation and run instructions for both components
- Neural network training procedure
- Expected runtimes and hardware requirements
- Troubleshooting common issues
- Contact information"

git push -u origin feat/documentation
# PR: gh pr create --title "docs: complete README and requirements" --body "Full documentation for reproducibility"
```

### Commit 14: Final Cleanup (Optional - 14th commit)
```bash
git checkout -b chore/final-submission main

# Files:
# - Verify all deliverables
# - Remove any debug/temporary files
# - Create submission ZIP (< 50MB)

git add -A
git commit -m "chore: final cleanup, verify all deliverables, zip submission package

- Verify all 4 deliverables present and complete
- Run simulations end-to-end in Webots
- Execute analysis notebooks without errors
- Compile report PDF successfully
- Create submission ZIP (< 50MB)
- Confirm GitHub repo public with 13+ meaningful commits
- No ROS/Gazebo references anywhere"

git push -u origin chore/final-submission
# PR: gh pr create --title "chore: final submission package" --body "Final verification and submission package"
```

---

## 4. Push Workflow

### 4.1 Daily Workflow
```bash
# 1. Start new feature
git checkout main
git pull origin main
git checkout -b feat/component-a-perception

# 2. Work on feature (multiple small commits OK locally)
# ... make changes ...
git add .
git commit -m "wip: working on LiDAR processing"

# 3. Before push: squash/cleanup local commits
git rebase -i main  # Squash WIP commits into 1 meaningful commit

# 4. Push feature branch
git push -u origin feat/component-a-perception

# 5. Create PR
gh pr create --title "feat: Component A perception pipeline" \
  --body "Implements LiDAR + camera fusion with occupancy grid" \
  --base main

# 6. After review/approval: merge via GitHub UI or CLI
gh pr merge --squash --delete-branch
```

### 4.2 Commit Message Convention
```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:** feat, fix, docs, style, refactor, test, chore  
**Scopes:** component-a, component-b, perception, planning, control, swarm, ml, report, docs

**Examples:**
```
feat(component-a): implement A* planner with Euclidean heuristic

- Add heuristic weight parameter for weighted A*
- Support octile, Manhattan, Chebyshev distances
- Include planner statistics tracking

Closes #4
```

```
fix(component-b): resolve follower consensus convergence issue

- Increase consensus alpha from 0.05 to 0.1
- Add neighbor validation before consensus step
- Fix vector value synchronization

Fixes #12
```

---

## 5. GitHub CLI Commands Reference

### 5.1 Repository Management
```bash
# Create repo
gh repo create asr700-assignment --public -d "Description" --source=. --push

# View repo
gh repo view

# Clone
gh repo clone <username>/asr700-assignment
```

### 5.2 Branch & PR Management
```bash
# List branches
git branch -a

# Create PR
gh pr create --title "Title" --body "Body" --base main --head feat/branch

# List PRs
gh pr list

# View PR
gh pr view <number>

# Merge PR (squash, delete branch)
gh pr merge <number> --squash --delete-branch

# Merge PR (rebase, keep branch)
gh pr merge <number> --rebase

# Checkout PR locally
gh pr checkout <number>
```

### 5.3 Commit History Verification
```bash
# Check commit count
git rev-list --count main

# View commit log (pretty)
git log --oneline --graph --decorate -20

# Verify commit messages
git log --oneline --grep="feat\|fix\|docs" | head -20

# Check for conventional commits
git log --oneline | grep -E "^[a-f0-9]+ (feat|fix|docs|style|refactor|test|chore)\("
```

---

## 6. Verification Checklist (Pre-Submission)

### 6.1 Git Requirements
- [ ] Repository is public
- [ ] Minimum 13 commits on main branch
- [ ] All commits have descriptive messages (no "update", "final", "fix")
- [ ] Commit history shows incremental progress over time
- [ ] No single commit with all changes
- [ ] Conventional commit format used

### 6.2 Code Quality
- [ ] `uv run black .` - formatting applied
- [ ] `uv run ruff check .` - linting passes
- [ ] `uv run pytest tests/` - tests pass (if tests exist)
- [ ] `uv run python -m py_compile <file>` - no syntax errors

### 6.3 Simulation Verification
- [ ] `webots_project/worlds/component_a.wbt` opens and runs
- [ ] `webots_project/worlds/component_b.wbt` opens and runs
- [ ] Controllers build without errors in Webots
- [ ] Component A: robot navigates from start to goal
- [ ] Component B: 4 robots show emergent coverage behavior

### 6.4 Analysis Verification
- [ ] `python_analysis/component_a_analysis/analysis_a.ipynb` executes fully
- [ ] `python_analysis/component_b_analysis/analysis_b.ipynb` executes fully
- [ ] All figures generated in `report/figures/`
- [ ] Performance tables exported as CSV

### 6.5 Report Verification
- [ ] `report/report.tex` compiles to PDF (3-9 pages)
- [ ] Harvard (AGSM) referencing style used
- [ ] All figures referenced and included
- [ ] Tables formatted with booktabs
- [ ] Times New Roman 12pt, 1.5 line spacing

### 6.6 Documentation
- [ ] `README.txt` complete with all sections
- [ ] `requirements.txt` matches `uv pip freeze`
- [ ] Run instructions tested on clean environment

### 6.7 Submission Package
- [ ] ZIP file created: `StudentNumber_Surname_ASR700_Assignment.zip`
- [ ] ZIP size < 50MB
- [ ] Contains: webots_project/, python_analysis/, report/, data/, README.txt
- [ ] Excludes: .venv/, __pycache__/, .git/, build/, *.zip, *.mp4

---

## 7. Timeline & Milestones

| Week | Target Commits | Milestone |
|------|---------------|-----------|
| 1 | 1-2 | Repo created, UV initialized, Component A world |
| 2 | 3-4 | Perception + Planning complete |
| 3 | 5-6 | Control + Integration complete |
| 4 | 7-8 | Component B world + Swarm algorithms |
| 5 | 9-10 | ML training + Swarm controllers |
| 6 | 11-13 | Visualizations + Report + Documentation |
| 6+ | 14 | Final cleanup & submission |

---

## 8. Emergency Procedures

### 8.1 Fix Last Commit Message
```bash
git commit --amend -m "feat: corrected commit message"
git push --force-with-lease origin feat/branch-name
```

### 8.2 Recover Deleted Branch
```bash
git reflog
git checkout -b recovered-branch <commit-hash>
```

### 8.3 Fix Pushed Commit (before PR merge)
```bash
git commit --amend
git push --force-with-lease origin feat/branch-name
```

### 8.4 Large File Removal (if ZIP > 50MB)
```bash
# Find large files
git ls-files | xargs ls -la | sort -k5 -nr | head -20

# Remove from history (BFG Repo-Cleaner or git-filter-repo)
# Then force push
git push --force origin main
```

---

## 9. Useful Git Aliases (Add to ~/.gitconfig)

```ini
[alias]
    # Pretty log
    lg = log --oneline --graph --decorate --all
    lg1 = log --graph --abbrev-commit --decorate --format=format:'%C(bold blue)%h%C(reset) - %C(bold green)(%ar)%C(reset) %C(white)%s%C(reset) %C(dim white)- %an%C(reset)%C(auto)%d%C(reset)'
    
    # Status shortcuts
    st = status -sb
    co = checkout
    br = branch
    ci = commit
    
    # Commit with conventional format
    cm = "!f() { git commit -m \"$1($2): $3\"; }; f"
    
    # Push with upstream
    pu = "!git push -u origin $(git branch --show-current)"
    
    # PR creation
    pr = "!gh pr create --title \"$1\" --body \"$2\""
    
    # Cleanup merged branches
    clean-branches = "!git branch --merged main | grep -v 'main$' | xargs git branch -d"
```

---

## 10. Quick Reference Card

| Action | Command |
|--------|---------|
| New feature | `git checkout main && git pull && git checkout -b feat/xyz` |
| Stage changes | `git add -p` (interactive) or `git add .` |
| Commit | `git commit -m "feat(scope): message"` |
| Push | `git push -u origin branch-name` |
| Create PR | `gh pr create --title "..." --body "..."` |
| Merge PR | `gh pr merge --squash --delete-branch` |
| Sync main | `git checkout main && git pull` |
| Delete local merged | `git branch --merged main | grep -v main | xargs git branch -d` |
| View history | `git lg` |
| Verify commits | `git log --oneline -20` |

---

*Created: 2026-10-21*  
*Based on: PROJECT_PLAN.md v1.0*  
*Status: Ready for execution*