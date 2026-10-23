Autonomous Systems and Robotics 700
Intelligent Autonomous Robotic Systems Project
Richfield Graduate Institute of Technology

================================================================================
CONTENTS
================================================================================
webots_project/          Webots worlds + Python controllers (Component A & B)
python_simulations/      Fully offline reproducible simulations (Matplotlib)
python_analysis/         Analysis notebooks (component_a_analysis, component_b_analysis)
plots/                   All generated figures and metric files
tests/                   pytest suite (python -m pytest tests/)
report/ASR700_Report.docx Editable Word report (export to PDF for submission)
README.txt               This file

================================================================================
SOFTWARE VERSIONS
================================================================================
Webots: R2025a (or R2023b+)
Python: 3.10 – 3.12
Key packages: numpy, matplotlib, torch

Install offline dependencies:
  pip install numpy matplotlib torch

================================================================================
HOW TO RUN – OFFLINE PYTHON (recommended for marking reproducibility)
================================================================================
Component A:
  cd python_simulations/component_a
  python3 run_component_a.py

Component B:
  cd python_simulations/component_b
  python3 run_component_b.py

Plots and metrics are written to the plots/ directory.

================================================================================
HOW TO RUN – WEBOTS
================================================================================
1. Install Webots from https://cyberbotics.com
2. File → Open World → webots_project/worlds/component_a.wbt
   (or component_b.wbt for the swarm)
3. Press Play. Controllers are self-contained and do not require external packages.
4. Open the console to see pose / goal-distance logging.

Controller folders:
  robot_controller_a/   – single-robot perception + planning + PID tracking
  swarm_agents/         – leader/follower swarm (role via controllerArgs)

================================================================================
EXPECTED RUNTIME
================================================================================
Offline Component A: < 15 s
Offline Component B: < 30 s (includes neural-net training)
Webots Component A:  ~60–120 s of simulation time
Webots Component B:  ~120–180 s of simulation time

================================================================================
HARDWARE
================================================================================
Any modern CPU, ~2 GB free RAM. No GPU required for the offline demos.
Display required only for the Webots GUI.

================================================================================
ASSIGNMENT MAPPING
================================================================================
Component A (50)
  1. Sensor simulation & fusion     – run_component_a.py + Webots LiDAR/camera
  2. Dijkstra & A* + comparison     – run_component_a.py (plots A2)
  3. PID design & tuning            – run_component_a.py (plots A3)
  4. Integration & evaluation       – trajectory + metrics + ethics in report

Component B (50)
  1. Multi-agent Webots setup       – component_b.wbt (>=3 agents)
  2. Consensus + swarm intelligence – run_component_b.py (B1, B2)
  3. ML navigation policy           – NavNet training + confusion matrix (B3)
  4. Ethics / SA context            – Report section 4 (20 marks)

================================================================================
DECLARATION OF ORIGINALITY
================================================================================
This submission is the student's own work. Sources used for the literature
review and ethical discussion are cited in Harvard style inside the PDF report.
Simulation code and analysis scripts were developed for this assignment.
