ASR700 Assignment - Autonomous Systems and Robotics 700
=======================================================


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
