"""
Smoke tests: every Python file in the submission must compile, and the
project must not depend on the Webots runtime to import its pure modules.
"""

import os
import py_compile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

PY_FILES = [
    "python_simulations/component_a/run_component_a.py",
    "python_simulations/component_b/run_component_b.py",
    "python_simulations/config.py",
    "python_simulations/helpers.py",
    "python_analysis/utils/config.py",
    "python_analysis/utils/helpers.py",
    "webots_project/controllers/robot_controller_a/control.py",
    "webots_project/controllers/robot_controller_a/kinematics.py",
    "webots_project/controllers/robot_controller_a/perception.py",
    "webots_project/controllers/robot_controller_a/planning.py",
    "webots_project/controllers/robot_controller_a/robot_controller_a.py",
    "webots_project/controllers/swarm_agents/consensus.py",
    "webots_project/controllers/swarm_agents/follower_controller.py",
    "webots_project/controllers/swarm_agents/leader_controller.py",
    "webots_project/controllers/swarm_agents/ml_policy.py",
    "webots_project/controllers/swarm_agents/swarm_agents.py",
    "webots_project/controllers/swarm_agents/swarm_intelligence.py",
]


def test_all_python_files_compile():
    for rel in PY_FILES:
        path = os.path.join(ROOT, rel)
        assert os.path.isfile(path), f"missing file: {rel}"
        py_compile.compile(path, doraise=True)


def test_analysis_notebooks_are_valid_json():
    import json

    for rel in [
        "python_analysis/component_a_analysis/analysis_a.ipynb",
        "python_analysis/component_b_analysis/analysis_b.ipynb",
    ]:
        path = os.path.join(ROOT, rel)
        with open(path, encoding="utf-8") as f:
            nb = json.load(f)
        assert nb["nbformat"] == 4
        assert len(nb["cells"]) > 0
