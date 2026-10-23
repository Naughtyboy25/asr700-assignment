"""
Pytest configuration: make project modules importable from tests.

Adds the project root (for `python_analysis` package imports), the shared
utils package, and both Webots controller directories to sys.path.
"""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

for p in [
    ROOT,
    os.path.join(ROOT, "python_analysis", "utils"),
    os.path.join(ROOT, "webots_project", "controllers", "robot_controller_a"),
    os.path.join(ROOT, "webots_project", "controllers", "swarm_agents"),
]:
    if p not in sys.path:
        sys.path.insert(0, p)
