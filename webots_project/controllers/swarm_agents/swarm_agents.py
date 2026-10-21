"""Webots entry point for the swarm controllers.

Webots requires the controller file name to match its folder name, so the world
runs controller "swarm_agents" and picks the role from controllerArgs:
    controllerArgs ["leader"]   or   controllerArgs ["follower"]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

role = sys.argv[1] if len(sys.argv) > 1 else "follower"

if role == "leader":
    import leader_controller as module
else:
    import follower_controller as module

module.main()