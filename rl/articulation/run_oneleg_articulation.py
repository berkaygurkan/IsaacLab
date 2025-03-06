# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to spawn a cart-pole and interact with it.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/tutorials/01_assets/run_articulation.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse
import numpy as np

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with an articulation.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import math
import omni.isaac.core.utils.prims as prim_utils

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns


SquatTime = 3.0  # Duration of squat phase
JumpTime = 4.0   # Duration of jump phase
LandingTime = 5.0 # Duration of landing phase (can be adjusted or removed for simpler jump)
SimulationEndTime = 7.0

SquatTorque_thigh = -10 # Example squat thigh torque
SquatTorque_calf = -10  # Example squat calf torque
JumpTorque_thigh = 40   # Example jump thigh torque
JumpTorque_calf = 40    # Example jump calf torque
RestTorque_thigh = 11  # Example resting thigh torque
RestTorque_calf = -10 # Example resting calf torque
LandingTorque_thigh = -5 # Example landing thigh torque (adjust as needed)
LandingTorque_calf = -5  # Example landing calf torque (adjust as needed)






##
# Pre-defined configs
##

from omni.isaac.lab_assets.unitree import UNITREE_A1_ONELEG_CFG  # isort: skip


def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.physics_material.static_friction = cfg.physics_material.static_friction / 10
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Create separate groups called "Origin1", "Origin2"
    # Each group will have a robot in it
    origins = [[0.0, 0.0, 0.0]]
    # Origin 1
    prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])

    # Articulation
    unitreeA1_cfg = UNITREE_A1_ONELEG_CFG.copy()
    unitreeA1_cfg.prim_path = "/World/Origin.*/Robot"
    unitreeA1 = Articulation(cfg=unitreeA1_cfg)
  
    # return the scene information
    scene_entities = {"unitreeA1": unitreeA1}
    return scene_entities, origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability. In general, it is better to access the entities directly from
    #   the dictionary. This dictionary is replaced by the InteractiveScene class in the next tutorial.
    robot = entities["unitreeA1"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    print("Sim dt",sim_dt)
  
    ## Info Time
    print("Joint Names",robot.data.joint_names)
    print("Joint Friction",robot.data.joint_friction)
    print("Joint Limits",robot.data.joint_limits)


    # Simulation loop

    SimulationStartTime = 4
    SimulationEndTime = 8
    PulseTime = 6
    lastIndex = int(SimulationEndTime/sim_dt)
    print("[INFO] lastIndex",lastIndex)
    BodyState = torch.zeros((lastIndex,6,13))
    count = 0
    t = 0
    amplitude = 40
    frequency = 0.2  # Hz

    while simulation_app.is_running():
                
            # Generate sine wave
 
        
        if t < SquatTime:
            #torque = torch.tensor([[ 0, SquatTorque_thigh,  SquatTorque_calf]]) # Squat phase torque
            torque = torch.tensor([[ 0, 0,  -10]]) # Squat phase torque
            print("[INFO] Squat Time",t)
        elif t >= SquatTime and t < JumpTime:
            print("[INFO] Jump Time",t)
            torque = torch.tensor([[ 0, JumpTorque_thigh,  JumpTorque_calf]])  # Jump phase torque
            torque = torch.tensor([[ 0, 10,  -10]]) # Squat phase torque
        elif t >= JumpTime and t < LandingTime:
            torque = torch.tensor([[ 0, 0, 0]]) # Flight phase - no torque, or use resting torque `torch.tensor([[ 0, RestTorque_thigh,  RestTorque_calf]])`
            print("[Fly Time] Jump Time",t)
        elif t >= LandingTime and t < SimulationEndTime:
             torque = torch.tensor([[ 0, RestTorque_thigh,  RestTorque_calf]]) # Rest phase torque
             torque = torch.tensor([[ 0, 0,  0]])  # Jump phase torque
        else:
            torque = torch.tensor([[ 0, RestTorque_thigh,  RestTorque_calf]]) # Default rest torque after simulation
            torque = torch.tensor([[ 0, 0,  -33]]) # Squat phase torque    
        
        
        target_angles = torch.tensor([[ 0, 40, -55]]) # Squat phase torque   


        # Apply random action
        # -- generate random joint efforts
        efforts = torque
        #efforts = torch.randn_like(robot.data.joint_pos) * 4
        # -- apply action to the robot
        #robot.set_joint_effort_target(efforts)
        robot.set_joint_position_target(target_angles)
        # -- write data to sim
        robot.write_data_to_sim()
        # Perform step
        sim.step()
        # Increment counter
    

        #print("Contact Forces",contact_forces.data.net_forces_w)
        #process_and_log_tensor(robot.data.body_state_w)
 

        count += 1
        t += sim_dt

        
        if count%(1/sim_dt) == 0:
            print("[INFO] Time",t)

        if t >= SimulationEndTime:
            print("[INFO] Simulation Time Over")
            #print("[INFO] Body States",robot.data.body_state_w)
            break

        
        #print("[INFO] Sine",sine_wave)
        # Update buffers
        robot.update(sim_dt)
    


def process_and_log_tensor(tensor_data, filename="tensor_log.txt"):
    """Processes a tensor, stores it in a buffer (list), and logs it to a file.

    Args:
        tensor_data: The tensor to process.
        filename: The name of the file to log to.
    """

    # Convert tensor to NumPy array for easier handling
    tensor_np = tensor_data.cpu().numpy()  # Move to CPU if it's on GPU

    # Flatten the array for easier CSV writing
    tensor_flat = tensor_np.flatten()
    
    # Convert to string representation for logging (comma-separated)
    tensor_str = ",".join(map(str, tensor_flat))
    
    # Append the data to file
    with open(filename, "a") as f:
        f.write(tensor_str + "\n") # Add newline for each tensor


def main():
    """Main function."""

    # Load kit helper
    
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device,dt=0.005)
    sim = SimulationContext(sim_cfg)
    

    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    # Design scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    
    run_simulator(sim, scene_entities, scene_origins)
    

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()