# ***************************************************************************************
# *    Title: Adopted from gym-pybullet-drones
# *    Author: Learning Systems and Robotics Lab (LSY)
# *    Date: August 2023
# *    Availability: GitHub: https://github.com/utiasDSL/gym-pybullet-drones
# *
# ***************************************************************************************/


from gym.envs.registration import register

register(
    id="ctrl-aviary-v0",
    entry_point="gym_pybullet_drones.envs:CtrlAviary",
)

register(
    id="dyn-aviary-v0",
    entry_point="gym_pybullet_drones.envs:DynAviary",
)

register(
    id="velocity-aviary-v0",
    entry_point="gym_pybullet_drones.envs:VelocityAviary",
)

register(
    id="vision-aviary-v0",
    entry_point="gym_pybullet_drones.envs:VisionAviary",
)


register(
    id="takeoff-aviary-v0",
    entry_point="gym_pybullet_drones.envs.single_agent_rl:TakeoffAviary",
)

register(
    id="hover-aviary-v0",
    entry_point="gym_pybullet_drones.envs.single_agent_rl:HoverAviary",
)

register(
    id="flythrugate-aviary-v0",
    entry_point="gym_pybullet_drones.envs.single_agent_rl:FlyThruGateAviary",
)

register(
    id="tune-aviary-v0",
    entry_point="gym_pybullet_drones.envs.single_agent_rl:TuneAviary",
)


register(
    id="flock-aviary-v0",
    entry_point="gym_pybullet_drones.envs.multi_agent_rl:FlockAviary",
)

register(
    id="leaderfollower-aviary-v0",
    entry_point="gym_pybullet_drones.envs.multi_agent_rl:LeaderFollowerAviary",
)

register(
    id="meetup-aviary-v0",
    entry_point="gym_pybullet_drones.envs.multi_agent_rl:MeetupAviary",
)
