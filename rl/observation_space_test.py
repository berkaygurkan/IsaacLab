import gymnasium as gym
env = gym.make("Isaac-Velocity-Flat-UnitreeThreeLeg-A1-v0")
print(env.observation_space.shape)  # Örneğin: (39,)
print(env.action_space.shape)       # Örneğin: (9,)