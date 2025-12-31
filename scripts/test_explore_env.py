# Imports
import gym_super_mario_bros

# Create the Environment
env = gym_super_mario_bros.make("SuperMarioBros-v3")

# Reset environment to get initial observation
observation = env.reset()

print("=== OBSERVATION SPACE ===")
print(f"Type: {type(observation)}")
print(f"Shape: {observation.shape}")
print(f"Data type: {observation.dtype}")
print(f"Min value: {observation.min()}, Max value: {observation.max()}")

print("\n=== ACTION SPACE ===")
print(f"Action space: {env.action_space}")
print(f"Number of possible actions: {env.action_space.n}")

env.close()
