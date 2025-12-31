# Imports
import gym_super_mario_bros

# Create the Environment
env = gym_super_mario_bros.make("SuperMarioBros-v3", apply_api_compatibility=True)

# Reset environment to get initial observation
observation = env.reset()

# Episode Config
num_episode = 10

max_steps_per_episode = 1000  # Reasonable limit

# Episode Loop
for episode in range(num_episode):
    observation = env.reset()

    done = False
    total_reward = 0
    step = 0

    # Game Loop
    while not done and step < max_steps_per_episode:
        action = env.action_space.sample()
        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)
        total_reward += reward
        step += 1
        done = terminated or truncated
    print(f"Episode {episode + 1}: Reward-{total_reward}, Steps={step}")
env.close()
