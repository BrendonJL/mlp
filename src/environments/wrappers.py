import gym
import numpy as np
from gym import spaces
import cv2
from collections import deque


class CompatibilityWrapper(gym.Wrapper):
    """Wrapper to handle API compatibility between old and new Gym versions."""

    def reset(self, **kwargs):
        # Try with kwargs first (new API)
        try:
            result = self.env.reset(**kwargs)
        except TypeError:
            # Fallback to no kwargs (old API)
            result = self.env.reset()

        # Ensure result is always (obs, info) tuple for new API
        if isinstance(result, tuple):
            return result  # Already (obs, info)
        else:
            return result, {}  # Convert obs -> (obs, {})

    def step(self, action):
        # Call underlying environment
        result = self.env.step(action)

        # Handle old gym API (4 values) vs new gymnasium API (5 values)
        if len(result) == 4:
            obs, reward, done, info = result
            # Convert to new API: done -> (terminated, truncated)
            terminated = done
            truncated = False
            return obs, reward, terminated, truncated, info
        else:
            # Already new API
            return result


class GrayscaleWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(240, 256, 1), dtype=np.uint8
        )

    def observation(self, obs):
        greyscale = np.dot(obs, [0.299, 0.587, 0.114])
        greyscale = np.expand_dims(greyscale, axis=-1)

        return greyscale.astype(np.uint8)


class ResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env, size=84):
        super().__init__(env)
        self.size = size
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8
        )

    def observation(self, obs):
        # Resize (might lose channel dimension)
        resized = cv2.resize(obs, (self.size, self.size))

        # Add channel dimension if cv2 squeezed it out
        if len(resized.shape) == 2:
            resized = np.expand_dims(resized, axis=-1)

        return resized.astype(np.uint8)


class NormalizeWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(84, 84, 1), dtype=np.float32
        )

    def observation(self, obs):
        normalized = obs / 255.0
        return normalized.astype(np.float32)


class FrameStackWrapper(gym.ObservationWrapper):
    def __init__(self, env, num_stack=4):
        super().__init__(env)
        self.num_stack = num_stack

        self.frames = deque(maxlen=num_stack)

        # Use uint8 [0, 255] to match input from ResizeWrapper
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(84, 84, num_stack), dtype=np.uint8
        )

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)

        # Handle both old and new Gym API
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}

        # Initialize frame buffer
        for _ in range(self.num_stack):
            self.frames.append(obs)

        # Return (obs, info) tuple for new API
        return self._get_stacked_frames(), info

    def observation(self, obs):
        self.frames.append(obs)

        return self._get_stacked_frames()

    def _get_stacked_frames(self):
        return np.concatenate(list(self.frames), axis=-1)


class RewardShapingWrapper(gym.Wrapper):
    """Shape rewards to encourage forward progress and discourage standing still."""

    def __init__(
        self,
        env,
        forward_bonus=0.1,
        backward_penalty=0.1,
        idle_penalty=0.2,
        death_penalty=50.0,
        max_stuck_steps=300,  # End episode if stuck for this many steps (increased for v3)
    ):
        super().__init__(env)
        self.forward_bonus = forward_bonus  # Reward per pixel moved right
        self.backward_penalty = backward_penalty  # Penalty per pixel moved left
        self.idle_penalty = idle_penalty  # Penalty for not moving
        self.death_penalty = death_penalty  # Penalty for losing a life
        self.max_stuck_steps = max_stuck_steps  # Early termination threshold
        self.prev_x_pos = 0
        self.prev_life = 2
        self.stuck_count = 0  # Track consecutive steps without movement
        # Milestone bonuses for reaching certain x positions
        self.milestones = {650: 150, 900: 100, 1200: 150, 1600: 200, 2000: 250}
        self.reached_milestones = set()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_x_pos = info.get("x_pos", 0)
        self.prev_life = info.get("life", 2)
        self.stuck_count = 0
        self.reached_milestones = set()  # Reset milestones each episode
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        current_x_pos = info["x_pos"]
        current_life = info["life"]
        x_delta = current_x_pos - self.prev_x_pos

        # Apply reward shaping
        if x_delta > 0:
            reward = reward + self.forward_bonus * x_delta
            self.stuck_count = 0  # Reset stuck counter when moving forward
        if x_delta < 0:
            reward = reward - self.backward_penalty * abs(x_delta)
            self.stuck_count = 0  # Reset stuck counter when moving (even backward)
        if x_delta == 0:
            reward = reward - self.idle_penalty
            self.stuck_count += 1  # Increment stuck counter

        if current_life < self.prev_life:
            reward = reward - self.death_penalty

        # Milestone bonuses for reaching new x positions
        for threshold, bonus in self.milestones.items():
            if current_x_pos >= threshold and threshold not in self.reached_milestones:
                reward += bonus
                self.reached_milestones.add(threshold)

        # Early termination if stuck too long
        if self.stuck_count >= self.max_stuck_steps:
            terminated = True

        self.prev_x_pos = current_x_pos
        self.prev_life = current_life

        return obs, reward, terminated, truncated, info


class TransposeWrapper(gym.ObservationWrapper):
    """Transpose observation from (H, W, C) to (C, H, W) for PyTorch."""

    def __init__(self, env):
        super().__init__(env)
        # Get original shape (H, W, C)
        obs_shape = self.env.observation_space.shape
        # Transpose to (C, H, W)
        new_shape = (obs_shape[2], obs_shape[0], obs_shape[1])

        # Keep dtype as uint8 [0, 255] for SB3 compatibility
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=new_shape,
            dtype=np.uint8,
        )

    def observation(self, obs):
        # Transpose from (H, W, C) to (C, H, W)
        return np.transpose(obs, (2, 0, 1))


if __name__ == "__main__":
    import gym_super_mario_bros

    # Create env and wrap it
    env = gym_super_mario_bros.make("SuperMarioBros-v3", apply_api_compatibility=True)
    env = GrayscaleWrapper(env)
    env = ResizeWrapper(env)
    env = NormalizeWrapper(env)
    env = FrameStackWrapper(env)

    # Reset and check observation
    obs = env.reset()
    print(f"Observation shape: {obs.shape}")  # Should be (84, 84, 4)
    print(f"Observation dtype: {obs.dtype}")
    print(f"Min: {obs.min():.4f}, Max: {obs.max():.4f}")  # Should be 0.0-1.0

    env.close()
