import gym
import numpy as np
from gym import spaces
import cv2
from collections import deque


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

        self.observation_space = spaces.Box(
            low=0.0, high=1, shape=(84, 84, num_stack), dtype=np.float32
        )

    def reset(self):
        obs = self.env.reset()

        # Handle tuple return from gym API
        if isinstance(obs, tuple):
            obs = obs[0]  # Just take the observation

        # Initialize frame buffer
        for _ in range(self.num_stack):
            self.frames.append(obs)

        return self._get_stacked_frames()

    def observation(self, obs):
        self.frames.append(obs)

        return self._get_stacked_frames()

    def _get_stacked_frames(self):
        return np.concatenate(list(self.frames), axis=-1)


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
