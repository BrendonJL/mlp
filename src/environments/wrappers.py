from collections import deque
from typing import Any

import cv2
import gym  # type: ignore[import-untyped]
import numpy as np
from gym import spaces  # type: ignore[import-untyped]


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

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        # Call underlying environment
        result: tuple[Any, ...] = self.env.step(action)

        # Handle old gym API (4 values) vs new gymnasium API (5 values)
        if len(result) == 4:
            obs, reward, done, info = result
            terminated = done
            truncated = False
            return obs, reward, terminated, truncated, info
        else:
            return result  # type: ignore[return-value]


class SkipFrameWrapper(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = max(skip, 1)

    def step(self, action):
        total_reward = 0.0
        obs = None
        terminated = False
        truncated = False
        info = {}
        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return (obs, total_reward, terminated, truncated, info)


class GrayscaleWrapper(gym.ObservationWrapper):  # type: ignore[misc]
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(240, 256, 1), dtype=np.uint8
        )

    def observation(self, obs):
        greyscale = np.dot(obs, [0.299, 0.587, 0.114])
        greyscale = np.expand_dims(greyscale, axis=-1)

        return greyscale.astype(np.uint8)


class ResizeWrapper(gym.ObservationWrapper):  # type: ignore[misc]
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


class NormalizeWrapper(gym.ObservationWrapper):  # type: ignore[misc]
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(84, 84, 1), dtype=np.float32
        )

    def observation(self, obs):
        normalized = obs / 255.0
        return normalized.astype(np.float32)


class FrameStackWrapper(gym.ObservationWrapper):  # type: ignore[misc]
    def __init__(self, env, num_stack=4):
        super().__init__(env)
        self.num_stack = num_stack

        self.frames = deque(maxlen=num_stack)

        # Use uint8 [0, 255] to match input from ResizeWrapper
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(84, 84, num_stack), dtype=np.uint8
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Initialize frame buffer with copies of first observation
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
        completion_bonus=500.0,
        max_stuck_steps=300,  # End episode if stuck for this many steps (increased for v3)
    ):
        super().__init__(env)
        self.forward_bonus = forward_bonus  # Reward per pixel moved right
        self.backward_penalty = backward_penalty  # Penalty per pixel moved left
        self.idle_penalty = idle_penalty  # Penalty for not moving
        self.death_penalty = death_penalty  # Penalty for losing a life
        self.completion_bonus = completion_bonus  # Bonus for reaching the flag
        self.max_stuck_steps = max_stuck_steps  # Early termination threshold
        self.prev_x_pos = 0
        self.prev_life = 2
        self.prev_stage = 1
        self.accumulated_distance = 0  # Tracks total distance across stage transitions
        self.stuck_count = 0  # Track consecutive steps without movement
        self.level_completed = False  # Stage transition = level cleared
        # Milestone bonuses for reaching certain x positions
        self.milestones = {
            650: 150,
            900: 100,
            1200: 150,
            1600: 200,
            2000: 250,
            2500: 200,
            3000: 300,
            # Level 1-2 milestones (accumulated distance beyond 1-1's ~3154px)
            3500: 300,
            4000: 400,
            4500: 500,
            5000: 600,
        }
        self.reached_milestones = set()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_x_pos = info.get("x_pos", 0)
        self.prev_life = info.get("life", 2)
        self.prev_stage = info.get("stage", 1)
        self.accumulated_distance = 0
        self.stuck_count = 0
        self.level_completed = False
        self.reached_milestones = set()  # Reset milestones each episode
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        current_x_pos = info["x_pos"]
        current_life = info["life"]
        current_stage = info["stage"]

        # Detect stage transition (level completion) — accumulate distance
        if current_stage > self.prev_stage:
            self.accumulated_distance += self.prev_x_pos
            if not self.level_completed:
                reward += self.completion_bonus
                self.level_completed = True

        x_delta = current_x_pos - self.prev_x_pos

        # Guard: clamp x_delta at level transitions (x_pos resets to ~0 in 1-2)
        # No single frame should produce >50px movement; large deltas are transitions
        if abs(x_delta) > 50:
            x_delta = 0

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

        # Milestone bonuses using accumulated distance (survives stage transitions)
        total_distance = self.accumulated_distance + current_x_pos
        for threshold, bonus in self.milestones.items():
            if total_distance >= threshold and threshold not in self.reached_milestones:
                reward += bonus
                self.reached_milestones.add(threshold)

        # Early termination if stuck too long
        if self.stuck_count >= self.max_stuck_steps:
            terminated = True

        self.prev_x_pos = current_x_pos
        self.prev_life = current_life
        self.prev_stage = current_stage

        return obs, reward, terminated, truncated, info


class SpeedrunRewardWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        forward_bonus=0.1,  # Scaled down 10x for value function stability
        backward_penalty=0.0,  # Disabled - with RIGHT_ONLY, backward is involuntary physics
        idle_penalty=0.02,  # Low - standing still is okay
        death_penalty=5.0,  # Scaled down 10x
        completion_bonus=100.0,  # Scaled down 10x
        base_reward_scale=0.1,  # Scale down base game rewards (vietnh1009 uses /10)
        max_stuck_steps=300,
    ):
        super().__init__(env)
        self.forward_bonus = forward_bonus
        self.backward_penalty = backward_penalty
        self.idle_penalty = idle_penalty
        self.death_penalty = death_penalty
        self.completion_bonus = completion_bonus
        self.base_reward_scale = base_reward_scale
        self.max_stuck_steps = max_stuck_steps
        self.prev_x_pos = 0
        self.prev_life = 2
        self.stuck_count = 0
        # Milestones scaled down 10x for value function stability
        self.milestones = {
            650: 2.5,
            900: 4.0,
            1200: 5.0,
            1600: 7.5,
            2000: 10.0,
            2400: 15.0,
            2700: 22.5,
            3000: 35.0,
            3154: 40.0,
        }
        self.reached_milestones = set()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_x_pos = info.get("x_pos", 0)
        self.prev_life = info.get("life", 2)
        self.stuck_count = 0
        self.reached_milestones = set()
        # Cumulative reward tracking for debugging
        self.cumulative_base = 0.0
        self.cumulative_forward = 0.0
        self.cumulative_backward = 0.0
        self.cumulative_idle = 0.0
        self.cumulative_death = 0.0
        self.cumulative_milestone = 0.0
        return obs, info

    def step(self, action):
        obs, raw_base_reward, terminated, truncated, info = self.env.step(action)
        base_reward = (
            raw_base_reward * self.base_reward_scale
        )  # Scale down base game rewards
        current_x_pos = info["x_pos"]
        current_life = info["life"]
        x_delta = current_x_pos - self.prev_x_pos
        flag_get = info["flag_get"]

        # Track reward components for debugging
        forward_reward = 0.0
        backward_reward = 0.0
        idle_reward = 0.0
        death_reward = 0.0
        milestone_reward = 0.0
        completion_reward = 0.0

        if x_delta > 0:
            forward_reward = self.forward_bonus * x_delta
            self.stuck_count = 0
        if x_delta < 0:
            backward_reward = -self.backward_penalty * abs(x_delta)
            self.stuck_count = 0
        if x_delta == 0:
            idle_reward = -self.idle_penalty
            self.stuck_count += 1

        if current_life < self.prev_life:
            death_reward = -self.death_penalty

        # Milestone bonuses for reaching new x positions
        for threshold, bonus in self.milestones.items():
            if current_x_pos >= threshold and threshold not in self.reached_milestones:
                milestone_reward += bonus
                self.reached_milestones.add(threshold)

        # Early termination if stuck too long
        if self.stuck_count >= self.max_stuck_steps:
            terminated = True

        if flag_get:
            completion_reward = self.completion_bonus

        # Calculate total shaped reward
        shaped_reward = (
            forward_reward
            + backward_reward
            + idle_reward
            + death_reward
            + milestone_reward
            + completion_reward
        )
        total_reward = base_reward + shaped_reward

        # Accumulate for episode totals
        self.cumulative_base += base_reward
        self.cumulative_forward += forward_reward
        self.cumulative_backward += backward_reward
        self.cumulative_milestone += milestone_reward

        # Add cumulative reward breakdown to info for debugging
        info["reward_breakdown"] = {
            "base_game": round(self.cumulative_base, 1),
            "forward": round(self.cumulative_forward, 1),
            "backward": round(self.cumulative_backward, 1),
            "idle": round(self.cumulative_idle, 1),
            "death": round(self.cumulative_death, 1),
            "milestone": round(self.cumulative_milestone, 1),
            "completion": completion_reward,  # Only at end, no need to accumulate
        }

        self.prev_x_pos = current_x_pos
        self.prev_life = current_life

        return obs, total_reward, terminated, truncated, info


class SMBGrid:
    """
    Extract game state from NES RAM into a simplified 13x16 grid.
    Based on yumouwei's implementation.

    Grid values:
      0 = empty
      1 = tile/block/obstacle
      2 = Mario
     -1 = enemy
    """

    def __init__(self, env):
        # Traverse wrapper chain to find the NES environment with RAM
        current = env
        while hasattr(current, "env"):
            if hasattr(current, "ram"):
                break
            current = current.env
        self.ram = current.ram
        self.screen_size_x = 16
        self.screen_size_y = 13

        # Mario's position from RAM
        self.mario_level_x = self.ram[0x6D] * 256 + self.ram[0x86]
        self.mario_x = self.ram[0x3AD]
        self.mario_y = self.ram[0x3B8] + 16

        self.x_start = self.mario_level_x - self.mario_x
        self.rendered_screen = self._get_rendered_screen()

    def _tile_loc_to_ram_address(self, x: int, y: int) -> int:
        """Convert tile grid coordinates to RAM address."""
        page = x // 16
        x_loc = x % 16
        y_loc = page * 13 + y
        address = 0x500 + x_loc + y_loc * 16
        return address

    def _get_rendered_screen(self) -> np.ndarray:
        """Build the 13x16 grid representation from RAM."""
        rendered_screen = np.zeros((self.screen_size_y, self.screen_size_x))
        screen_start = int(np.rint(self.x_start / 16))

        # Fill in tiles/blocks
        for i in range(self.screen_size_x):
            for j in range(self.screen_size_y):
                x_loc = (screen_start + i) % (self.screen_size_x * 2)
                y_loc = j
                address = self._tile_loc_to_ram_address(x_loc, y_loc)

                if self.ram[address] != 0:
                    rendered_screen[j, i] = 1

        # Add Mario (value = 2)
        x_loc = (self.mario_x + 8) // 16
        y_loc = (self.mario_y - 32) // 16
        if 0 <= x_loc < 16 and 0 <= y_loc < 13:
            rendered_screen[y_loc, x_loc] = 2

        # Add enemies (value = -1)
        for i in range(5):
            if self.ram[0x0F + i] == 1:  # Enemy drawn flag
                enemy_x = self.ram[0x6E + i] * 256 + self.ram[0x87 + i] - self.x_start
                enemy_y = self.ram[0xCF + i]
                x_loc = (enemy_x + 8) // 16
                y_loc = (enemy_y + 8 - 32) // 16

                if 0 <= x_loc < 16 and 0 <= y_loc < 13:
                    rendered_screen[y_loc, x_loc] = -1

        return rendered_screen


class RAMObservationWrapper(gym.ObservationWrapper):  # type: ignore[misc]
    """
    Replace pixel observations with RAM-based grid observations.

    Uses SMBGrid to create a 13x16 grid representation of the game state.
    Includes frame stacking for temporal information.

    Output shape: (13 * 16 * n_stack,) = (832,) flattened for MlpPolicy
    """

    def __init__(self, env, n_stack: int = 4, n_skip: int = 1):
        super().__init__(env)
        self.n_stack = n_stack
        self.n_skip = n_skip
        self.width = 16
        self.height = 13

        # Flattened observation for MlpPolicy
        obs_size = self.height * self.width * self.n_stack
        self.observation_space = spaces.Box(
            low=-1.0, high=2.0, shape=(obs_size,), dtype=np.float32
        )

        # Frame buffer for stacking
        buffer_size = (self.n_stack - 1) * self.n_skip + 1
        self.frame_stack = np.zeros((self.height, self.width, buffer_size))

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)

        # Reset frame buffer
        buffer_size = (self.n_stack - 1) * self.n_skip + 1
        self.frame_stack = np.zeros((self.height, self.width, buffer_size))

        # Get initial grid and fill buffer
        grid = SMBGrid(self.env)
        frame = grid.rendered_screen
        for i in range(self.frame_stack.shape[-1]):
            self.frame_stack[:, :, i] = frame

        # Get stacked observation
        stacked = self.frame_stack[:, :, :: self.n_skip]
        return stacked.flatten().astype(np.float32), info

    def observation(self, _):  # unused - we read directly from RAM
        # Shift frames
        self.frame_stack[:, :, 1:] = self.frame_stack[:, :, :-1]

        # Get new grid
        grid = SMBGrid(self.env)
        self.frame_stack[:, :, 0] = grid.rendered_screen

        # Get stacked observation
        stacked = self.frame_stack[:, :, :: self.n_skip]
        return stacked.flatten().astype(np.float32)


class TransposeWrapper(gym.ObservationWrapper):  # type: ignore[misc]
    """Transpose observation from (H, W, C) to (C, H, W) for PyTorch."""

    def __init__(self, env):
        super().__init__(env)
        # Get original shape (H, W, C)
        obs_shape = self.env.observation_space.shape
        assert obs_shape is not None, "observation_space.shape cannot be None"
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
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")  # Should be (84, 84, 4)
    print(f"Observation dtype: {obs.dtype}")
    print(f"Min: {obs.min():.4f}, Max: {obs.max():.4f}")  # Should be 0.0-1.0

    env.close()
