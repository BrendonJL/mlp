"""
POC: Frame Skip Replay Test
============================
Validates whether subsampled (every 4th frame) demonstration data
from smbdataset can still complete World 1-1 in gym-super-mario-bros.
"""

from __future__ import annotations

import re
from pathlib import Path

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

# --- Constants ---

SMBDATASET_DIR = Path(__file__).parent.parent / "smbdataset"

# NES button bit positions (MSB→LSB): A, up, left, B, start, right, select, down
NES_BUTTONS = {
    "A": 128,
    "up": 64,
    "left": 32,
    "B": 16,
    "start": 8,
    "right": 4,
    "select": 2,
    "down": 1,
}

# Mapping from NES 8-bit action → SIMPLE_MOVEMENT index
# SIMPLE_MOVEMENT = [NOOP, right, right+A, right+B, right+A+B, A, left]
#                     0      1       2        3         4       5    6
NES_TO_SIMPLE = {
    0: 0,  # NOOP → NOOP
    4: 1,  # right → right
    132: 2,  # A+right → right+A
    20: 3,  # B+right → right+B
    148: 4,  # A+B+right → right+A+B
    128: 5,  # A → A (jump)
    32: 6,  # left → left
    # Unmapped actions → nearest equivalent
    16: 0,  # B alone → NOOP (human error)
    144: 5,  # A+B (no direction) → A (jump)
    48: 6,  # left+B → left
    176: 6,  # A+left+B → left
    18: 0,  # B+down → NOOP
    50: 6,  # left+B+down → left
    22: 3,  # B+right+down → right+B
}


# --- Step 1: Parse smbdataset episode ---
def parse_episode_actions(episode_dir: Path) -> list[int]:
    pairs = []
    for filename in episode_dir.iterdir():
        match = re.search(r"_f(\d+)_a(\d+)_", filename.name)
        if match:
            x = int(match.group(1))
            y = int(match.group(2))
            pairs.append((x, y))
    pairs.sort()
    return [y for _, y in pairs]


# --- Step 2: Map NES actions to SIMPLE_MOVEMENT ---
def map_to_simple_movement(nes_actions: list[int]) -> list[int]:
    simple_actions = []
    unmapped_count = 0
    for action in nes_actions:
        if action in NES_TO_SIMPLE:
            simple_actions.append(NES_TO_SIMPLE[action])
        else:
            # Unknown action — default to NOOP, print warning
            print(f"  Warning: unmapped NES action {action} → defaulting to NOOP")
            simple_actions.append(0)
            unmapped_count += 1

    if unmapped_count > 0:
        print(f"  Total unmapped actions: {unmapped_count}/{len(nes_actions)}")

    return simple_actions


# --- Step 3: Subsample for frame skip ---
def subsample_actions(actions: list[int], skip: int = 4) -> list[int]:
    return actions[::skip]


# --- Step 4: Replay through environment ---
def replay_episode(actions: list[int], render: bool = True) -> dict:
    env = gym_super_mario_bros.make(
        "SuperMarioBros-1-1-v3",
        apply_api_compatibility=True,
        render_mode="human" if render else "rgb_array",
    )
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    _, info = env.reset()
    total_reward = 0
    steps = 0
    max_x_pos = 0

    for action in actions:
        _, reward, terminated, truncated, info = env.step(action)
        max_x_pos = max(max_x_pos, info["x_pos"])
        total_reward += reward
        steps += 1
        done = terminated or truncated
        if done:
            break

    env.close()
    return {
        "max_x": max_x_pos,
        "completed": info["flag_get"],
        "total_reward": total_reward,
        "steps": steps,
    }


# --- Step 5: Run the full POC ---
def find_winning_episodes(world: str = "1-1") -> list[Path]:
    """Find all winning episode folders for a given world-level."""
    pattern = re.compile(rf"_e\d+_{re.escape(world)}_win$")
    episodes = [
        d
        for d in sorted(SMBDATASET_DIR.iterdir())
        if d.is_dir() and pattern.search(d.name)
    ]
    return episodes


def main() -> None:
    print("=" * 60)
    print("POC: Frame Skip Replay Validation")
    print("=" * 60)

    episodes = find_winning_episodes("1-1")
    print(f"\nFound {len(episodes)} winning 1-1 episodes")

    for ep_dir in episodes:
        print(f"\n--- {ep_dir.name} ---")

        # Parse raw actions
        nes_actions = parse_episode_actions(ep_dir)
        print(f"  Total frames (60fps): {len(nes_actions)}")

        # Map to SIMPLE_MOVEMENT
        simple_actions = map_to_simple_movement(nes_actions)

        # Full-speed replay (baseline)
        print("  Replaying full-speed (no skip)...")
        full_result = replay_episode(simple_actions)
        print(
            f"    Max X: {full_result['max_x']}, Completed: {full_result['completed']}"
        )

        # Subsampled replay (frame skip = 4)
        subsampled = subsample_actions(simple_actions, skip=4)
        print(f"  Replaying subsampled (skip=4, {len(subsampled)} actions)...")
        skip_result = replay_episode(subsampled)
        print(
            f"    Max X: {skip_result['max_x']}, Completed: {skip_result['completed']}"
        )

        # Compare
        print(
            f"  Distance retained: {skip_result['max_x'] / max(full_result['max_x'], 1) * 100:.1f}%"
        )

    print("\n" + "=" * 60)
    print("POC Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
