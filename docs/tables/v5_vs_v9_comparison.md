# v5 vs v9: Two Paths to 100% Completion

| Metric       | ppo_v5         | ppo_v9              |
|:-------------|:---------------|:--------------------|
| Avg Reward   | 1722.8         | 8520.3              |
| Avg Steps    | 879.6          | 672.0               |
| Avg Distance | 3158.5         | 3155.6              |
| Avg Score    | 31544          | 20874               |
| Avg Coins    | 3.6            | 8.2                 |
| Completion % | 100.0%         | 100.0%              |
| Policy       | CnnPolicy      | MlpPolicy           |
| Observations | 84x84x4 pixels | 128-byte RAM        |
| Model Size   | ~20 MB         | ~1.4 MB             |
| Action Space | RIGHT_ONLY (4) | SIMPLE_MOVEMENT (7) |
