#!/usr/bin/env python3

import pandas as pd
from sqlalchemy import create_engine

engine = create_engine(
    "postgresql://mario_rl_user:Bingbongbing123!@localhost:5432/mario_rl_db"
)

# =============================================================================
# QUERY 1: Basic Comparison (v4 vs v5)
# =============================================================================
query_comparison = """
SELECT
    experiment_id,
    COUNT(*) as episode_count,
    ROUND(AVG(distance_traveled)::numeric, 1) as avg_distance,
    ROUND(AVG(reward)::numeric, 1) as avg_reward,
    MAX(distance_traveled) as max_distance,
    MIN(distance_traveled) as min_distance
FROM episodes
WHERE experiment_id IN (38, 39)
GROUP BY experiment_id
ORDER BY experiment_id;
"""

# =============================================================================
# QUERY 2: High Reward, Low Distance Anomalies
# =============================================================================
query_anomalies = """
SELECT
    experiment_id,
    episode_number,
    distance_traveled,
    reward,
    ROUND((reward / NULLIF(distance_traveled, 0))::numeric, 2) as reward_per_pixel
FROM episodes
WHERE experiment_id = 39
AND reward > 1500
AND distance_traveled < 1000
ORDER BY reward DESC
LIMIT 20;
"""

# =============================================================================
# QUERY 3: Distance Distribution Buckets
# =============================================================================
query_buckets = """
SELECT
    experiment_id,
    CASE
        WHEN distance_traveled < 500 THEN '0-500'
        WHEN distance_traveled < 750 THEN '500-750'
        WHEN distance_traveled < 1000 THEN '750-1000'
        WHEN distance_traveled < 1500 THEN '1000-1500'
        WHEN distance_traveled < 2000 THEN '1500-2000'
        WHEN distance_traveled < 2500 THEN '2000-2500'
        ELSE '2500+'
    END as distance_bucket,
    COUNT(*) as episode_count,
    ROUND(AVG(reward)::numeric, 1) as avg_reward
FROM episodes
WHERE experiment_id IN (38, 39)
GROUP BY experiment_id, distance_bucket
ORDER BY experiment_id, distance_bucket;
"""

# =============================================================================
# QUERY 4: Milestone Analysis (for v5 specifically)
# =============================================================================
query_milestones = """
SELECT
    SUM(CASE WHEN distance_traveled >= 650 THEN 1 ELSE 0 END) as reached_650,
    SUM(CASE WHEN distance_traveled >= 900 THEN 1 ELSE 0 END) as reached_900,
    SUM(CASE WHEN distance_traveled >= 1200 THEN 1 ELSE 0 END) as reached_1200,
    SUM(CASE WHEN distance_traveled >= 1600 THEN 1 ELSE 0 END) as reached_1600,
    SUM(CASE WHEN distance_traveled >= 2000 THEN 1 ELSE 0 END) as reached_2000,
    SUM(CASE WHEN distance_traveled >= 2700 THEN 1 ELSE 0 END) as reached_2700,
    COUNT(*) as total_episodes
FROM episodes
WHERE experiment_id = 39;
"""


print("=" * 60)
print("PPO v5 EXPERIMENT ANALYSIS (Experiment 39)")
print("=" * 60)

print("\n📊 QUERY 1: Basic Comparison (v4 vs v5)")
print("-" * 40)
df_comparison = pd.read_sql(query_comparison, engine)
print(df_comparison.to_string(index=False))

print("\n\n🚨 QUERY 2: High Reward + Low Distance Anomalies")
print("-" * 40)
df_anomalies = pd.read_sql(query_anomalies, engine)
if len(df_anomalies) > 0:
    print(df_anomalies.to_string(index=False))
else:
    print("No anomalies found with current thresholds.")

print("\n\n📈 QUERY 3: Distance Distribution Buckets")
print("-" * 40)
df_buckets = pd.read_sql(query_buckets, engine)
print(df_buckets.to_string(index=False))

print("\n\n🏁 QUERY 4: Milestone Reach Rates (v5 only)")
print("-" * 40)
df_milestones = pd.read_sql(query_milestones, engine)
# Calculate percentages
total = df_milestones["total_episodes"].iloc[0]
print(f"Total episodes: {total}")
for col in df_milestones.columns:
    if col.startswith("reached_"):
        milestone = col.replace("reached_", "")
        count = df_milestones[col].iloc[0]
        pct = (count / total) * 100
        print(f"  {milestone}px: {count:,} episodes ({pct:.1f}%)")

print("\n" + "=" * 60)
print("Analysis complete!")
