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
