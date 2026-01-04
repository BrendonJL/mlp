"""
Database logging utilities for Mario RL experiments.
"""

from psycopg2 import pool
from psycopg2.extensions import cursor as Cursor
from datetime import datetime
from typing import Dict, Any, Optional


_connection_pool = None


def get_connection_pool():
    """Get or create the database connection pool."""
    global _connection_pool

    if _connection_pool is None:
        _connection_pool = pool.SimpleConnectionPool(
            minconn=1,
            maxconn=5,
            host="localhost",
            database="mario_rl_db",
            user="mario_rl_user",
            password="Bingbongbing123!",
        )
        print("✅ Database connection pool created")

    return _connection_pool


def close_connection_pool():
    """Close all connections in the pool."""
    global _connection_pool
    if _connection_pool is not None:
        _connection_pool.closeall()
        _connection_pool = None
        print("✅ Database connection pool closed")


def create_experiment(
    experiment_name: str,
    algorithm: str,
    git_commit_hash: str,
    python_version: str,
    pytorch_version: str,
    notes: Optional[str] = None,
) -> int:
    """Create a new experiment record in the database."""
    pool = get_connection_pool()
    conn = pool.getconn()
    cursor: Optional[Cursor] = None

    try:
        cursor = conn.cursor()
        assert cursor is not None

        query = """
            INSERT INTO experiments (
                experiment_name, algorithm, status, start_timestamp,
                git_commit_hash, python_version, pytorch_version, notes
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING experiment_id;
        """

        cursor.execute(
            query,
            (
                experiment_name,
                algorithm,
                "running",
                datetime.now(),
                git_commit_hash,
                python_version,
                pytorch_version,
                notes,
            ),
        )

        result = cursor.fetchone()
        assert result is not None
        experiment_id = result[0]
        conn.commit()

        print(f"✅ Created experiment {experiment_id}: {experiment_name}")
        return experiment_id

    except Exception as e:
        conn.rollback()
        print(f"❌ Error creating experiment: {e}")
        raise

    finally:
        if cursor:
            cursor.close()
        pool.putconn(conn)


def log_hyperparameters(experiment_id: int, hyperparams_dict: Dict[str, Any]):
    """Log hyperparameters for an experiment."""
    pool = get_connection_pool()
    conn = pool.getconn()
    cursor: Optional[Cursor] = None

    try:
        cursor = conn.cursor()
        assert cursor is not None

        for param_name, param_value in hyperparams_dict.items():
            query = """
                INSERT INTO hyperparameters (
                   experiment_id, parameter_name, parameter_value
                ) VALUES (%s, %s, %s)
            """
            cursor.execute(query, (experiment_id, param_name, str(param_value)))

        conn.commit()
        print(
            f"✅ Logged {len(hyperparams_dict)} hyperparameters for experiment {experiment_id}"
        )

    except Exception as e:
        conn.rollback()
        print(f"❌ Error logging hyperparameters: {e}")
        raise

    finally:
        if cursor:
            cursor.close()
        pool.putconn(conn)


def log_episode(experiment_id: int, episode_data: Dict[str, Any]):
    """Log results from a completed episode."""
    pool = get_connection_pool()
    conn = pool.getconn()
    cursor: Optional[Cursor] = None

    try:
        cursor = conn.cursor()
        assert cursor is not None

        episode_number = episode_data["episode_number"]
        reward = episode_data["total_reward"]  # Maps to 'reward' column
        episode_length = episode_data["episode_length"]
        distance_traveled = episode_data["x_pos"]  # Maps to 'distance_traveled' column
        y_pos = episode_data["y_pos"]
        score = episode_data["score"]
        time = episode_data["time"]
        coins = episode_data["coins"]
        life = episode_data["life"]
        status = episode_data["status"]
        level_completed = episode_data["flag_get"]  # Maps to 'level_completed' column
        world = episode_data["world"]
        stage = episode_data["stage"]

        query = """
            INSERT INTO episodes (
                experiment_id, episode_number, timestamp, reward, episode_length,
                distance_traveled, y_pos, score, time, coins, life, status,
                level_completed, world, stage
             ) VALUES (%s, %s, NOW(), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """

        cursor.execute(
            query,
            (
                experiment_id,
                episode_number,
                reward,
                episode_length,
                distance_traveled,
                y_pos,
                score,
                time,
                coins,
                life,
                status,
                level_completed,
                world,
                stage,
            ),
        )

        conn.commit()
        print(
            f"✅ Logged episode {episode_number}: reward={reward}, distance={distance_traveled}"
        )

    except Exception as e:
        conn.rollback()
        print(f"❌ Error logging episode: {e}")
        raise

    finally:
        if cursor:
            cursor.close()
        pool.putconn(conn)


def update_experiment(experiment_id: int, status: str, total_episodes: int):
    """Update experiment with final statistics when training completes."""
    pool = get_connection_pool()
    conn = pool.getconn()
    cursor: Optional[Cursor] = None

    try:
        cursor = conn.cursor()
        assert cursor is not None

        query = """
            UPDATE experiments
            SET status = %s, end_timestamp = %s, total_episodes = %s
            WHERE experiment_id = %s
        """

        cursor.execute(query, (status, datetime.now(), total_episodes, experiment_id))
        conn.commit()

        print(
            f"✅ Updated experiment {experiment_id}: status={status}, episodes={total_episodes}"
        )

    except Exception as e:
        conn.rollback()
        print(f"❌ Error updating experiment: {e}")
        raise

    finally:
        if cursor:
            cursor.close()
        pool.putconn(conn)


def log_training_metrics(experiment_id: int, metrics_data: Dict[str, Any]):
    """Log periodic training metrics."""
    pool = get_connection_pool()
    conn = pool.getconn()
    cursor: Optional[Cursor] = None

    try:
        cursor = conn.cursor()
        assert cursor is not None

        query = """
            INSERT INTO training_metrics (
                experiment_id, timestep, loss, mean_reward,
                mean_episode_length, mean_q_value
            ) VALUES (%s, %s, %s, %s, %s, %s)
        """

        cursor.execute(
            query,
            (
                experiment_id,
                metrics_data.get("timestep"),
                metrics_data.get("loss"),
                metrics_data.get("mean_reward"),
                metrics_data.get("mean_episode_length"),
                metrics_data.get("mean_q_value"),
            ),
        )

        conn.commit()

    except Exception as e:
        conn.rollback()
        print(f"❌ Error logging training metrics: {e}")
        raise

    finally:
        if cursor:
            cursor.close()
        pool.putconn(conn)
