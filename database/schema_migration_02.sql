-- Migration 02: Add multi-stage tracking columns
-- These columns support tracking progress across multiple stages in a level

-- max_x_pos: Highest total distance reached during the episode (accumulated across stages)
ALTER TABLE episodes ADD COLUMN max_x_pos INTEGER;

-- final_x_pos: Total distance at episode end (accumulated across stages)
ALTER TABLE episodes ADD COLUMN final_x_pos INTEGER;

-- Note: level_completed column already exists, but semantics changed:
-- Previously: based on flag_get (touching the flag)
-- Now: based on stage transitions (moving to a new stage)
-- This is a semantic change, not a schema change.
