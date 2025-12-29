CREATE TABLE experiments (
      experiment_id SERIAL primary key,
      experiment_name TEXT NOT NULL,
      algorithm text not null,
      status text not null,
      start_timestamp timestamp not null,
      end_timestamp timestamp,
      total_episodes integer,
      git_commit_hash text not null,
      python_version text not null,
      pytorch_version text not null,
      notes text
  );



 create table hyperparameters (
      hyperparameter_id SERIAL primary key,
      experiment_id integer references experiments(experiment_id),
      parameter_name text not null,
      parameter_value text not null
 );

create table episodes (
      episode_id serial primary key,
      experiment_id integer references experiments(experiment_id),
      episode_number integer not null,
      timestamp timestamp not null,
      reward numeric not null,
      episode_length integer not null,
      distance_traveled integer not null,
      level_completed boolean not null
);

create table training_metrics (
      training_metric_id serial primary key,
      experiment_id integer references experiments(experiment_id),
      training_step integer not null,
      timestamp timestamp not null,
      loss_value real not null,
      q_value_mean real not null,
      epsilon real not null,
      learning_rate real not null
); 

