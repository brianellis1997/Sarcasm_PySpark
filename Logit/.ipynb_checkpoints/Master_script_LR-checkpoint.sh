#!/bin/bash

# Initialize csv file as such:
echo "nodes, tasks_per_node, execution_time" > execution_times_logit.csv

# Array of nodes to test
nodes_arr=(1 2 4 6 8)
# Array of tasks per node to test
tasks_per_node_arr=(1 2 4 8)

# Loop over each node count
for nodes in "${nodes_arr[@]}"
do
  # Loop over each tasks per node count
  for tasks_per_node in "${tasks_per_node_arr[@]}"
  do
    # Submit the job with current nodes and tasks per node settings
    sbatch --nodes=$nodes --ntasks-per-node=$tasks_per_node Parameterized_Script_logit.sh
    # Optional: wait a bit between submissions to not overload the scheduler
    sleep 5
  done
done