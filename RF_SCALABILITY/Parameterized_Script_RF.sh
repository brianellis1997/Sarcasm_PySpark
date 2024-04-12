#!/bin/bash

#SBATCH --job-name=spark_job
#SBATCH --mem=128G                        
#SBATCH --time=3:00:00
#SBATCH --account=open
#SBATCH --mail-user=ump5025@psu.edu
#SBATCH --mail-type=BEGIN

# Load necessary modules
module load anaconda3
source activate ds410_sp24
module use /gpfs/group/RISE/sw7/modules
module load spark/3.3.0
export PYSPARK_PYTHON=python3
export PYSPARK_DRIVER_PYTHON=python3

# Run PySpark
start_time=$(date +%s)
spark-submit --deploy-mode client Modeling_RF.py
end_time=$(date +%s)

# Calculate and print the execution time
execution_time=$((end_time - start_time))
echo "$SLURM_NNODES, $SLURM_NTASKS_PER_NODE, $execution_time" >> execution_times_rf.csv