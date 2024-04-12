#!/bin/bash
#SBATCH --job-name=spark_job          # Job name
#SBATCH --nodes=4                     # Number of nodes to request
#SBATCH --ntasks-per-node=4           # Number of processes per node
#SBATCH --mem=8G                     # Adjusted Memory per node, consider increasing if available
#SBATCH --time=6:00:00                # Maximum runtime in HH:MM:SS
#SBATCH --account=open                # Queue
#SBATCH --mail-user=bje5256@psu.edu   # Your email
#SBATCH --mail-type=BEGIN             # Email notifications

# Load necessary modules (if required)
module load anaconda3
source activate ds410_f23
module use /gpfs/group/RISE/sw7/modules
module load spark/3.3.0
export PYSPARK_PYTHON=python3
export PYSPARK_DRIVER_PYTHON=python3

# # Adjust Spark executor memory and other configurations for optimization
# # Consider adjusting these values based on the cluster's available resources and your job's requirements
# export SPARK_SUBMIT_OPTS="-Dspark.driver.memory=8g -Dspark.executor.memory=8g -Dspark.executor.cores=2 -Dspark.task.cpus=1 -Dspark.executor.instances=4 -Dspark.dynamicAllocation.enabled=false -Dspark.shuffle.service.enabled=false"

# Optionally, set additional Spark configurations for optimization
# Uncomment and adjust the following line as needed
# export SPARK_SUBMIT_OPTS="${SPARK_SUBMIT_OPTS} -Dspark.default.parallelism=100 -Dspark.sql.shuffle.partitions=100"

# Run PySpark
# Record the start time
start_time=$(date +%s)

spark-submit --deploy-mode client Testing_Pipeline.py

# Record the end time
end_time=$(date +%s)

# Calculate and print the execution time
execution_time=$((end_time - start_time))
echo "Execution time: $execution_time seconds"