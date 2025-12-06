#!/bin/bash
#SBATCH --job-name=spark_standalone
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=spark-%j.out
#SBATCH --error=spark-%j.err
#SBATCH --mail-user=lfm5648@psu.edu
#SBATCH --mail-type=BEGIN,END

# -----------------------------
# Load modules and Python environment
# -----------------------------
module load anaconda3
module load jdk
source activate ds410_f25
module use /gpfs/group/RISE/sw7/modules
module load spark/3.3.0

export PYSPARK_PYTHON=python3
export PYSPARK_DRIVER_PYTHON=python3

# -----------------------------
# Spark worker + executor memory/cores
# -----------------------------
# Slurm gives you 64G per node; we'll give ~48G to Spark executors
# and leave the rest for OS / Python / overhead.
EXEC_MEM="48g"
DRIVER_MEM="8g"

export SPARK_EXECUTOR_MEMORY=$EXEC_MEM
export SPARK_DRIVER_MEMORY=$DRIVER_MEM
export SPARK_WORKER_MEMORY=$EXEC_MEM

# Total executor cores: nodes * tasks-per-node * cpus-per-task
TEC=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE * SLURM_CPUS_PER_TASK))

echo "total-executor-cores=${TEC}"
echo "executor-memory=${EXEC_MEM}"
echo "driver-memory=${DRIVER_MEM}"

# -----------------------------
# Spark logs in current directory
# -----------------------------
export SPARK_WORKER_DIR=$PWD/spark_work
mkdir -p "$SPARK_WORKER_DIR"

# -----------------------------
# Start Spark master
# -----------------------------
MASTER_NODE=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_URL="spark://$MASTER_NODE:7077"

srun --nodes=1 --ntasks=1 --exclusive \
     spark-class org.apache.spark.deploy.master.Master \
     --host "$MASTER_NODE" --port 7077 &

sleep 15

# -----------------------------
# Start Spark workers on all other nodes
# -----------------------------
for node in $(scontrol show hostnames "$SLURM_NODELIST"); do
    if [ "$node" != "$MASTER_NODE" ]; then
        srun --nodes=1 --ntasks=1 --nodelist="$node" --exclusive \
             spark-class org.apache.spark.deploy.worker.Worker \
               --cores "$SLURM_CPUS_PER_TASK" \
               --memory "$EXEC_MEM" \
               "$MASTER_URL" &
    fi
done

sleep 15

echo "Master URL: $MASTER_URL"

# -----------------------------
# Run your Spark job
# -----------------------------
start_time=$(date +%s)

"$SPARK_HOME"/bin/spark-submit \
  --master "$MASTER_URL" \
  --total-executor-cores "$TEC" \
  --executor-cores "$SLURM_CPUS_PER_TASK" \
  --executor-memory "$EXEC_MEM" \
  --driver-memory "$DRIVER_MEM" \
  Final_Project_Cluster.py

end_time=$(date +%s)
execution_time=$((end_time - start_time))
echo "Execution time: $execution_time seconds"
