#!/bin/bash
#SBATCH --job-name=temporal_sim
#SBATCH --nodes=10               # Χρήση 10 κόμβων (για αρχή)
#SBATCH --ntasks-per-node=1      # 1 MPI task ανά GH200 GPU
#SBATCH --partition=debug        # Η partition που έχεις πρόσβαση
#SBATCH --time=00:10:00          # 10 λεπτά limit

module load daint-gpu            # Φόρτωση περιβάλλοντος (προσαρμογή ανάλογα το setup)
export MPICH_GPU_SUPPORT_ENABLED=1

srun python temporal_cluster_sim.py