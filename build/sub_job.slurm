#!/bin/bash

#SBATCH --account=CSC143
#SBATCH --job-name=test_xgc
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH -o test_xgc.%J

ulimit -n 40960

srun -n1 ./test_xgc

