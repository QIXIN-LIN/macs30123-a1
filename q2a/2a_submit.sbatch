#!/bin/bash

#SBATCH --job-name=2a_simulation
#SBATCH -o 2a_simulation_runs_%j.out
#SBATCH -e 2a_error_simulation_runs_%j.err
#SBATCH --ntasks=10
#SBATCH --nodes=1
#SBATCH --partition=caslake
#SBATCH --mail-type=ALL
#SBATCH --mail-user=qxlin@rcc.uchicago.edu
#SBATCH --account=macs30123

module load python mpich

cd /home/qxlin/a1-QIXIN-ACT/q2a

python health_simulation.py
mpirun -np $SLURM_NTASKS python mpi_grid_search.py