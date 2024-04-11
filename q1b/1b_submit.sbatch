#!/bin/bash

#SBATCH --job-name=1b_simulation
#SBATCH -o 1b_simulation_runs_%j.out
#SBATCH -e 1b_error_simulation_runs_%j.err
#SBATCH --ntasks=20
#SBATCH --nodes=1
#SBATCH --partition=caslake
#SBATCH --mail-type=ALL
#SBATCH --mail-user=qxlin@rcc.uchicago.edu
#SBATCH --account=macs30123

module load python mpich

cd /home/qxlin/a1-QIXIN-ACT/q1b

# Compile the health_index_simulation module
python compile_health_index.py

# Loop to run the simulations with increasing core counts
for i in {1..20}
do
    echo "Running on $i cores"
    mpirun -np $i python mpi_health_simulation.py >> timing_data_$i.out
done