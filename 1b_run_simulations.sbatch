#!/bin/bash

#SBATCH --job-name=simulation_runs
#SBATCH --output=simulation_runs_%j.out
#SBATCH -o simulation_runs_%j.out
#SBATCH -e error_simulation_runs_%j.err
#SBATCH --ntasks=20
#SBATCH --nodes=1
#SBATCH --partition=caslake
#SBATCH --mail-type=ALL
#SBATCH --mail-user=qxlin@rcc.uchicago.edu
#SBATCH --account=macs30123

module load python mpich 

echo "Cores,Time" > mpi_times.csv

for i in {1..20}
do
    mpirun -n $i python3 1b_mpi_simulation.py >> temp_time.out
    time=$(tail -n 1 temp_time.out)
    echo "$i,$time" >> mpi_times.csv
done

