#!/bin/bash

#SBATCH --job-name=test_simulation_runs
#SBATCH --output=simulation_runs_%j.out
#SBATCH -o simulation_runs_%j.out
#SBATCH -e error_simulation_runs_%j.err
#SBATCH --ntasks=5
#SBATCH --nodes=1
#SBATCH --partition=caslake
#SBATCH --mail-type=ALL
#SBATCH --mail-user=qxlin@rcc.uchicago.edu
#SBATCH --account=macs30123

# Load the necessary modules. Adjust these lines based on your cluster's setup.
module load python mpich 

echo "Cores,Time" > test_mpi_times.csv

for i in {1..5}
do
    mpirun -n $i python3 1b_mpi_simulation.py >> test_temp_time.out
    time=$(tail -n 1 test_temp_time.out)
    echo "$i,$time" >> test_mpi_times.csv
done
