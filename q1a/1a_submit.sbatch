#!/bin/bash

#SBATCH --job-name=1a_simulation
#SBATCH -o 1a_simulation_runs_%j.out
#SBATCH -e 1a_error_simulation_runs_%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=caslake
#SBATCH --mail-type=ALL
#SBATCH --mail-user=qxlin@rcc.uchicago.edu
#SBATCH --account=macs30123

module load python 

cd /home/qxlin/a1-QIXIN-ACT/q1a

python compile_health_index.py
python run_compiled_simulation.py

python run_simulation.py