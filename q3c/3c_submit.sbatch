#!/bin/bash
#SBATCH --job-name=3c_calculation
#SBATCH -o 3c_calculation_runs_%j.out
#SBATCH -e 3c_error_calculation_runs_%j.err
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --account=macs30123
#SBATCH --mail-type=ALL
#SBATCH --mail-user=qxlin@rcc.uchicago.edu
#SBATCH --mem-per-cpu=30G

module load python cuda

cd /home/qxlin/a1-QIXIN-ACT/q3c

python3 tile_calculation.py