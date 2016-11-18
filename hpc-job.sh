#!/bin/sh
#PBS -N sparsemax-tables
#PBS -l walltime=03:00:00
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -m eba
#PBS -M amwebdk@gmail.com
#PBS -q k40_interactive

cd $PBS_O_WORKDIR

# Enable python3
export PYTHONPATH=
source ~/stdpy3/bin/activate

# rebuild
CUDA_VISIBLE_DEVICES=2 make clean all

# collect data and build tables
CUDA_VISIBLE_DEVICES=2 python3 benchmark/run_description.py
CUDA_VISIBLE_DEVICES=2 python3 benchmark/run_results.py
CUDA_VISIBLE_DEVICES=2 python3 benchmark/run_timings.py
