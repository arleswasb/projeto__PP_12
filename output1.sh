#!/bin/bash
#SBATCH --job-name=multithreading_example
#SBATCH --time=0-1:0
#SBATCH --partition=amd-512
#SBATCH --cpus-per-task=64
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK


pascalanalyzer ./navier_stokes_otm -c 1,2,4,8,16,32,64 -i 32,64,128,512,1024,2048,4096 --inst aut -o output1.json 
