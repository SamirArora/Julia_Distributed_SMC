#!/bin/bash
#SBATCH --job-name=distributed_resampling
#SBATCH --output=distributed_resampling_output_%j.log
#SBATCH --error=distributed_resampling_error_%j.log
#SBATCH --mail-user=samira@sfu.ca
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --account=def-liang-ab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --constraint=turin
#SBATCH --distribution=block:block
#SBATCH --hint=nomultithread
#SBATCH --exclusive

# Load Julia only
module load julia


# Run Julia
numactl --cpunodebind=0,1 --membind=0,1 \
julia --project=$HOME/scratch/distributed_smc distributed_resample_experiment.jl
