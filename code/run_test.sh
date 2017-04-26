#!/bin/bash
#SBATCH -J test
#SBATCH -o test.out
#SBATCH -e test.err
#SBATCH -t 0-01:00
#SBATCH --constraint=cuda-7.5
#SBATCH -p holyseasgpu
#SBATCH --mem-per-cpu=4000
#SBATCH -N 2
#SBATCH -n 2
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --gres=gpu:2

##SBATCH --begin=now+10

module load Anaconda/2.1.0-fasrc01
#module load python/2.7.11-fasrc01
module load cuda/7.5-fasrc02
module load gcc/4.8.2-fasrc01 openmpi/1.10.2-fasrc01
source activate ody

srun -n $SLURM_NTASKS --mpi=pmi2 python MPI+theano.py