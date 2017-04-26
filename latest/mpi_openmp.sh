#!/bin/bash
#SBATCH -n 3  # Number of cores requested
#SBATCH -t 120   # Runtime in minutes
#SBATCH -c 2
#SBATCH -p general  
#SBATCH --mem=1000      # Memory per cpu in MB (see also --mem-per-cpu)
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --mail-type=END,FAIL    # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=gang_liu@g.harvard.edu        # Email to which notifications will be sent
#SBATCH -o mpiopenmp%j.out       # Standard out goes to this file
#SBATCH -e mpiopenmp%j.err       # Standard err goes to this filehostname

module load Anaconda/2.1.0-fasrc01
module load gcc
module load openmpi
#conda create -n ody --clone="$PYTHON_HOME"
#source activate ody
#conda install -c anaconda pygpu=0.6.4
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun -n $SLURM_NTASKS -c $SLURM_CPUS_PER_TASK --mpi=pmi2 python MPI+theano.py