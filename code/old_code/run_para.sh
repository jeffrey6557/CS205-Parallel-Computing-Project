#!/bin/bash
#SBATCH --mem-per-cpu=2000 #Memory per cpu in MB (see also --mem) 
#SBATCH -o out/test_%j.out      # File to which STDOUT will be written
#SBATCH -e err/test_%j.err      # File to which STDERR will be written
#SBATCH --mail-type=ALL         # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=linglin_huang@g.harvard.edu # Email to which notifications will be sent
#SBATCH -p serial_requeue #Partition to submit to 
#SBATCH -t 60 #Runtime in minutes 
#SBATCH -n 3 #Number of MPI tasks 
#SBATCH -c 2 #Number of cores per task
#SBATCH --ntasks-per-node=2 #Number of mpi tasks per node

source new-modules.sh
module load python/2.7.6-fasrc01
module load Anaconda/2.1.0-fasrc01 #contains Cython and mpi4py

# Set OMP_NUM_THREADS to the same value as -c
# with a fallback in case it isn't set.
# SLURM_CPUS_PER_TASK is set to the value of -c, but only if -c is explicitly set
if [ -n "$SLURM_CPUS_PER_TASK" ]; then
  omp_threads=$SLURM_CPUS_PER_TASK
else
  omp_threads=1
fi
export OMP_NUM_THREADS=$omp_threads

# --cpu_bind=rank_ldom is only possible if a complete node is
# allocated.
# This allocates one MPI task with its 6 OpenMP cores
# per NUMA unit.
# srun --cpu_bind=rank_ldom ./mpi_openmp_program

# Test mpi4py
# srun -n 2 --mpi=pmi2 python mpi4py_test.py

# Test mpi * openmp
python setup.py build_ext --inplace
srun -n 3 --mpi=pmi2 python MPI_final.py


