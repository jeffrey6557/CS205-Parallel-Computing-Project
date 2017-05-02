#!/bin/bash
for i in 4 8 16 32;
do

	echo "#!/bin/bash
#SBATCH -n ${i}  # Number of cores requested
#SBATCH -t 30   # Runtime in minutes
#SBATCH -c 1
#SBATCH -p general #seas_iacs #general  
#SBATCH --mem=1000      # Memory per cpu in MB (see also --mem-per-cpu)
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --mail-type=END,FAIL    # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=linglin_huang@g.harvard.edu        # Email to which notifications will be sent
#SBATCH -o mpi_pso_${i}n1c_%j.out       # Standard out goes to this file
#SBATCH -e mpi_pso_${i}n1c_%j.err       # Standard err goes to this filehostname

module load Anaconda/2.1.0-fasrc01
module load gcc
module load openmpi
source activate ody
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
srun -n \$SLURM_NTASKS -c \$SLURM_CPUS_PER_TASK --mpi=pmi2 python MPI_pso.py
" > run_${i}.sh

sbatch run_${i}.sh
rm run_${i}.sh

done

