#!/bin/bash
#SBATCH -n 1    # Number of cores requested
#SBATCH -t 30   # Runtime in minutes
#SBATCH -p      serial_requeue  # Partition to submit to
#SBATCH --mem=1000      # Memory per cpu in MB (see also --mem-per-cpu)
#SBATCH --open-mode=append
#SBATCH --mail-type=END,FAIL    # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=gang_liu@g.harvard.edu        # Email to which notifications will be sent
#SBATCH -o ADA_CPU_%j.out       # Standard out goes to this file
#SBATCH -e ADA_CPU_%j.err       # Standard err goes to this filehostname

module load Anaconda/2.1.0-fasrc01
conda create -n ody --clone="$PYTHON_HOME"
source activate ody
conda install -c anaconda pygpu=0.6.4
THEANO_FLAGS=device=cpu python ADA_GPU_CPU.py