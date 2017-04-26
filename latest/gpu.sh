#!/bin/bash
#SBATCH -p serial_requeue #Partition to submit to 
#SBATCH -n 1 #Number of cores 
#SBATCH --gres=gpu:1
#SBATCH -t 500 #Runtime in minutes 
#SBATCH --mem-per-cpu=1000 #Memory per cpu in MB (see also --mem) 
#SBATCH --constraint=cuda-7.5
#SBATCH -o gpu%j.out      # File to which STDOUT will be written
#SBATCH -e gpu%j.err      # File to which STDERR will be written
#SBATCH --mail-type=ALL         # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=gang_liu@g.harvard.edu # Email to which notifications will be sent


module load Anaconda/2.1.0-fasrc01
conda create -n ody --clone="$PYTHON_HOME"
source activate ody
conda install -c anaconda pygpu=0.6.4
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python gpu_test.py