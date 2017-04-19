#!/bin/bash
#SBATCH -n 1    # Number of cores requested
#SBATCH -t 600   # Runtime in minutes
#SBATCH -p gpu
#SBATCH --mem 5000 
#SBATCH --gres=gpu:1
#SBATCH --constraint=cuda-7.5
#SBATCH -o Out%j.out       # Standard out goes to this file
#SBATCH -e Err%j.err       # Standard err goes to this filehostname


module load python/2.7.11-fasrc01
source activate ody
module load gcc/4.9.3-fasrc01 cuda/7.5-fasrc02 cudnn/7.0-fasrc02





# to set device, in script 
import os
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['GOTO_NUM_THREADS'] = '8'
os.environ['OMP_NUM_THREADS'] = '8'
os.environ["THEANO_FLAGS"] = "device=gpu,floatX=float32"
# terminal command
THEANO_FLAGS='floatX=float32,device=cpu,lib.cnmem=1,openmp=1'  python <myscript>.py

# get into an iteractive gpu node and acitvate virtual environment ody 
srun --pty --mem 4000 --gres gpu:1 -t 1200 -p gpu /bin/bash
module load python/2.7.11-fasrc01
source activate ody

# update cmake to > 3.0
wget --no-check-certificate http://www.cmake.org/files/v3.8/cmake-3.8.0.tar.gz 
tar xzf cmake-3.8.0.tar.gz
cd cmake-3.8.0
./configure --prefix=/opt/cmake
make
make install
/opt/cmake/bin/cmake -version

# speedups over implementations
seq = [302,53,35,35,23,21,20,19]
cpu = [295,39,25,18,13,13,12]
gpu = [73,37,24,18,15,13,12,11]



Training time 302.584539175 seconds.
Training time 53.3988828659 seconds.
Training time 35.7516560555 seconds.
Training time 35.8533699512 seconds.
Training time 23.496876955 seconds.
Training time 21.2230210304 seconds.
Training time 20.1605038643 seconds.
Training time 19.3286709785 seconds.
fastest batch_size is 2048
Training time 20.7856659889 seconds.
Test MSE: 1.063060619


Training time 295.031805992 seconds.
Training time 39.2692260742 seconds.
Training time 25.0488519669 seconds.
Training time 18.6898150444 seconds.
Training time 15.7333071232 seconds.
Training time 13.9079339504 seconds.
Training time 13.1158969402 seconds.
Training time 12.8139388561 seconds.
fastest batch_size is 2048
Training time 9.73022007942 seconds.
Test MSE: 0.809448948213


Training time 73.8043010235 seconds.
Training time 37.9162008762 seconds.
Training time 24.7067160606 seconds.
Training time 18.4121701717 seconds.
Training time 15.0236859322 seconds.
Training time 13.0703530312 seconds.
Training time 12.1615319252 seconds.
Training time 11.7693269253 seconds.
fastest batch_size is 2048
Training time 9.50624489784 seconds.
Test MSE: 0.892133107033


