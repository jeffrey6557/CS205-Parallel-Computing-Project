#!/usr/bin/env python
from mpi4py import MPI
import numpy
import sys

comm = MPI.COMM_SELF.Spawn(sys.executable,
                           args=['NN_worker.py'],
                           maxprocs=1)

# load data
# Bcast/Scatter to workers

# initialize  theta1, theta2

# bcast( theta1 , theta2) to every worker

# cost_change = 1
# Loop until: cost_change < threshold
#   
#   ### let workers compute d_theta1, d_theta2
#   receive( d_theta1, d_theta2 )
#	update theta1, theta2 with d_theta1, d_theta2
#	compute cost_new 
# 	cost_change = cost_new - cost_last
#   send(theta1, theta2)



comm.Disconnect()