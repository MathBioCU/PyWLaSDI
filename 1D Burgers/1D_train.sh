#!/bin/bash 

### LSF syntax
#BSUB -nnodes 1                 #number of nodes
#BSUB -W 5:00                   #walltime in hours:minutes
#BSUB -G your_acount            #account
#BSUB -J name_job               #name of job
#BSUB -q queue                  #queue to use

### Shell scripting
date; hostname
echo -n '> JobID is '
echo $LSB_JOBID
ssh
source /your/anaconda/
module unload cuda
conda activate opence-1.8.0

python 1D_train.py
