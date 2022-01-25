#!/bin/bash
#BSUB -n 2
#BSUB -q general
#BSUB -G compute-crponce
#BSUB -J 'CMA_noise_ctrl[1-4]'
#BSUB -gpu "num=1:gmodel=TeslaV100_SXM2_32GB:mode=exclusive_process"
#BSUB -R 'gpuhost'
#BSUB -R 'select[mem>16G]'
#BSUB -R 'rusage[mem=16GB]'
#BSUB -M 16G
#BSUB -u binxu.wang@wustl.edu
#BSUB -o  /scratch1/fs1/crponce/CMA_covmat.%J.%I
#BSUB -a 'docker(pytorchlightning/pytorch_lightning:base-cuda-py3.9-torch1.9)'

echo "$LSB_JOBINDEX"

export TORCH_HOME="/scratch1/fs1/crponce/torch"
#export LSF_DOCKER_SHM_SIZE=16g
#export LSF_DOCKER_VOLUMES="$HOME:$HOME $SCRATCH1:$SCRATCH1 $STORAGE1:$STORAGE1"
param_list='--rep 25 --feval 3000
--rep 25 --feval 3000
--rep 25 --feval 3000
--rep 25 --feval 3000
'

export unit_name="$(echo "$param_list" | head -n $LSB_JOBINDEX | tail -1)"
echo "$unit_name"
# Append the extra command to the script.
cd ~/ActMax-Optimizer-Dev
python cma_noise_control.py  $unit_name

#!/usr/bin/env bash