#!/bin/bash
#BSUB -n 2
#BSUB -q general
#BSUB -G compute-crponce
#BSUB -J 'CMA_covmat[1-7]'
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
param_list='--units alexnet .features.ReLU4 1 13 13  --chan_rng 0 25 --rep 1 --feval 3000
--units alexnet .features.ReLU7 1 6 6  --chan_rng 0 25 --rep 1 --feval 3000
--units alexnet .features.ReLU9 1 6 6  --chan_rng 0 25 --rep 1 --feval 3000
--units alexnet .features.ReLU11 1 6 6  --chan_rng 0 25 --rep 1 --feval 3000
--units alexnet .classifier.ReLU2 1  --chan_rng 0 25 --rep 1 --feval 3000
--units alexnet .classifier.ReLU5 1  --chan_rng 0 25 --rep 1 --feval 3000
--units alexnet .classifier.Linear6 1  --chan_rng 0 25 --rep 1 --feval 3000
--units resnet50_linf8 .layer1.Bottleneck2  1 28 28  --chan_rng 0 10 --rep 1 --feval 3000
--units resnet50_linf8 .layer2.Bottleneck3  1 14 14  --chan_rng 0 10 --rep 1 --feval 3000
--units resnet50_linf8 .layer3.Bottleneck5  1 7 7  --chan_rng 0 10 --rep 1 --feval 3000
--units resnet50_linf8 .layer4.Bottleneck2  1 4 4  --chan_rng 0 10 --rep 1 --feval 3000
--units resnet50_linf8 .Linearfc 1  --chan_rng 0 10 --rep 1 --feval 3000
'
#--units alexnet .features.ReLU4 1 --chan_rng 0 10 --rep 5 --noise_lvl 0.5 --feval 3000 conv2
#--units alexnet .features.ReLU7 1 --chan_rng 0 10 --rep 5 --noise_lvl 0.5 --feval 3000 conv3
#--units alexnet .features.ReLU9 1 --chan_rng 0 10 --rep 5 --noise_lvl 0.5 --feval 3000 conv4
#--units alexnet .features.ReLU11 1 --chan_rng 0 10 --rep 5 --noise_lvl 0.5 --feval 3000 conv5
#--units alexnet .classifier.ReLU2 1 --chan_rng 0 10 --rep 5 --noise_lvl 0.5 --feval 3000 fc6
#--units alexnet .classifier.ReLU5 1 --chan_rng 0 10 --rep 5 --noise_lvl 0.5 --feval 3000 fc7
#--units alexnet .classifier.Linear6 1 --chan_rng 0 10 --rep 5 --noise_lvl 0.5 --feval 3000 fc8

export unit_name="$(echo "$param_list" | head -n $LSB_JOBINDEX | tail -1)"
echo "$unit_name"
# Append the extra command to the script.
cd ~/ActMax-Optimizer-Dev
python cma_covmat_record.py  $unit_name

#!/usr/bin/env bash