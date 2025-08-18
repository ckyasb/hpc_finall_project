#!/bin/bash

export TMPDIR=~/tmp
mkdir -p $TMPDIR

# Corrected command: Removed the extra "report_name"
srun -p V100 --gpus 1 /usr/local/cuda/bin/ncu --export tiling ./winograd inputs/config.txt
