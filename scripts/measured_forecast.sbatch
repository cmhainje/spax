#!/bin/bash
#SBATCH -J spax-forecast
#SBATCH -p gen
#SBATCH -N 4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --ntasks=160
#SBATCH -c 1
#SBATCH -t 12:00:00
#SBATCH -o /mnt/sdceph/users/chainje/spax/logs/forecast-%j.out
# the scan of scripts/measured_forecast.py; task i handles chunks[i::ntasks], one file per
# chunk: resubmit to resume
set -euo pipefail
module load uv
cd ~/spax
export TALLTABLE_DB_DIR=/mnt/sdceph/users/spherex/talltable
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
srun uv run python -u scripts/measured_forecast.py
