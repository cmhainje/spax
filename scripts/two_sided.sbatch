#!/bin/bash
#SBATCH -J spax-two-sided
#SBATCH -p gen
#SBATCH -N 2
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --ntasks=80
#SBATCH -c 1
#SBATCH -t 1-00:00:00
#SBATCH -o /mnt/sdceph/users/chainje/spax/logs/two-sided-%j.out
#SBATCH --mail-type=all
#SBATCH --mail-user=ch4407@nyu.edu
#
# Two-sided per-bin A, B in one deep field (see scripts/two_sided.py), e.g.
#   sbatch scripts/two_sided.sbatch ndf
#   sbatch scripts/two_sided.sbatch sdf
# Pass the same field to resume: the output directory is named after the field and RUN.
# srun starts one process per task; task i (SLURM_PROCID) handles partitions[i::ntasks].
# Checkpointed per partition: resubmitting this script resumes where it left off
# (with any --ntasks; finished partitions are skipped by whichever task gets them).
# Slurm won't create the -o directory: mkdir -p .../spax/logs before the first submit.

set -euo pipefail
FIELD=$1
RUN=2026-10-26
OUT=/mnt/sdceph/users/chainje/spax/two-sided-$RUN-$FIELD
module load uv
cd ~/spax
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TALLTABLE_DB_DIR=/mnt/sdceph/users/spherex/talltable
echo "field $FIELD, out $OUT, host $(hostname), $SLURM_NTASKS tasks, start $(date)"

# sync the environment once, so the tasks don't all try to at the same time
uv sync
srun --label uv run --no-sync python -u scripts/two_sided.py "$OUT" --field "$FIELD"
echo "end $(date)"
