#!/bin/bash


for job_script in pkl_*.sh; do
    echo "Submitting job script: $job_script"
    sbatch "$job_script"
    sleep 5
done
