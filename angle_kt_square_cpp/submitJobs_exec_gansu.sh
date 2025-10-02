#!/bin/bash


for job_script in exec_*.sh; do
    echo "Submitting job script: $job_script"
    sbatch "$job_script"
    sleep 5
done
