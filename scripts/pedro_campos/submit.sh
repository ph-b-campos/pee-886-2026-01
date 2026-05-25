#!/bin/bash
#SBATCH --job-name=qml_pedro
#SBATCH --output=slurm_logs/qml_%j.out   
#SBATCH --error=slurm_logs/qml_%j.err    
#SBATCH --partition=cpu                 

source ~/miniconda3/etc/profile.d/conda.sh
conda activate qml-env

cd ~/Desktop/QML/pee-886-2026-01

# Força o uso do python que está DENTRO do ambiente qml-env
/home/pedro.campos/miniconda3/envs/qml-env/bin/python scripts/pedro_campos/run_experiment.py