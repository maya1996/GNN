#!/bin/bash
#SBATCH --job-name=train_gnn
#SBATCH --account=cis240108p
#SBATCH --partition=GPU-small
#SBATCH --gres=gpu:v100:1   
#SBATCH --mem=22000MB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mayasubra1996@gmail.com
#SBATCH --time=00:45:00
#SBATCH --output=train_gnn.out
#SBATCH --error=train_gnn.err

module purge
module load anaconda3
conda activate /ocean/projects/cis240108p/subramam/GNN/.env

cd /ocean/projects/cis240108p/subramam/GNN/GNN
python model_training.py
