#!/bin/bash

#SBATCH --mem-per-cpu=3000
#SBATCH -A research
#SBATCH -n 39
#SBATCH --gres=gpu:4
#SBATCH -t 4-00:00:00
#SBATCH --mail-type=NONE
#SBATCH --job-name="DeepSPInN Training"
#SBATCH --output=ir_nmr_file.txt

ulimit -n 40960

source /home2/sriram.devata/miniconda3/etc/profile.d/conda.sh
source /home2/sriram.devata/miniconda3/etc/profile.d/mamba.sh

rm -r /ssd_scratch/users/sriram.devata/deepspinn/
mamba create --prefix /ssd_scratch/users/sriram.devata/deepspinn python=3.9.6 --yes
mamba activate /ssd_scratch/users/sriram.devata/deepspinn
module load u18/cuda/10.2

pip install torch==1.9.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html
mamba install -c conda-forge ipdb --yes
mamba install -c dglteam dgl-cuda10.2 --yes
mamba install -c conda-forge rdkit --yes
pip install tqdm ray chemprop==1.3.0

cd /home2/sriram.devata/DeepSPInN/train
ray start --head --num-cpus=39 --num-gpus=4 --object-store-memory 50000000000
python parallel_agent.py

