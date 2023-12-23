#!/bin/bash

#SBATCH --mem-per-cpu=3000
#SBATCH -A research
#SBATCH -n 40
#SBATCH --gres=gpu:4
#SBATCH -t 4-00:00:00
#SBATCH --mail-type=NONE
#SBATCH --job-name="DeepSPInN Testing"
#SBATCH --output=ir_nmr_file.txt

ulimit -n 40960

source "/home2/$USER/miniconda3/etc/profile.d/conda.sh"
source "/home2/$USER/miniconda3/etc/profile.d/mamba.sh"

rm -r "/scratch/$USER/deepspinn/"
mkdir -p "/scratch/$USER"
rm -r "/scratch/$USER/conda_pkgs_dirs"
mkdir -p "/scratch/$USER/conda_pkgs_dirs"
export CONDA_PKGS_DIRS="/scratch/$USER/conda_pkgs_dirs"

mamba create --prefix "/scratch/$USER/deepspinn" python=3.9.6 --yes
mamba activate "/scratch/$USER/deepspinn"
module load u18/cuda/10.2

pip install torch==1.9.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html
mamba install -c conda-forge ipdb --yes
mamba install -c dglteam dgl-cuda10.2 --yes
mamba install -c conda-forge rdkit --only-deps --yes
pip install protobuf==3.20.*
pip install rdkit-pypi
pip install tqdm ray chemprop==1.3.0

cd "/home2/$USER/DeepSPInN/test"
ray start --head --num-cpus=40 --num-gpus=4 --object-store-memory 50000000000
python parallel_agent.py -s 0 -e 1000 -d test

