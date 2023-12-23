#!/bin/bash

mamba create -n deepspinn python=3.9.6 --yes
mamba activate deepspinn

# module load u18/cuda/10.2

pip install torch==1.9.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html
mamba install -c conda-forge ipdb --yes
mamba install -c dglteam dgl-cuda10.2 --yes
mamba install -c conda-forge rdkit --only-deps --yes
pip install protobuf==3.20.*
pip install rdkit-pypi
pip install tqdm ray chemprop==1.3.0

