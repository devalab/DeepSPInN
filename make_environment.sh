mamba create --prefix ./deepspinn python=3.9.6
mamba activate ./deepspinn
# pip install torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
module load u18/cuda/10.2
pip install torch==1.9.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html
mamba install -c conda-forge ipdb
# mamba install -c dglteam dgl==0.6.1
mamba install -c dglteam dgl-cuda10.2
mamba install -c conda-forge rdkit
pip install tqdm ray chemprop==1.3.0
