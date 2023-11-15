mamba create --prefix ./deepspinn python=3.9.6
mamba activate ./deepspinn
pip install torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
mamba install -c conda-forge ipdb
mamba install -c dglteam dgl==0.6.1
mamba install -c conda-forge rdkit
pip install tqdm ray chemprop==1.3.0
