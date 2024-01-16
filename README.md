# DeepSPInN
A framework that predicts the molecular structure when given Infrared and 13C Nuclear magnetic resonance spectra without referring to any pre-existing spectral databases or molecular fragment knowledge bases

### Setting up dependencies

The dependencies required for running this code can be installed by setting up an Anaconda environment.

```
conda env create -f custom_environment.yml
```

If making an environment from the .yml file does not work, the script `make_environment.sh` contains the commands to set up a minimal environment that worked on an environment with CUDA 10.2. If it is being run on a HPC with a Slurm workload manager, the CUDA 10.2 toolkit can be loaded with the `module load...` command that is commented out in the script.

### Data

The data required for training and testing the code can be downloaded by following the instructions in the [data folder](data/README.md).

### Training Code

```
cd train
ray start --head --num-cpus=39 --num-gpus=4 --object-store-memory 50000000000
python parallel_agent.py
```

### Testing Code

```
cd test
ray start --head --num-cpus=40 --num-gpus=4 --object-store-memory 50000000000 
python parallel_agent.py -s MOL_START_INDEX -e MOL_END_INDEX -d test
```

`MOL_START_INDEX` and `MOL_END_INDEX` are the start and end indices in the test dataset.

For testing on a HPC with a Slurm workload manager, the script `test/sbatch_test.sh` is a good starting point for an SBATCH script that tests the code for 1000 molecules in the test set. 

The script creates log files in the folder `test/test_outputs/`. These log files can be analysed with the script `test/test_outputs/check_results.py`.
