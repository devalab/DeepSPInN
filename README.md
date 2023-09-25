# DeepSPInN
A framework that predicts the molecular structure when given Infrared and 13C Nuclear magnetic resonance spectra without referring to any pre-existing spectral databases or molecular fragment knowledge bases

### Setting up dependencies

The dependencies required for running this code can be installed by setting up an Anaconda environment.

```
conda env create -f custom_environment.yml
```

### Data

The data required for training and testing the code can be downloaded by following the instructions in the [data folder](data/README.md).

### Training Code

```
cd train
ray start --head --num-cpus=NUM_CPUS --num-gpus=NUM_GPUS --object-store-memory 50000000000
python parallel_agent.py
```

### Testing Code

```
cd test
ray start --head --num-cpus=NUM_CPUS --num-gpus=NUM_GPUS --object-store-memory 50000000000 
python parallel_agent.py -s MOL_START_INDEX -e MOL_END_INDEX
```

