# GPflow extension for autoregressive GPs

## Project structure
Source code for for the algorithms can be found in `gpnarx`. End-to-end example for modeling a dynamical system can be found in `notebooks`.

## Environment
Conda environment can be set up the following way
```bash
conda create --name tensorflow-2.10.0 python==3.10
conda activate tensorflow-2.10.0
pip install -r requirements.txt
```
Global varaible `ABS_PATH` should be in `notebooks` replaced by your own path to the project folder.