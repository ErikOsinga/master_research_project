#!/bin/sh

# Assumes you have installed miniconda from
# https://conda.io/miniconda.html
# and you are in the miniconda directory

# This will create a conda directory called kerastf
# which will be a virtual environment

KERAS_BACKEND=tensorflow

conda create -n kerastf python=3.6 qt=5
source activate kerastf 
# change this to the location of pip-requirements.txt
pip install -r pip-requirements.txt

# I think you can also do
# pip install keras-gpu for a gpu version of keras

echo "Created conda env 'kerastf'. Recommended to add the following alias to your ~/.bashrc for convieniece:"
echo
echo "alias kerastf='source /path/to/miniconda/bin/activate kerastf'"