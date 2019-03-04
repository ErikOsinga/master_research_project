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



# > If you want to skip all above steps you can also copy Erik's conda environment
# > If you are on the sterrewacht only
# mkdir /data1/$USER/miniconda
# cd /data1/$USER/miniconda
# cp -r /net/reusel/data1/osinga/miniconda/* ./ 

# > Or maybe even use only a symbolic link (better)
# ln -s /net/reusel/data1/osinga/miniconda/ /data1/$USER/

# > To activate
# source /data1/$USER/miniconda/bin/activate kerastf