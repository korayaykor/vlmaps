#!/bin/bash
# Downgrade pip to version compatible with pytorch-lightning 1.8.1 metadata
# Newer pip versions (24.1+) have stricter metadata validation
python -m pip install --upgrade "pip<24.1"

pip install -r requirements.txt

# Use mamba for faster dependency solving (drop-in replacement for conda)
# If mamba is not available, falls back to conda
if command -v mamba &> /dev/null; then
    mamba install habitat-sim headless -c conda-forge -c aihabitat -y
else
    # Use conda with solver options to speed up resolution
    conda install habitat-sim headless -c conda-forge -c aihabitat -y --solver=libmamba || \
    conda install habitat-sim headless -c conda-forge -c aihabitat -y
fi

cd ~
if [ ! -d "Hierarchical-Localization" ]; then
    git clone --recursive https://github.com/cvg/Hierarchical-Localization/
fi
cd Hierarchical-Localization/

# switch to a compatible version of hloc, see the commit: https://github.com/cvg/Hierarchical-Localization/commit/936040e8d67244cc6c8c9d1667701f3ce87bf075
git checkout 936040e8d67244cc6c8c9d1667701f3ce87bf075 

python -m pip install -e .
