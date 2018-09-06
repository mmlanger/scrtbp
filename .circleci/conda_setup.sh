#!/bin/bash

if [ ! -d $HOME/miniconda/envs/circleci ]; then
    # Download miniconda
    if [ "$PYTHON_VERSION" = "2.7" ]; then
        wget http://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh -O miniconda.sh
    else
        wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    fi

    chmod +x miniconda.sh

    # Install miniconda to folder in home directory
    ./miniconda.sh -b -p $HOME/miniconda

    # Create environment
    $HOME/miniconda/bin/conda create -n circleci --yes python=$PYTHON_VERSION \
        numpy numba pytest flake8
fi