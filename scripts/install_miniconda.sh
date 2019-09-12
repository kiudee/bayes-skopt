#!/usr/bin/env bash

set -e # fail on first error

if conda --version > /dev/null 2>&1; then
   echo "conda appears to already be installed"
   exit 0
 fi

PYTHON_VERSION=${PYTHON_VERSION:-3.6} # if no python specified, use 3.6

INSTALL_FOLDER="$HOME/miniconda"

if [ ! -d $INSTALL_FOLDER ] || [ ! -e $INSTALL_FOLDER/bin/conda ]; then
  echo "Downloading miniconda"
  wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;

  echo "Installing miniconda for python-$PYTHON_VERSION"
  bash miniconda.sh -b -f -p $INSTALL_FOLDER

  rm miniconda.sh
else
  echo "Miniconda already installed at ${INSTALL_FOLDER}.  Updating, adding to path and exiting"
fi

source "$HOME/miniconda/etc/profile.d/conda.sh"
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda
#Useful for debugging any issues with conda
conda info -a
