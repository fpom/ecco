#!/usr/bin/bash

# venv where ecco will be installed
VENV="$HOME/.local/share/ecco"
# how to invoke pip
PIP3=pip3
# how to invoque python
PYTHON3=python3

# stop on error
set -ev

# 1. install system packages

sudo apt-get install tzdata apt-utils
# to build ecco and dependencies
sudo apt-get install gcc g++ make git openjdk-8-jre-headless
# to download libDDD
sudo apr-get install wget
# directly used by ecco
sudo apt-get install graphviz
# needed by Jupyter
sudo apt-get install nodejs npm
# Python, comment to use an installed version
sudo apt-get install python3 python3-dev python3-pip python3-venv
# libDDD: download prebuilt binaries and copy to /usr/local
wget -nv https://github.com/fpom/pyddd/raw/master/libDDD.tar.gz
sudo --chdir /usr/local tar xzf "$(realpath libDDD.tar.gz)"
sudo ldconfig
rm libDDD.tar.gz

# 2. install user packages

# for Jupyter
npm install -g configurable-http-proxy
# make venv if it does not exist
test -d "$VENV" || $PYTHON3 -m venv "$VENV"
. "$VENV/bin/activate"
# pip install dependencies
$PIP3 install setuptools wheel jupyterhub jupyter networkx pandas numpy bqplot colour bitarray sympy cython tatsu==5.5.0 psutil prince pydot python-igraph ipycytoscape unidecode scikit-learn typeguard rich[jupyter]
# ecofolder
git clone https://github.com/fpom/ecofolder.git ecofolder.git
cd ecofolder.git && make clean && make
cp -a ecofolder.git/ecofolder ecofolder.git/mci2dot ecofolder.git/mci2dot_ev ecofolder.git/pr_encoding ecofolder.git/mci2csv ecofolder.git/rs_complement "$VENV/bin/"
rm -rf ecofolder.git
# various dependencies
$PIP3 install git+https://github.com/fpom/pytl.git
$PIP3 install git+https://github.com/fpom/cunf-ptnet-py3.git
$PIP3 install git+https://github.com/fpom/pyddd.git
$PIP3 install git+https://github.com/fpom/pyits.git
$PIP3 install git+https://github.com/fpom/pymc.git

# 3. ecco

$PIP3 install git+https://github.com/fpom/ecco.git
