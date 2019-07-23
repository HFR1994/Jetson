#!/usr/bin/env bash

set -e

sudo apt-get update && sudo apt-get upgrade
python3 -m venv venv
sudo pip3 install --upgrade pip
source venv/bin/activate
sudo pip3 install  --upgrade opencv-python re requests face-recognition
sudo pip3 install --upgrade setuptools ez_setup
sudo pip3 install --upgrade matplotlib
sudo pip3 install --upgrade configparser
sudo pip3 install --upgrade numpy
sudo apt-get install libblas-dev liblapack-dev libatlas-base-dev gfortran
sudo pip3 install --upgrade scipy
sudo pip3 install --upgrade pandas
sudo pip3 install --upgrade sklearn