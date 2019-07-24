#!/usr/bin/env bash

set -e

yes |sudo apt-get update && sudo apt-get upgrade
yes | sudo pip3 install --upgrade pip
yes | sudo -H pip3 install  --upgrade setuptools ez_setup
yes | sudo -H pip3 install  --upgrade matplotlib
yes | sudo -H pip3 install  --upgrade configparser
yes | sudo -H pip3 install  --upgrade numpy
yes | sudo apt-get install libblas-dev liblapack-dev libatlas-base-dev gfortran
yes | sudo -H pip3 install  --upgrade scipy
yes | sudo -H pip3 install  --upgrade pandas
yes | sudo -H pip3 install --upgrade sklearn
yes | sudo -H pip3 install --upgrade opencv-python requests face_recognition
