#!/bin/bash

echo "Preparing play"
python setup.py build_ext --inplace &&
echo "Playing"
python capture.py