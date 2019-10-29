#!/bin/bash

PYVERSION=$(python --version)

if [ "$PYVERSION" != "Python 3.6.5" ]; then
    echo Your python version is not 3.6.5, only python 3.6.5 is confirmed to work
fi

echo "Correct version of Python, 3.6.5, detacted."

echo Installing requirements
pip install -r requirements.txt 


echo Installing Bert
python -m deeppavlov install squad_bert



